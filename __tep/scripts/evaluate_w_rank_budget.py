#!/usr/bin/env python3
"""Evaluate hidden-affine adapters under a W-form LoRA rank budget.

The budget owner is the token/W-form low-rank update

    Y_hat_c = X_c + D_r, rank(D_r) <= r_w

with parameter count h + r_w (vocab + h).  For the hidden affine adapter

    Y_hat_c = X_c + X_c B_r, rank(B_r) <= r_aff

we choose the largest rank that does not exceed the same non-bias budget:

    r_aff = floor(r_w (vocab + h) / (2 h)).

This answers the practical question: if a W-form LoRA uses rank 1, 2, ...
what hidden-affine rank can be bought by the same parameter budget, and which
approximation explains more of the Base->Instruct update?
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SRC_ROOT = REPO_ROOT / "cross_model_geometry" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cross_model_geometry.data import load_info_json  # noqa: E402
from cross_model_geometry.experiments.full_vocab_affine import BATCH_ROWS  # noqa: E402

TASK6_CSV = (
    REPO_ROOT
    / "cross_model_geometry"
    / "results"
    / "task6_base_instruct_full_vocab"
    / "summary_pair_base_instruct_full_vocab.csv"
)
OUT_CSV = ROOT / "affine" / "tables" / "e" / "affine_w_rank_budget_clean.csv"

DEFAULT_MODELS = [
    "Qwen3.5-0.8B-Base",
    "Qwen3-0.6B-Base",
    "Llama-3.2-1B",
    "Gemma-2-2B",
]


def is_anomaly(model_a: str) -> bool:
    return model_a == "Gemma-3-1B" or model_a.startswith("Gemma-4-")


def load_matrix(model: str, matrix_kind: str) -> np.ndarray:
    extracts = REPO_ROOT / "extracts"
    info = load_info_json(extracts, model)
    src = info.get("standardized_sources") or {}
    if matrix_kind == "embed":
        tensor_key = src.get("embed") or "model.embed_tokens.weight"
    elif matrix_kind == "lm_head":
        tensor_key = src.get("lm_head") or "lm_head.weight"
    else:
        raise ValueError(f"unknown matrix kind: {matrix_kind}")
    st_path = extracts / f"{model}.safetensors"
    with safe_open(st_path, framework="pt", device="cpu") as f:
        return f.get_tensor(tensor_key).float().cpu().numpy()


def as_tensor(arr: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(
        device=device, dtype=dtype
    )


def descending_eigvalsh(M: torch.Tensor) -> torch.Tensor:
    eig = torch.linalg.eigvalsh((M + M.T) * 0.5)
    return torch.clamp(eig, min=0.0).flip(0)


def energy_at_rank(eigvals: torch.Tensor, rank: int) -> float:
    total = float(eigvals.sum().item())
    if total <= 0 or rank <= 0:
        return 0.0
    rank = min(rank, int(eigvals.numel()))
    return float(eigvals[:rank].sum().item()) / total


def affine_rank_from_w_budget(rank_w: int, n: int, h: int) -> int:
    rank = math.floor(rank_w * (n + h) / (2 * h))
    return min(h, max(1, rank))


def compute_pair(
    row: dict[str, str],
    *,
    matrix_kind: str,
    w_ranks: list[int],
    device: torch.device,
    dtype: torch.dtype,
    batch_rows: int,
) -> list[dict[str, Any]]:
    model_a = row["model_a"]
    model_b = row["model_b"]
    X_np = load_matrix(model_a, matrix_kind)
    Y_np = load_matrix(model_b, matrix_kind)
    n, h = X_np.shape
    if Y_np.shape != X_np.shape:
        raise ValueError(f"shape mismatch: {X_np.shape} vs {Y_np.shape}")

    sum_x = np.zeros(h, dtype=np.float64)
    sum_y = np.zeros(h, dtype=np.float64)
    y2_sum = 0.0
    for start in range(0, n, batch_rows):
        end = min(n, start + batch_rows)
        xb = X_np[start:end].astype(np.float64, copy=False)
        yb = Y_np[start:end].astype(np.float64, copy=False)
        sum_x += xb.sum(axis=0)
        sum_y += yb.sum(axis=0)
        y2_sum += float((yb * yb).sum())

    mean_x = sum_x / n
    mean_y = sum_y / n
    y_center_energy = y2_sum - n * float(mean_y @ mean_y)
    mx = torch.from_numpy(mean_x.astype(np.float32)).to(device=device, dtype=dtype)
    my = torch.from_numpy(mean_y.astype(np.float32)).to(device=device, dtype=dtype)

    S = torch.zeros((h, h), device=device, dtype=dtype)
    C = torch.zeros((h, h), device=device, dtype=dtype)
    Gcd = torch.zeros((h, h), device=device, dtype=dtype)
    for start in range(0, n, batch_rows):
        end = min(n, start + batch_rows)
        x = as_tensor(X_np[start:end], device, dtype) - mx
        y = as_tensor(Y_np[start:end], device, dtype) - my
        d = y - x
        S.add_(x.T @ x)
        C.add_(x.T @ y)
        Gcd.add_(d.T @ d)
        del x, y, d

    A = torch.linalg.solve(S, C)
    B = A - torch.eye(h, device=device, dtype=dtype)
    Gp = B.T @ (S @ B)
    Gp = (Gp + Gp.T) * 0.5

    eig_p = descending_eigvalsh(Gp)
    eig_cd = descending_eigvalsh(Gcd)
    pred_energy = float(eig_p.sum().item())
    centered_energy = float(eig_cd.sum().item())
    residual_energy = max(0.0, centered_energy - pred_energy)

    mean_delta = mean_y - mean_x
    mean_shift_energy = n * float(mean_delta @ mean_delta)
    raw_delta_energy = centered_energy + mean_shift_energy
    mean_over_raw = (
        mean_shift_energy / raw_delta_energy if raw_delta_energy > 0 else float("nan")
    )
    centered_over_raw = (
        centered_energy / raw_delta_energy if raw_delta_energy > 0 else float("nan")
    )

    identity_r2 = 1.0 - centered_energy / y_center_energy
    full_affine_r2 = 1.0 - residual_energy / y_center_energy
    full_affine_update_gain = pred_energy / centered_energy if centered_energy > 0 else 0.0
    centered_delta_over_yc = centered_energy / y_center_energy
    full_affine_gain_over_raw = (
        mean_over_raw + centered_over_raw * full_affine_update_gain
        if raw_delta_energy > 0
        else float("nan")
    )

    out_rows: list[dict[str, Any]] = []
    for rank_w in w_ranks:
        rank_aff = affine_rank_from_w_budget(rank_w, n, h)
        w_params = h + rank_w * (n + h)
        affine_params = h + 2 * h * rank_aff

        c_pred = energy_at_rank(eig_p, rank_aff)
        c_w = energy_at_rank(eig_cd, rank_w)
        affine_gain = full_affine_update_gain * c_pred
        w_gain = c_w
        affine_gain_raw = mean_over_raw + centered_over_raw * affine_gain
        w_gain_raw = mean_over_raw + centered_over_raw * w_gain
        affine_r2 = identity_r2 + centered_delta_over_yc * affine_gain
        w_r2 = identity_r2 + centered_delta_over_yc * w_gain

        out_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "matrix_kind": matrix_kind,
                "hidden_dim": h,
                "vocab": n,
                "rank_w": rank_w,
                "rank_affine_budgeted": rank_aff,
                "rank_affine_over_h": rank_aff / h,
                "w_params_with_bias": w_params,
                "affine_params_with_bias": affine_params,
                "affine_over_w_params": affine_params / w_params,
                "identity_R2": identity_r2,
                "full_affine_R2": full_affine_r2,
                "reported_full_affine_R2": float(
                    row["E_R2"] if matrix_kind == "embed" else row["U_R2"]
                ),
                "w_rank_R2": w_r2,
                "rank_affine_R2": affine_r2,
                "w_rank_update_gain": w_gain,
                "rank_affine_update_gain": affine_gain,
                "w_rank_update_gain_over_raw": w_gain_raw,
                "rank_affine_update_gain_over_raw": affine_gain_raw,
                "rank_affine_gain_over_w_gain": (
                    affine_gain / w_gain if w_gain > 0 else float("inf")
                ),
                "rank_affine_gain_over_w_gain_raw": (
                    affine_gain_raw / w_gain_raw if w_gain_raw > 0 else float("inf")
                ),
                "rank_affine_fraction_of_full_affine_gain": c_pred,
                "w_delta_energy_at_rank": c_w,
                "full_affine_update_gain": full_affine_update_gain,
                "full_affine_update_gain_over_raw": full_affine_gain_over_raw,
                "mean_shift_over_raw_delta": mean_over_raw,
                "centered_delta_over_raw_delta": centered_over_raw,
                "raw_delta_over_Yc_energy": raw_delta_energy / y_center_energy,
                "centered_delta_over_Yc_energy": centered_delta_over_yc,
            }
        )

    del X_np, Y_np, S, C, Gcd, A, B, Gp, eig_p, eig_cd, mx, my
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument(
        "--all-clean",
        action="store_true",
        help="run all 30 BI-clean pairs (35 registered pairs minus 5 excluded anomalies)",
    )
    parser.add_argument("--w-ranks", type=int, nargs="+")
    parser.add_argument("--w-rank-max", type=int, default=128)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-rows", type=int, default=BATCH_ROWS)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    parser.add_argument("--matrix-kind", choices=["embed", "lm_head"], default="embed")
    args = parser.parse_args()

    if args.w_ranks:
        w_ranks = sorted(set(args.w_ranks))
    else:
        w_ranks = list(range(1, args.w_rank_max + 1))
    if not w_ranks or min(w_ranks) < 1:
        raise SystemExit("W ranks must be positive integers")

    with TASK6_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.all_clean:
        selected = [r for r in rows if not is_anomaly(r["model_a"])]
    else:
        wanted = set(args.models)
        selected = [r for r in rows if r["model_a"] in wanted]
    if not selected:
        raise SystemExit("No selected models found in Task6 CSV")

    device = torch.device(args.device)
    dtype = torch.float32
    out_rows = []
    for idx, row in enumerate(selected, 1):
        print(
            f"[{idx}/{len(selected)}] {args.matrix_kind} "
            f"{row['model_a']} -> {row['model_b']}",
            flush=True,
        )
        out_rows.extend(
            compute_pair(
                row,
                matrix_kind=args.matrix_kind,
                w_ranks=w_ranks,
                device=device,
                dtype=dtype,
                batch_rows=args.batch_rows,
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"WROTE {args.out}")
    for row in out_rows:
        if row["rank_w"] in {1, 2, 4, 8, 16, 32, 64, 128}:
            print(
                f"{row['model_a']}: rW={row['rank_w']} "
                f"rAff={row['rank_affine_budgeted']} "
                f"aff/W params={row['affine_over_w_params']:.3f} "
                f"gain aff={row['rank_affine_update_gain']:.6f} "
                f"W={row['w_rank_update_gain']:.6f} "
                f"ratio={row['rank_affine_gain_over_w_gain']:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
