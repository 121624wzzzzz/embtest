#!/usr/bin/env python3
"""Evaluate a decomposed hybrid of hidden-affine LoRA plus W-form residual LoRA.

The centered update is decomposed as

    D = Y_c - X_c = P + R,   P = X_c (A - I),

where A is the centered least-squares affine map and R is orthogonal to the
full affine prediction.  The hybrid adapter uses

    P_a  : rank-a approximation to P, implemented by hidden affine LoRA
    R_q  : rank-q approximation to R, implemented by W-form LoRA

with parameter count h + 2 h a + q (vocab + h), sharing one mean/bias term.

For each pure W-form baseline rank t, the script reports:
- pure W top-t delta energy,
- best decomposed hybrid under the same parameter budget,
- minimum hybrid parameters needed to match the pure W top-t energy.
"""

from __future__ import annotations

import argparse
import csv
import sys
from bisect import bisect_left
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
OUT_CSV = ROOT / "affine" / "tables" / "e" / "affine_hybrid_w_budget_clean.csv"

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


def cumulative_abs(eigvals: torch.Tensor, denom: float) -> list[float]:
    vals = [0.0]
    if denom <= 0:
        vals.extend([0.0] * int(eigvals.numel()))
        return vals
    cumsum = torch.cumsum(eigvals, dim=0).detach().cpu().numpy()
    vals.extend((cumsum / denom).astype(float).tolist())
    return vals


def at_rank(cumsum: list[float], rank: int) -> float:
    rank = min(max(0, rank), len(cumsum) - 1)
    return cumsum[rank]


def first_rank_at_least(cumsum: list[float], target: float) -> int | None:
    if target <= 0:
        return 0
    idx = bisect_left(cumsum, target)
    if idx >= len(cumsum):
        return None
    return idx


def compute_pair(
    row: dict[str, str],
    *,
    matrix_kind: str,
    baseline_ranks: list[int],
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
    Gd = torch.zeros((h, h), device=device, dtype=dtype)
    for start in range(0, n, batch_rows):
        end = min(n, start + batch_rows)
        x = as_tensor(X_np[start:end], device, dtype) - mx
        y = as_tensor(Y_np[start:end], device, dtype) - my
        d = y - x
        S.add_(x.T @ x)
        C.add_(x.T @ y)
        Gd.add_(d.T @ d)
        del x, y, d

    A = torch.linalg.solve(S, C)
    B = A - torch.eye(h, device=device, dtype=dtype)
    Gp = B.T @ (S @ B)
    Gp = (Gp + Gp.T) * 0.5
    Gr = (Gd - Gp)
    Gr = (Gr + Gr.T) * 0.5

    eig_d = descending_eigvalsh(Gd)
    eig_p = descending_eigvalsh(Gp)
    eig_r = descending_eigvalsh(Gr)
    delta_energy = float(eig_d.sum().item())
    pred_energy = float(eig_p.sum().item())
    residual_energy = float(eig_r.sum().item())
    y_delta_ratio = delta_energy / y_center_energy
    identity_r2 = 1.0 - y_delta_ratio
    full_affine_gain = pred_energy / delta_energy if delta_energy > 0 else 0.0
    residual_gain = residual_energy / delta_energy if delta_energy > 0 else 0.0

    d_abs = cumulative_abs(eig_d, delta_energy)
    p_abs = cumulative_abs(eig_p, delta_energy)
    r_abs = cumulative_abs(eig_r, delta_energy)

    out_rows: list[dict[str, Any]] = []
    for rank_w in baseline_ranks:
        w_params = h + rank_w * (n + h)
        budget_no_bias = w_params - h
        w_gain = at_rank(d_abs, rank_w)

        best_same: dict[str, Any] | None = {
            "same_budget_hybrid_gain": w_gain,
            "same_budget_rank_affine": 0,
            "same_budget_rank_residual_w": rank_w,
            "same_budget_hybrid_params": w_params,
            "same_budget_mode": "pure_W_fallback",
        }
        max_q_same = min(rank_w, len(r_abs) - 1)
        for q in range(max_q_same + 1):
            remain = budget_no_bias - q * (n + h)
            if remain < 0:
                continue
            a = min(h, remain // (2 * h))
            gain = at_rank(p_abs, a) + at_rank(r_abs, q)
            params = h + 2 * h * a + q * (n + h)
            cand = {
                "same_budget_hybrid_gain": gain,
                "same_budget_rank_affine": a,
                "same_budget_rank_residual_w": q,
                "same_budget_hybrid_params": params,
                "same_budget_mode": "decomposed_hybrid",
            }
            if gain > best_same["same_budget_hybrid_gain"]:
                best_same = cand

        # Minimum decomposed-hybrid params to match W-rank-t gain.  Pure W rank t
        # is included as a fallback with param ratio 1.
        best_match: dict[str, Any] = {
            "match_hybrid_params": w_params,
            "match_rank_affine": 0,
            "match_rank_residual_w": rank_w,
            "match_hybrid_gain": w_gain,
            "match_mode": "pure_W_fallback",
        }
        max_q_match = min(rank_w, len(r_abs) - 1)
        for q in range(max_q_match + 1):
            residual_part = at_rank(r_abs, q)
            need_p = w_gain - residual_part
            a = first_rank_at_least(p_abs, need_p)
            if a is None:
                continue
            params = h + 2 * h * a + q * (n + h)
            if params < best_match["match_hybrid_params"]:
                best_match = {
                    "match_hybrid_params": params,
                    "match_rank_affine": a,
                    "match_rank_residual_w": q,
                    "match_hybrid_gain": at_rank(p_abs, a) + residual_part,
                    "match_mode": "decomposed_hybrid",
                }

        same_gain = best_same["same_budget_hybrid_gain"]
        out_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "matrix_kind": matrix_kind,
                "hidden_dim": h,
                "vocab": n,
                "baseline_rank_w": rank_w,
                "baseline_w_params": w_params,
                "baseline_w_gain": w_gain,
                "identity_R2": identity_r2,
                "delta_over_Yc_energy": y_delta_ratio,
                "full_affine_gain_over_delta": full_affine_gain,
                "residual_gain_over_delta": residual_gain,
                **best_same,
                "same_budget_gain_over_w": same_gain / w_gain if w_gain > 0 else float("inf"),
                "same_budget_param_ratio": best_same["same_budget_hybrid_params"] / w_params,
                **best_match,
                "match_param_ratio_over_w": best_match["match_hybrid_params"] / w_params,
                "match_param_saving": 1.0 - best_match["match_hybrid_params"] / w_params,
            }
        )

    del X_np, Y_np, S, C, Gd, A, B, Gp, Gr, eig_d, eig_p, eig_r, mx, my
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
    parser.add_argument("--baseline-ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-rows", type=int, default=BATCH_ROWS)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    parser.add_argument("--matrix-kind", choices=["embed", "lm_head"], default="embed")
    args = parser.parse_args()

    baseline_ranks = sorted(set(args.baseline_ranks))
    if not baseline_ranks or min(baseline_ranks) < 1:
        raise SystemExit("Baseline ranks must be positive integers")

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
                baseline_ranks=baseline_ranks,
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


if __name__ == "__main__":
    main()
