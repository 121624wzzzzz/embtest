#!/usr/bin/env python3
"""Compare low-rank affine adapters with parameter-matched token-delta adapters.

For each BI pair, evaluate two approximations with nearly equal parameter count:

1. Hidden-space affine/LoRA adapter
   Y_hat_c = X_c + X_c B_r, rank(B_r) <= r_aff
   params ~= h + 2 h r_aff (bias + U,V)

2. Token-level delta adapter
   Y_hat_c = X_c + D_r, rank(D_r) <= r_delta
   params ~= h + r_delta (n + h) (mean shift + SVD factors)

The hidden adapter's best rank-r effect is computed from the spectrum of
G_p = (A-I)^T X_c^T X_c (A-I), because affine residuals are orthogonal to
col(X_c). The token delta baseline uses the spectrum of
G_cd = (Y_c-X_c)^T(Y_c-X_c).
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
SRC_ROOT = REPO_ROOT / "ijcai_clean" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ijcai_clean.data import load_info_json  # noqa: E402
from ijcai_clean.experiments.full_vocab_affine import BATCH_ROWS  # noqa: E402

TASK6_CSV = (
    REPO_ROOT
    / "ijcai_clean"
    / "results"
    / "task6_base_instruct_full_vocab"
    / "summary_pair_base_instruct_full_vocab.csv"
)
OUT_CSV = ROOT / "affine" / "tables" / "affine_param_matched_adapter_eval.csv"


DEFAULT_MODELS = [
    "Qwen3.5-0.8B-Base",
    "Qwen3-0.6B-Base",
    "Llama-3.2-1B",
    "Gemma-2-2B",
]


def is_anomaly(model_a: str) -> bool:
    return model_a == "Gemma-3-1B" or model_a.startswith("Gemma-4-")


def load_embed(model: str) -> np.ndarray:
    extracts = REPO_ROOT / "extracts"
    info = load_info_json(extracts, model)
    src = info.get("standardized_sources") or {}
    embed_key = src.get("embed") or "model.embed_tokens.weight"
    st_path = extracts / f"{model}.safetensors"
    with safe_open(st_path, framework="pt", device="cpu") as f:
        return f.get_tensor(embed_key).float().cpu().numpy()


def as_tensor(arr: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(
        device=device, dtype=dtype
    )


def descending_eigvalsh(M: torch.Tensor) -> torch.Tensor:
    eig = torch.linalg.eigvalsh((M + M.T) * 0.5)
    return torch.clamp(eig, min=0.0).flip(0)


def energy_at_rank(eigvals: torch.Tensor, rank: int) -> float:
    total = float(eigvals.sum().item())
    if total <= 0:
        return 0.0
    if rank <= 0:
        return 0.0
    rank = max(1, min(rank, int(eigvals.numel())))
    return float(eigvals[:rank].sum().item()) / total


def closest_token_rank(param_budget_no_bias: int, n: int, h: int) -> int:
    raw = param_budget_no_bias / (n + h)
    candidates = {max(1, math.floor(raw)), max(1, math.ceil(raw)), max(1, round(raw))}
    return min(candidates, key=lambda r: abs(r * (n + h) - param_budget_no_bias))


def floor_token_rank(param_budget_no_bias: int, n: int, h: int) -> int:
    return max(0, math.floor(param_budget_no_bias / (n + h)))


def compute_pair(
    row: dict[str, str],
    *,
    rank_fracs: list[float],
    device: torch.device,
    dtype: torch.dtype,
    batch_rows: int,
) -> list[dict[str, Any]]:
    model_a = row["model_a"]
    model_b = row["model_b"]
    X_np = load_embed(model_a)
    Y_np = load_embed(model_b)
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
    Gai = B.T @ B
    Gai = (Gai + Gai.T) * 0.5

    eig_p = descending_eigvalsh(Gp)
    eig_cd = descending_eigvalsh(Gcd)
    eig_ai = descending_eigvalsh(Gai)
    pred_energy = float(eig_p.sum().item())
    centered_energy = float(eig_cd.sum().item())
    residual_energy = max(0.0, centered_energy - pred_energy)

    identity_r2 = 1.0 - centered_energy / y_center_energy
    full_affine_r2 = 1.0 - residual_energy / y_center_energy
    out_rows: list[dict[str, Any]] = []
    for rank_frac in rank_fracs:
        r_aff = max(1, math.ceil(rank_frac * h))
        aff_params_no_bias = 2 * h * r_aff
        r_delta = closest_token_rank(aff_params_no_bias, n, h)
        r_delta_floor = floor_token_rank(aff_params_no_bias, n, h)
        aff_params = h + aff_params_no_bias
        delta_params = h + r_delta * (n + h)
        delta_floor_params = h + r_delta_floor * (n + h)

        c_pred = energy_at_rank(eig_p, r_aff)
        c_ai = energy_at_rank(eig_ai, r_aff)
        c_delta = energy_at_rank(eig_cd, r_delta)
        c_delta_floor = energy_at_rank(eig_cd, r_delta_floor)
        c_delta_same_rank = energy_at_rank(eig_cd, r_aff)

        aff_gain_update = (pred_energy / centered_energy) * c_pred
        delta_gain_update = c_delta
        delta_floor_gain_update = c_delta_floor
        aff_r2 = identity_r2 + (centered_energy / y_center_energy) * aff_gain_update
        delta_r2 = identity_r2 + (centered_energy / y_center_energy) * delta_gain_update
        delta_floor_r2 = (
            identity_r2 + (centered_energy / y_center_energy) * delta_floor_gain_update
        )

        out_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "hidden_dim": h,
                "vocab": n,
                "rank_frac": rank_frac,
                "rank_affine": r_aff,
                "rank_delta_param_matched": r_delta,
                "rank_delta_no_overbudget": r_delta_floor,
                "affine_params_with_bias": aff_params,
                "delta_params_with_bias": delta_params,
                "delta_no_overbudget_params_with_bias": delta_floor_params,
                "delta_over_affine_params": delta_params / aff_params,
                "delta_no_overbudget_over_affine_params": delta_floor_params / aff_params,
                "full_hh_params": h * h,
                "affine_params_over_full_hh": aff_params / (h * h),
                "identity_R2": identity_r2,
                "full_affine_R2": full_affine_r2,
                "reported_full_affine_R2": float(row["E_R2"]),
                "rank_affine_R2": aff_r2,
                "param_matched_delta_R2": delta_r2,
                "no_overbudget_delta_R2": delta_floor_r2,
                "rank_affine_update_gain": aff_gain_update,
                "param_matched_delta_update_gain": delta_gain_update,
                "no_overbudget_delta_update_gain": delta_floor_gain_update,
                "rank_affine_gain_over_delta_gain": (
                    aff_gain_update / delta_gain_update
                    if delta_gain_update > 0
                    else float("inf")
                ),
                "rank_affine_gain_over_no_overbudget_delta_gain": (
                    aff_gain_update / delta_floor_gain_update
                    if delta_floor_gain_update > 0
                    else float("inf")
                ),
                "rank_affine_fraction_of_full_affine_gain": c_pred,
                "A_I_energy_at_rank_affine": c_ai,
                "pred_energy_at_rank_affine": c_pred,
                "delta_energy_at_param_matched_rank": c_delta,
                "delta_energy_at_no_overbudget_rank": c_delta_floor,
                "delta_energy_at_same_rank_not_param_matched": c_delta_same_rank,
                "full_affine_update_gain": pred_energy / centered_energy,
                "centered_delta_over_Yc_energy": centered_energy / y_center_energy,
            }
        )

    del X_np, Y_np, S, C, Gcd, A, B, Gp, Gai, eig_p, eig_cd, eig_ai, mx, my
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--all-main", action="store_true")
    parser.add_argument("--rank-fracs", type=float, nargs="+", default=[0.05])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-rows", type=int, default=BATCH_ROWS)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    with TASK6_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.all_main:
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
        print(f"[{idx}/{len(selected)}] {row['model_a']} -> {row['model_b']}", flush=True)
        out_rows.extend(
            compute_pair(
                row,
                rank_fracs=args.rank_fracs,
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
        print(
            f"{row['model_a']}: rank_aff={row['rank_affine']} "
            f"rank_frac={row['rank_frac']:.4f} "
            f"rank_delta={row['rank_delta_param_matched']} "
            f"params_ratio={row['delta_over_affine_params']:.3f} "
            f"R2 aff={row['rank_affine_R2']:.6f} "
            f"delta={row['param_matched_delta_R2']:.6f} "
            f"gain aff={row['rank_affine_update_gain']:.6f} "
            f"delta={row['param_matched_delta_update_gain']:.6f} "
            f"gain_ratio={row['rank_affine_gain_over_delta_gain']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
