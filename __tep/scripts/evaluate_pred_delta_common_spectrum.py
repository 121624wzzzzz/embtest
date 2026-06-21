#!/usr/bin/env python3
"""Export common-k spectra for centered delta and affine prediction.

For each Base/Instruct pair:

    D = Y_c - X_c
    P = X_c (A - I)

where A is the centered least-squares affine map.  The output reports
cumulative top-k energy for D, P, and R (= Y_c - X_c A), plus the amount of complete delta
energy explained by the top-k affine prediction directions.
"""

from __future__ import annotations

import argparse
import csv
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
OUT_CSV = ROOT / "affine" / "tables" / "affine_pred_delta_common_spectrum.csv"

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


def cumulative_energy(eigvals: torch.Tensor, ranks: list[int]) -> dict[int, float]:
    total = float(eigvals.sum().item())
    if total <= 0:
        return {rank: 0.0 for rank in ranks}
    cumsum = torch.cumsum(eigvals, dim=0)
    out: dict[int, float] = {}
    for rank in ranks:
        rank_clipped = min(max(1, rank), int(eigvals.numel()))
        out[rank] = float(cumsum[rank_clipped - 1].item()) / total
    return out


def compute_pair(
    row: dict[str, str],
    *,
    matrix_kind: str,
    ranks: list[int],
    device: torch.device,
    dtype: torch.dtype,
    batch_rows: int,
) -> dict[str, Any]:
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

    Gr = torch.zeros((h, h), device=device, dtype=dtype)
    for start in range(0, n, batch_rows):
        end = min(n, start + batch_rows)
        x = as_tensor(X_np[start:end], device, dtype) - mx
        y = as_tensor(Y_np[start:end], device, dtype) - my
        r = y - x @ A
        Gr.add_(r.T @ r)
        del x, y, r
    Gr = (Gr + Gr.T) * 0.5

    eig_delta = descending_eigvalsh(Gcd)
    eig_pred = descending_eigvalsh(Gp)
    eig_resid = descending_eigvalsh(Gr)
    delta_energy = float(eig_delta.sum().item())
    pred_energy = float(eig_pred.sum().item())
    full_affine_gain = pred_energy / delta_energy if delta_energy > 0 else 0.0
    identity_r2 = 1.0 - delta_energy / y_center_energy
    full_affine_r2 = identity_r2 + (delta_energy / y_center_energy) * full_affine_gain

    delta_c = cumulative_energy(eig_delta, ranks)
    pred_c = cumulative_energy(eig_pred, ranks)
    resid_c = cumulative_energy(eig_resid, ranks)

    out: dict[str, Any] = {
        "model_a": model_a,
        "model_b": model_b,
        "matrix_kind": matrix_kind,
        "hidden_dim": h,
        "vocab": n,
        "identity_R2": identity_r2,
        "full_affine_R2": full_affine_r2,
        "delta_over_Yc_energy": delta_energy / y_center_energy,
        "full_affine_gain_over_delta": full_affine_gain,
    }
    for rank in ranks:
        out[f"D_energy_at_{rank}"] = delta_c[rank]
        out[f"P_energy_at_{rank}"] = pred_c[rank]
        out[f"R_energy_at_{rank}"] = resid_c[rank]
        out[f"P_abs_delta_energy_at_{rank}"] = full_affine_gain * pred_c[rank]

    del X_np, Y_np, S, C, Gcd, Gr, A, B, Gp, eig_delta, eig_pred, eig_resid, mx, my
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument(
        "--all-clean",
        action="store_true",
        help="run all 30 BI-clean pairs (35 registered pairs minus 5 excluded anomalies)",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-rows", type=int, default=BATCH_ROWS)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    parser.add_argument("--matrix-kind", choices=["embed", "lm_head"], default="embed")
    args = parser.parse_args()

    ranks = sorted(set(args.ranks))
    if not ranks or min(ranks) < 1:
        raise SystemExit("Ranks must be positive integers")

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
        out_rows.append(
            compute_pair(
                row,
                matrix_kind=args.matrix_kind,
                ranks=ranks,
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
