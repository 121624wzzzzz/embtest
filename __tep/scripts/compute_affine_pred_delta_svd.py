#!/usr/bin/env python3
"""Compute decomposition SVD diagnostics for Task6 BI pairs.

This is a read-only follow-up diagnostic for the affine notes. It does not
overwrite Task6 outputs; it writes a derived CSV under __tep/affine/tables/.

It compares:
- raw delta:          (Y-X)^T(Y-X)            (already in Task6 CSV)
- centered delta:     (Y_c-X_c)^T(Y_c-X_c)
- affine main term:   (A-I)^T X_c^T X_c (A-I)
- residual term:      (Y_c-X_c A)^T(Y_c-X_c A)
- mean shift term:    n (mu_Y-mu_X)^T(mu_Y-mu_X)
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

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "ijcai_clean" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ijcai_clean.data import load_info_json  # noqa: E402
from ijcai_clean.experiments.full_vocab_affine import (  # noqa: E402
    BATCH_ROWS,
    svd_energy_from_gram,
)

TASK6_CSV = (
    REPO_ROOT
    / "ijcai_clean"
    / "results"
    / "task6_base_instruct_full_vocab"
    / "summary_pair_base_instruct_full_vocab.csv"
)
OUT_CSV = (
    REPO_ROOT
    / "__tep"
    / "affine"
    / "tables"
    / "affine_task6_decomposition_svd.csv"
)

ANOMALY_SERIES = {"Gemma-4"}
ANOMALY_MODELS = {"Gemma-3-1B"}


def is_anomaly(model_a: str) -> bool:
    return model_a in ANOMALY_MODELS or model_a.startswith("Gemma-4-")


def series_of(model_a: str) -> str:
    if model_a.startswith("Qwen3.5"):
        return "Qwen3.5"
    if model_a.startswith("Qwen3-"):
        return "Qwen3"
    if model_a.startswith("Qwen2.5"):
        return "Qwen2.5"
    if model_a.startswith("Llama"):
        return "Llama"
    if model_a.startswith("Gemma-2"):
        return "Gemma-2"
    if model_a.startswith("Gemma-3"):
        return "Gemma-3"
    if model_a.startswith("Gemma-4"):
        return "Gemma-4"
    return model_a.split("-", 1)[0]


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


def prefixed(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in values.items()}


def compute_decomposition_metrics(
    X_np: np.ndarray,
    Y_np: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_rows: int,
) -> dict[str, Any]:
    n, d = X_np.shape
    if Y_np.shape != X_np.shape:
        raise ValueError(f"shape mismatch: {X_np.shape} vs {Y_np.shape}")

    sum_x = np.zeros(d, dtype=np.float64)
    sum_y = np.zeros(d, dtype=np.float64)
    for start in range(0, n, batch_rows):
        end = min(n, start + batch_rows)
        sum_x += X_np[start:end].astype(np.float64, copy=False).sum(axis=0)
        sum_y += Y_np[start:end].astype(np.float64, copy=False).sum(axis=0)

    mx = torch.from_numpy((sum_x / n).astype(np.float32)).to(device=device, dtype=dtype)
    my = torch.from_numpy((sum_y / n).astype(np.float32)).to(device=device, dtype=dtype)
    G = torch.zeros((d, d), device=device, dtype=dtype)
    C = torch.zeros((d, d), device=device, dtype=dtype)
    centered_delta_gram = torch.zeros((d, d), device=device, dtype=dtype)

    for start in range(0, n, batch_rows):
        end = min(n, start + batch_rows)
        x = as_tensor(X_np[start:end], device, dtype) - mx
        y = as_tensor(Y_np[start:end], device, dtype) - my
        centered_delta = y - x
        G.add_(x.T @ x)
        C.add_(x.T @ y)
        centered_delta_gram.add_(centered_delta.T @ centered_delta)
        del x, y, centered_delta

    try:
        A = torch.linalg.solve(G, C)
        solver = "solve"
    except Exception:
        A = torch.linalg.lstsq(G, C, rcond=None).solution
        solver = "lstsq"

    delta_a = A - torch.eye(d, device=device, dtype=dtype)
    a_delta_gram = delta_a.T @ delta_a
    a_delta_gram = (a_delta_gram + a_delta_gram.T) * 0.5
    pred_gram = delta_a.T @ (G @ delta_a)
    pred_gram = (pred_gram + pred_gram.T) * 0.5

    residual_gram = torch.zeros((d, d), device=device, dtype=dtype)
    for start in range(0, n, batch_rows):
        end = min(n, start + batch_rows)
        x = as_tensor(X_np[start:end], device, dtype) - mx
        y = as_tensor(Y_np[start:end], device, dtype) - my
        residual = y - x @ A
        residual_gram.add_(residual.T @ residual)
        del x, y, residual

    mean_delta = my - mx
    mean_shift_gram = n * torch.outer(mean_delta, mean_delta)
    raw_delta_gram = centered_delta_gram + mean_shift_gram

    out: dict[str, Any] = {"solver": solver}
    out.update(prefixed("Delta", svd_energy_from_gram(raw_delta_gram, prefix="svd")))
    out.update(prefixed("A_delta", svd_energy_from_gram(a_delta_gram, prefix="svd")))
    out.update(prefixed("Centered_delta", svd_energy_from_gram(centered_delta_gram, prefix="svd")))
    out.update(prefixed("Pred_delta", svd_energy_from_gram(pred_gram, prefix="svd")))
    out.update(prefixed("Residual", svd_energy_from_gram(residual_gram, prefix="svd")))
    out.update(prefixed("Mean_shift", svd_energy_from_gram(mean_shift_gram, prefix="svd")))
    out["centered_delta_total_energy"] = float(centered_delta_gram.trace().item())
    out["raw_delta_total_energy"] = float(raw_delta_gram.trace().item())
    out["a_delta_total_energy"] = float(a_delta_gram.trace().item())
    out["pred_delta_total_energy"] = float(pred_gram.trace().item())
    out["residual_total_energy"] = float(residual_gram.trace().item())
    out["mean_shift_total_energy"] = float(mean_shift_gram.trace().item())
    out["residual_over_centered_delta_energy"] = (
        out["residual_total_energy"] / out["centered_delta_total_energy"]
        if out["centered_delta_total_energy"]
        else float("nan")
    )
    out["mean_shift_over_raw_delta_energy"] = (
        out["mean_shift_total_energy"]
        / (out["centered_delta_total_energy"] + out["mean_shift_total_energy"])
        if out["centered_delta_total_energy"] + out["mean_shift_total_energy"]
        else float("nan")
    )

    del (
        G,
        C,
        A,
        delta_a,
        a_delta_gram,
        pred_gram,
        centered_delta_gram,
        raw_delta_gram,
        residual_gram,
        mean_shift_gram,
        mx,
        my,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return out


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="debug: only run first N rows")
    parser.add_argument("--models", nargs="*", default=None, help="only these model_a ids")
    parser.add_argument("--include-anomalies", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-rows", type=int, default=BATCH_ROWS)
    parser.add_argument("--matrix-kind", choices=["embed", "lm_head"], default="embed")
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    rows = list(csv.DictReader(TASK6_CSV.open(newline="", encoding="utf-8")))
    if not args.include_anomalies:
        rows = [r for r in rows if not is_anomaly(r["model_a"])]
    if args.models:
        wanted = set(args.models)
        rows = [r for r in rows if r["model_a"] in wanted]
    if args.limit:
        rows = rows[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float32
    out_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, 1):
        model_a = row["model_a"]
        model_b = row["model_b"]
        print(f"[{idx}/{len(rows)}] {args.matrix_kind} {model_a} -> {model_b}", flush=True)
        X = load_matrix(model_a, args.matrix_kind)
        Y = load_matrix(model_b, args.matrix_kind)
        metrics = compute_decomposition_metrics(
            X,
            Y,
            device=device,
            dtype=dtype,
            batch_rows=args.batch_rows,
        )
        h = int(row["hidden_dim_a"])
        raw_rank95 = float(metrics["Delta_svd_rank_95"])
        pred_rank95 = float(metrics["Pred_delta_svd_rank_95"])
        centered_rank95 = float(metrics["Centered_delta_svd_rank_95"])
        residual_rank95 = float(metrics["Residual_svd_rank_95"])
        a_rank95 = float(metrics["A_delta_svd_rank_95"])
        raw_eff = float(metrics["Delta_svd_effective_rank"])
        pred_eff = float(metrics["Pred_delta_svd_effective_rank"])
        centered_eff = float(metrics["Centered_delta_svd_effective_rank"])
        residual_eff = float(metrics["Residual_svd_effective_rank"])
        a_eff = float(metrics["A_delta_svd_effective_rank"])
        reported_r2 = row["E_R2"] if args.matrix_kind == "embed" else row["U_R2"]
        out = {
            "model_a": model_a,
            "model_b": model_b,
            "matrix_kind": args.matrix_kind,
            "series": series_of(model_a),
            "hidden_dim": h,
            "is_anomaly": is_anomaly(model_a),
            "reported_affine_R2": reported_r2,
            "Delta_rank95_over_h": raw_rank95 / h,
            "Centered_delta_rank95_over_h": centered_rank95 / h,
            "A_delta_rank95_over_h": a_rank95 / h,
            "Pred_delta_rank95_over_h": pred_rank95 / h,
            "Residual_rank95_over_h": residual_rank95 / h,
            "Delta_eff_over_h": raw_eff / h,
            "Centered_delta_eff_over_h": centered_eff / h,
            "A_delta_eff_over_h": a_eff / h,
            "Pred_delta_eff_over_h": pred_eff / h,
            "Residual_eff_over_h": residual_eff / h,
            "Delta_energy_5pct_h": metrics["Delta_svd_energy_at_5pct_h"],
            "Centered_delta_energy_5pct_h": metrics[
                "Centered_delta_svd_energy_at_5pct_h"
            ],
            "A_delta_energy_5pct_h": metrics["A_delta_svd_energy_at_5pct_h"],
            "Pred_delta_energy_5pct_h": metrics["Pred_delta_svd_energy_at_5pct_h"],
            "Residual_energy_5pct_h": metrics["Residual_svd_energy_at_5pct_h"],
            "Mean_shift_energy_5pct_h": metrics["Mean_shift_svd_energy_at_5pct_h"],
            "Delta_rank_95": metrics["Delta_svd_rank_95"],
            "Pred_delta_rank_95": metrics["Pred_delta_svd_rank_95"],
            "Delta_effective_rank": metrics["Delta_svd_effective_rank"],
            "Pred_delta_effective_rank": metrics["Pred_delta_svd_effective_rank"],
            "Delta_total_energy": metrics["raw_delta_total_energy"],
            "Pred_delta_total_energy": metrics["pred_delta_total_energy"],
            "Centered_delta_total_energy": metrics["centered_delta_total_energy"],
            "Residual_total_energy": metrics["residual_total_energy"],
            "Mean_shift_total_energy": metrics["mean_shift_total_energy"],
            "Residual_over_centered_delta_energy": metrics[
                "residual_over_centered_delta_energy"
            ],
            "Mean_shift_over_raw_delta_energy": metrics["mean_shift_over_raw_delta_energy"],
            "solver": metrics["solver"],
        }
        out_rows.append(out)
        with args.out.open("w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        print(
            "  rank95/h raw={:.3f} centered={:.3f} pred={:.3f} A={:.3f}; "
            "res/centered={:.3f} mean/raw={:.3f}".format(
                out["Delta_rank95_over_h"],
                out["Centered_delta_rank95_over_h"],
                out["Pred_delta_rank95_over_h"],
                out["A_delta_rank95_over_h"],
                float(out["Residual_over_centered_delta_energy"]),
                float(out["Mean_shift_over_raw_delta_energy"]),
            ),
            flush=True,
        )
        del X, Y

    def mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else math.nan

    main_rows = [r for r in out_rows if not r["is_anomaly"]]
    print("\nSUMMARY main rows n=", len(main_rows), flush=True)
    for key in (
        "Delta_rank95_over_h",
        "Centered_delta_rank95_over_h",
        "Pred_delta_rank95_over_h",
        "A_delta_rank95_over_h",
        "Delta_energy_5pct_h",
        "Centered_delta_energy_5pct_h",
        "Pred_delta_energy_5pct_h",
        "A_delta_energy_5pct_h",
        "Residual_over_centered_delta_energy",
        "Mean_shift_over_raw_delta_energy",
    ):
        print(f"  {key}: {mean([float(r[key]) for r in main_rows]):.6f}", flush=True)
    try:
        out_label = args.out.resolve().relative_to(REPO_ROOT)
    except ValueError:
        out_label = args.out
    print(f"\nWROTE {out_label}", flush=True)


if __name__ == "__main__":
    main()
