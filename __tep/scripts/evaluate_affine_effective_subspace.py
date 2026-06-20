#!/usr/bin/env python3
"""Validate whether raw ΔW = Y-X lies in the effective affine subspace of Z=[X, 1_n].

Metrics (E/U separate):
  - R²_aff       : full column-space projection onto span(Z)
  - R²_aff,K     : projection onto top-K left singular directions of Z (η=0.90/0.95/0.99)
  - r_L (L=1,5,10): spectral concentration of ΔW
  - a_1(K), A_{L,K}: rank-1 and rank-L weighted affine alignment with Q_K
  - R²_aff,K^(L) = r_L * A_{L,K}: top-L contribution to total affine explainability
  - a_bias, B_L  : bias-type energy in rank-1 / top-L left singular vectors
  - random baseline K/n
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SRC_ROOT = REPO_ROOT / "ijcai_clean" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ijcai_clean.data import load_info_json  # noqa: E402

TASK6_CSV = (
    REPO_ROOT
    / "ijcai_clean"
    / "results"
    / "task6_base_instruct_full_vocab"
    / "summary_pair_base_instruct_full_vocab.csv"
)
DEFAULT_OUT = REPO_ROOT / "bi_analysis" / "tables" / "affine_effective_subspace.csv"

ETA_THRESHOLDS = (0.90, 0.95, 0.99)
RANK_L_LEVELS = (5, 10)


def is_anomaly(model_a: str) -> bool:
    return model_a == "Gemma-3-1B" or model_a.startswith("Gemma-4-")


def load_matrix(model: str, matrix_kind: str) -> torch.Tensor:
    extracts = REPO_ROOT / "extracts"
    info = load_info_json(extracts, model)
    src = info.get("standardized_sources") or {}
    key = (
        src.get("embed") or "model.embed_tokens.weight"
        if matrix_kind == "embed"
        else src.get("lm_head") or "lm_head.weight"
    )
    with safe_open(extracts / f"{model}.safetensors", framework="pt", device="cpu") as f:
        return f.get_tensor(key).float()


def k_for_eta(singular_values: torch.Tensor, eta: float) -> int:
    e = singular_values.square()
    total = e.sum()
    if total <= 0:
        return 1
    cum = torch.cumsum(e, dim=0) / total
    idx = int(torch.searchsorted(cum, torch.tensor(eta, device=cum.device)).item())
    return min(max(1, idx + 1), singular_values.numel())


def rank_energy_fraction(singular_values: torch.Tensor, dw_norm_sq: float, m: int) -> float:
    if dw_norm_sq <= 0:
        return 0.0
    m = min(m, singular_values.numel())
    return float(singular_values[:m].square().sum().item() / dw_norm_sq)


def rank_l_alignment_metrics(
    Qk: torch.Tensor,
    U_dw: torch.Tensor,
    S_dw: torch.Tensor,
    dw_norm_sq: float,
    *,
    l_max: int,
) -> dict[str, float]:
    """Compute A_{L,K} and R²_aff,K^(L) for each L in RANK_L_LEVELS up to l_max."""
    l_max = min(l_max, U_dw.shape[1])
    if l_max <= 0 or dw_norm_sq <= 0:
        out: dict[str, float] = {}
        for l in RANK_L_LEVELS:
            out[f"A{l}_K"] = 0.0
            out[f"R2_aff_K_L{l}"] = 0.0
        return out

    Ul = U_dw[:, :l_max]
    weights = S_dw[:l_max].square()
    proj = Qk.T @ Ul
    a_i = proj.square().sum(dim=0)

    out = {}
    for l in RANK_L_LEVELS:
        ll = min(l, l_max)
        w = weights[:ll]
        ws = float(w.sum().item())
        if ws <= 0:
            out[f"A{l}_K"] = 0.0
            out[f"R2_aff_K_L{l}"] = 0.0
            continue
        a_lk = float((w * a_i[:ll]).sum().item() / ws)
        r_l = ws / dw_norm_sq
        out[f"A{l}_K"] = a_lk
        out[f"R2_aff_K_L{l}"] = r_l * a_lk
    return out


def analyze_pair(
    model_a: str,
    model_b: str,
    *,
    matrix_kind: str,
    side: str,
    device: torch.device,
) -> dict[str, Any]:
    X = load_matrix(model_a, matrix_kind).to(device)
    Y = load_matrix(model_b, matrix_kind).to(device)
    if X.shape != Y.shape:
        raise ValueError(f"shape mismatch {X.shape} vs {Y.shape}")

    n, d = X.shape
    delta = Y - X
    dw_norm_sq = float(delta.square().sum().item())
    ones = torch.ones(n, 1, device=device, dtype=X.dtype)
    Z = torch.cat([X, ones], dim=1)  # n x (d+1)

    # SVD of design matrix Z (sample-space left vectors Q)
    U_z, S_z, _ = torch.linalg.svd(Z, full_matrices=False)

    # SVD of ΔW (sample-space left vectors u_i)
    U_dw, S_dw, _ = torch.linalg.svd(delta, full_matrices=False)
    u1 = U_dw[:, 0]
    r1 = rank_energy_fraction(S_dw, dw_norm_sq, 1)
    r5 = rank_energy_fraction(S_dw, dw_norm_sq, 5)
    r10 = rank_energy_fraction(S_dw, dw_norm_sq, 10)

    ones_unit = ones.squeeze() / (n**0.5)
    a_bias = float((u1 @ ones_unit).square().item())

    def bias_top_l(l: int) -> float:
        ll = min(l, U_dw.shape[1])
        w = S_dw[:ll].square()
        ws = float(w.sum().item())
        if ws <= 0:
            return 0.0
        bias_i = (U_dw[:, :ll].T @ ones_unit).square()
        return float((w * bias_i).sum().item() / ws)

    b5 = bias_top_l(5)
    b10 = bias_top_l(10)

    # Full affine explainability R²_aff = ||P_Z ΔW||² / ||ΔW||²
    ZtZ = Z.T @ Z
    ZtDelta = Z.T @ delta
    theta = torch.linalg.solve(ZtZ, ZtDelta)
    pred = Z @ theta
    r2_aff = float(pred.square().sum().item() / dw_norm_sq) if dw_norm_sq > 0 else 0.0

    row: dict[str, Any] = {
        "model_a": model_a,
        "model_b": model_b,
        "side": side,
        "matrix_kind": matrix_kind,
        "n": n,
        "d": d,
        "delta_fro_norm_sq": dw_norm_sq,
        "r1": r1,
        "r5": r5,
        "r10": r10,
        "a_bias": a_bias,
        "B5": b5,
        "B10": b10,
        "R2_aff_full": r2_aff,
        "rank_Z": int(S_z.numel()),
    }

    for eta in ETA_THRESHOLDS:
        tag = str(int(eta * 100))
        k = k_for_eta(S_z, eta)
        Qk = U_z[:, :k]
        proj = Qk @ (Qk.T @ delta)
        r2_k = float(proj.square().sum().item() / dw_norm_sq) if dw_norm_sq > 0 else 0.0
        a1_k = float((Qk.T @ u1).square().sum().item())
        baseline = k / n
        row[f"k_eta{tag}"] = k
        row[f"R2_aff_k_eta{tag}"] = r2_k
        row[f"a1_k_eta{tag}"] = a1_k
        row[f"random_baseline_k_over_n_eta{tag}"] = baseline
        row[f"a1_over_baseline_eta{tag}"] = a1_k / baseline if baseline > 0 else float("inf")

        rank_l = rank_l_alignment_metrics(Qk, U_dw, S_dw, dw_norm_sq, l_max=10)
        for l in RANK_L_LEVELS:
            row[f"A{l}_K_eta{tag}"] = rank_l[f"A{l}_K"]
            row[f"A{l}_K_over_baseline_eta{tag}"] = (
                rank_l[f"A{l}_K"] / baseline if baseline > 0 else float("inf")
            )
            row[f"R2_aff_K_L{l}_eta{tag}"] = rank_l[f"R2_aff_K_L{l}"]

    del X, Y, delta, Z, U_z, S_z, U_dw, S_dw, pred, theta, Qk, proj
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument(
        "--all-clean",
        action="store_true",
        help="run all 30 BI-clean pairs (35 registered pairs minus 5 excluded anomalies)",
    )
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with TASK6_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    selected = [r for r in rows if not is_anomaly(r["model_a"])]
    if args.models:
        wanted = set(args.models)
        selected = [r for r in selected if r["model_a"] in wanted]
    elif not args.all_clean:
        parser.error("Specify --all-clean or --models")
    if args.limit:
        selected = selected[: args.limit]

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"CUDA unavailable, requested {args.device}")

    out_rows: list[dict[str, Any]] = []
    for i, row in enumerate(selected, 1):
        tied = row["actual_tied_a"].lower() == "true"
        for matrix_kind, side in [("embed", "E"), ("lm_head", "U")]:
            if tied and side == "U":
                continue
            print(
                f"[{i}/{len(selected)}] {side} {row['model_a']} -> {row['model_b']}",
                flush=True,
            )
            out_rows.append(
                analyze_pair(
                    row["model_a"],
                    row["model_b"],
                    matrix_kind=matrix_kind,
                    side=side,
                    device=device,
                )
            )
            if tied:
                break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"WROTE {args.out} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
