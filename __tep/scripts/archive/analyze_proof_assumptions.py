#!/usr/bin/env python3
"""Rigorous verification of the two remaining empirical assumptions in
affine_decomposition_proof.md:

  (A) $\\kappa(U_r^\\top S U_r)$ is small => Ostrowski gives $G_p \\approx G_{A-I}$.
  (B) top-k eigenspaces of $G_p$ and $G_R$ are highly aligned, which makes the
      Ky Fan upper bound tight.

Runs on all 31 Task6 BI pairs (main + anomaly) so we can also test the
theorem's failure boundary.
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
    / "affine_task6_proof_diagnostics.csv"
)

ANOMALY_MODELS = {"Gemma-3-1B"}
BATCH_ROWS = 8192


def is_anomaly(model_a: str) -> bool:
    return model_a in ANOMALY_MODELS or model_a.startswith("Gemma-4-")


def series_of(model_a: str) -> str:
    for prefix in (
        "Qwen3.5",
        "Qwen3-",
        "Qwen2.5",
        "Llama",
        "Gemma-2",
        "Gemma-3",
        "Gemma-4",
    ):
        if model_a.startswith(prefix):
            return prefix.rstrip("-")
    return model_a.split("-", 1)[0]


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
    eig = torch.linalg.eigvalsh(M)
    return torch.clamp(eig, min=0.0).flip(0)


def descending_eigh(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    vals, vecs = torch.linalg.eigh(M)
    vals = torch.clamp(vals, min=0.0)
    return vals.flip(0), vecs.flip(1)


def energy_at_k(eigvals: torch.Tensor, k: int) -> float:
    total = float(eigvals.sum())
    if total <= 0:
        return 0.0
    k = max(1, min(k, int(eigvals.numel())))
    return float(eigvals[:k].sum()) / total


def rank_for_threshold(eigvals: torch.Tensor, threshold: float) -> int:
    total = float(eigvals.sum())
    if total <= 0:
        return 0
    cum = torch.cumsum(eigvals, dim=0) / total
    idx = torch.searchsorted(
        cum, torch.tensor(threshold, device=cum.device, dtype=cum.dtype)
    )
    return int(idx.item()) + 1


def principal_angle_stats(U_a: torch.Tensor, U_b: torch.Tensor) -> dict[str, float]:
    """For orthonormal-column matrices U_a, U_b in R^{d x k}, returns:
    - mean_cos2 = (1/k) * tr(Pi_a Pi_b) = average squared principal cosine
    - max_cos2 / min_cos2 = extreme squared principal cosines
    """
    k = U_a.shape[1]
    svals = torch.linalg.svdvals(U_a.T @ U_b)
    cos2 = svals.clamp(min=0.0) ** 2
    return {
        "mean_cos2": float(cos2.mean().item()),
        "max_cos2": float(cos2.max().item()),
        "min_cos2": float(cos2.min().item()),
    }


def compute_for_pair(
    model_a: str,
    model_b: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    X_np = load_embed(model_a)
    Y_np = load_embed(model_b)
    n, d = X_np.shape
    if Y_np.shape != X_np.shape:
        raise ValueError(f"shape mismatch: {X_np.shape} vs {Y_np.shape}")

    sum_x = np.zeros(d, dtype=np.float64)
    sum_y = np.zeros(d, dtype=np.float64)
    for start in range(0, n, BATCH_ROWS):
        end = min(n, start + BATCH_ROWS)
        sum_x += X_np[start:end].astype(np.float64, copy=False).sum(axis=0)
        sum_y += Y_np[start:end].astype(np.float64, copy=False).sum(axis=0)
    mx = torch.from_numpy((sum_x / n).astype(np.float32)).to(device=device, dtype=dtype)
    my = torch.from_numpy((sum_y / n).astype(np.float32)).to(device=device, dtype=dtype)

    S = torch.zeros((d, d), device=device, dtype=dtype)
    C = torch.zeros((d, d), device=device, dtype=dtype)
    Gcd = torch.zeros((d, d), device=device, dtype=dtype)
    for start in range(0, n, BATCH_ROWS):
        end = min(n, start + BATCH_ROWS)
        x = as_tensor(X_np[start:end], device, dtype) - mx
        y = as_tensor(Y_np[start:end], device, dtype) - my
        S.add_(x.T @ x)
        C.add_(x.T @ y)
        cd = y - x
        Gcd.add_(cd.T @ cd)
        del x, y, cd

    A = torch.linalg.solve(S, C)
    Delta = A - torch.eye(d, device=device, dtype=dtype)
    Gp = Delta.T @ S @ Delta
    Gp = (Gp + Gp.T) * 0.5
    GR = Gcd - Gp
    GR = (GR + GR.T) * 0.5
    GAI = Delta.T @ Delta
    GAI = (GAI + GAI.T) * 0.5

    U_Delta, sigma_Delta, _ = torch.linalg.svd(Delta, full_matrices=False)

    S_eig = descending_eigvalsh(S)
    kappa_S = float(S_eig[0]) / max(float(S_eig[-1]), 1e-20)

    def kappa_K_top(r: int) -> float:
        r = max(1, min(r, d))
        U_r = U_Delta[:, :r]
        K_r = U_r.T @ S @ U_r
        K_r = (K_r + K_r.T) * 0.5
        eig = descending_eigvalsh(K_r)
        return float(eig[0]) / max(float(eig[-1]), 1e-20)

    r5 = max(1, math.ceil(0.05 * d))
    r10 = max(1, math.ceil(0.10 * d))
    r50 = max(1, math.ceil(0.50 * d))

    GAI_vals = descending_eigvalsh(GAI)
    r95_AI = rank_for_threshold(GAI_vals, 0.95)

    kappa_K5 = kappa_K_top(r5)
    kappa_K10 = kappa_K_top(r10)
    kappa_K50 = kappa_K_top(r50)
    kappa_K_rAI = kappa_K_top(r95_AI)

    Gp_vals, Gp_vecs = descending_eigh(Gp)
    GR_vals, GR_vecs = descending_eigh(GR)
    Gcd_vals = descending_eigvalsh(Gcd)

    overlap_5 = principal_angle_stats(Gp_vecs[:, :r5], GR_vecs[:, :r5])

    Cp = energy_at_k(Gp_vals, r5)
    CR = energy_at_k(GR_vals, r5)
    Ccd = energy_at_k(Gcd_vals, r5)
    CAI = energy_at_k(GAI_vals, r5)
    T_Gp = float(Gp_vals.sum())
    T_Gcd = float(Gcd_vals.sum())
    w = T_Gp / T_Gcd if T_Gcd > 0 else float("nan")
    UB = w * Cp + (1.0 - w) * CR
    LB = max(w * Cp, (1.0 - w) * CR)

    r95_p = rank_for_threshold(Gp_vals, 0.95)
    r95_R = rank_for_threshold(GR_vals, 0.95)
    r95_cd = rank_for_threshold(Gcd_vals, 0.95)

    out = {
        "model_a": model_a,
        "model_b": model_b,
        "series": series_of(model_a),
        "hidden_dim": d,
        "is_anomaly": is_anomaly(model_a),
        "k_5pct": r5,
        "rank95_A_minus_I": r95_AI,
        "kappa_S": kappa_S,
        "kappa_K_top_5pct_AminusI": kappa_K5,
        "kappa_K_top_10pct_AminusI": kappa_K10,
        "kappa_K_top_50pct_AminusI": kappa_K50,
        "kappa_K_rank95_AminusI": kappa_K_rAI,
        "rank95_G_p_over_h": r95_p / d,
        "rank95_G_R_over_h": r95_R / d,
        "rank95_G_cd_over_h": r95_cd / d,
        "rank95_G_AminusI_over_h": r95_AI / d,
        "overlap_mean_cos2_5pct": overlap_5["mean_cos2"],
        "overlap_max_cos2_5pct": overlap_5["max_cos2"],
        "overlap_min_cos2_5pct": overlap_5["min_cos2"],
        "C_G_p_5pct": Cp,
        "C_G_R_5pct": CR,
        "C_G_cd_5pct": Ccd,
        "C_G_AminusI_5pct": CAI,
        "C_Gp_minus_C_AminusI_5pct": Cp - CAI,
        "w_share_pred": w,
        "convex_UB_5pct": UB,
        "convex_LB_5pct": LB,
        "gap_UB_minus_obs": UB - Ccd,
        "gap_obs_minus_LB": Ccd - LB,
    }

    del X_np, Y_np
    del A, Delta, S, C, Gcd, Gp, GR, GAI
    del U_Delta, sigma_Delta, Gp_vecs, GR_vecs
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    pairs = list(csv.DictReader(TASK6_CSV.open(newline="", encoding="utf-8")))
    if args.limit:
        pairs = pairs[: args.limit]

    device = torch.device(args.device)
    dtype = torch.float32

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict[str, Any]] = []

    for idx, p in enumerate(pairs, 1):
        ma, mb = p["model_a"], p["model_b"]
        print(
            f"[{idx}/{len(pairs)}] {ma} -> {mb}  anomaly={is_anomaly(ma)}",
            flush=True,
        )
        row = compute_for_pair(ma, mb, device=device, dtype=dtype)
        out_rows.append(row)
        with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        print(
            "  kappa_S={ks:.1f} kappa_K_5pct={k5:.2f} kappa_K_rAI={kr:.2f} "
            "overlap_5pct={ov:.3f} |Cp-CAI|={dc:.3f} UB-obs={gu:+.4f}".format(
                ks=row["kappa_S"],
                k5=row["kappa_K_top_5pct_AminusI"],
                kr=row["kappa_K_rank95_AminusI"],
                ov=row["overlap_mean_cos2_5pct"],
                dc=abs(row["C_Gp_minus_C_AminusI_5pct"]),
                gu=row["gap_UB_minus_obs"],
            ),
            flush=True,
        )

    def summarize(rows: list[dict[str, Any]], title: str) -> None:
        if not rows:
            return
        print(f"\n=== {title} (n={len(rows)}) ===")
        keys = [
            "kappa_S",
            "kappa_K_top_5pct_AminusI",
            "kappa_K_top_10pct_AminusI",
            "kappa_K_top_50pct_AminusI",
            "kappa_K_rank95_AminusI",
            "overlap_mean_cos2_5pct",
            "overlap_max_cos2_5pct",
            "overlap_min_cos2_5pct",
            "C_G_p_5pct",
            "C_G_AminusI_5pct",
            "C_Gp_minus_C_AminusI_5pct",
            "w_share_pred",
            "gap_UB_minus_obs",
        ]
        for k in keys:
            xs = sorted(row[k] for row in rows)
            n = len(xs)
            mean = sum(xs) / n
            med = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
            print(
                f"  {k}: mean={mean:.4f} median={med:.4f} min={xs[0]:.4f} max={xs[-1]:.4f}"
            )

    summarize([r for r in out_rows if not r["is_anomaly"]], "MAIN")
    summarize([r for r in out_rows if r["is_anomaly"]], "ANOMALY")

    print(f"\nWROTE {OUT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
