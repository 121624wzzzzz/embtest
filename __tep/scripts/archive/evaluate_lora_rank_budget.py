#!/usr/bin/env python3
"""Evaluate LoRA-style low-rank affine budget from existing Task6 diagnostics.

This script does not refit models. It combines:
- Task6 full-vocab affine summary for R2, vocab, hidden, and A-I spectra.
- decomposition_svd for centered-delta / residual energies.
- proof_diagnostics for C_p(k), where p = X_c(A-I), at k = ceil(0.05 h).

The key derived quantity is the actual update-error reduction from an optimal
rank-k hidden-space affine adapter:

    gain_rank_k = (||D||^2 - ||D - X_c B_k||^2) / ||D||^2
                = w * C_p(k),

where D = Y_c - X_c, w = ||X_c(A-I)||^2 / ||D||^2, and C_p(k) is the top-k
energy share of the affine prediction term. This is the right quantity for a
LoRA-style adapter B_k = U V^T with U,V in R^{h x k}.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
TASK6_CSV = (
    REPO_ROOT
    / "ijcai_clean"
    / "results"
    / "task6_base_instruct_full_vocab"
    / "summary_pair_base_instruct_full_vocab.csv"
)
DECOMP_CSV = ROOT / "affine" / "tables" / "affine_task6_decomposition_svd.csv"
PROOF_CSV = ROOT / "affine" / "tables" / "affine_task6_proof_diagnostics.csv"
OUT_CSV = ROOT / "affine" / "tables" / "affine_task6_lora_rank_budget.csv"


def is_anomaly(model_a: str) -> bool:
    return model_a == "Gemma-3-1B" or model_a.startswith("Gemma-4-")


def read_by_model(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["model_a"]: row for row in csv.DictReader(f)}


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-anomalies", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    task6 = read_by_model(TASK6_CSV)
    decomp = read_by_model(DECOMP_CSV)
    proof = read_by_model(PROOF_CSV)

    rows: list[dict[str, Any]] = []
    for model_a, t in task6.items():
        if not args.include_anomalies and is_anomaly(model_a):
            continue
        if model_a not in decomp or model_a not in proof:
            continue

        drow = decomp[model_a]
        prow = proof[model_a]

        h = int(float(t["hidden_dim_a"]))
        n_common = int(float(t["n_common"]))
        k = int(float(prow["k_5pct"]))
        e_r2 = f(t, "E_R2")

        centered_energy = f(drow, "Centered_delta_total_energy")
        residual_energy = f(drow, "Residual_total_energy")
        pred_energy = f(drow, "Pred_delta_total_energy")
        mean_shift_energy = f(drow, "Mean_shift_total_energy")

        if e_r2 >= 1.0:
            y_center_energy = float("nan")
        else:
            y_center_energy = residual_energy / max(1.0 - e_r2, 1e-30)

        centered_over_y = centered_energy / y_center_energy
        mean_over_raw = mean_shift_energy / (centered_energy + mean_shift_energy)

        w_pred = pred_energy / centered_energy
        c_pred_k = f(prow, "C_G_p_5pct")
        c_ai_k = f(prow, "C_G_AminusI_5pct")
        c_cd_k = f(prow, "C_G_cd_5pct")

        rankk_update_gain = w_pred * c_pred_k
        identity_r2 = 1.0 - centered_over_y
        rankk_affine_r2 = identity_r2 + centered_over_y * rankk_update_gain
        full_affine_r2_from_decomp = identity_r2 + centered_over_y * w_pred
        oracle_delta_rankk_r2 = identity_r2 + centered_over_y * c_cd_k

        lora_params = 2 * h * k
        lora_params_with_bias = lora_params + h
        full_hh_params = h * h
        same_rank_token_delta_params = k * (n_common + h)

        rows.append(
            {
                "model_a": model_a,
                "model_b": t["model_b"],
                "is_anomaly": is_anomaly(model_a),
                "hidden_dim": h,
                "n_common": n_common,
                "rank_k": k,
                "rank_over_h": k / h,
                "lora_params_2hk": lora_params,
                "lora_params_2hk_plus_bias": lora_params_with_bias,
                "lora_params_over_full_hh": lora_params / full_hh_params,
                "same_rank_token_delta_params": same_rank_token_delta_params,
                "same_rank_token_delta_param_factor_vs_lora": (
                    same_rank_token_delta_params / lora_params
                ),
                "identity_R2": identity_r2,
                "rankk_affine_R2": rankk_affine_r2,
                "full_affine_R2_from_decomp": full_affine_r2_from_decomp,
                "full_affine_R2_reported": e_r2,
                "oracle_delta_rankk_R2_not_param_matched": oracle_delta_rankk_r2,
                "centered_delta_over_Yc_energy": centered_over_y,
                "mean_shift_over_raw_delta_energy": mean_over_raw,
                "full_affine_update_gain_w": w_pred,
                "rankk_affine_update_gain_wCp": rankk_update_gain,
                "rankk_fraction_of_full_affine_gain_Cp": c_pred_k,
                "A_I_energy_at_rank_k": c_ai_k,
                "pred_energy_at_rank_k": c_pred_k,
                "centered_delta_energy_at_rank_k": c_cd_k,
                "rankk_affine_gain_minus_oracle_delta_gain": rankk_update_gain - c_cd_k,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def summarize(key: str) -> tuple[float, float, float, float]:
        vals = sorted(float(r[key]) for r in rows)
        n = len(vals)
        median = (vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2)
        return sum(vals) / n, median, vals[0], vals[-1]

    print(f"WROTE {args.out}")
    print(f"n = {len(rows)}")
    for key in (
        "identity_R2",
        "rankk_affine_R2",
        "full_affine_R2_reported",
        "rankk_affine_update_gain_wCp",
        "rankk_fraction_of_full_affine_gain_Cp",
        "A_I_energy_at_rank_k",
        "centered_delta_energy_at_rank_k",
        "same_rank_token_delta_param_factor_vs_lora",
    ):
        mean, median, min_v, max_v = summarize(key)
        print(
            f"{key}: mean={mean:.6f} median={median:.6f} "
            f"min={min_v:.6f} max={max_v:.6f}"
        )


if __name__ == "__main__":
    main()
