#!/usr/bin/env python3
"""Verify Ky-Fan convex-combination bound for the centered-delta decomposition.

Reads __tep/affine/tables/affine_task6_decomposition_svd.csv and checks, per
pair, the inequality

    max(w C_p(k), (1-w) C_R(k))
        <= C_cd(k)
        <= w C_p(k) + (1-w) C_R(k)

where w = tr(G_p)/tr(G_cd), C_M(k) = top-k energy fraction of M, and k = 5% h.
Also reports the scale ratio rho = ||Y_c|| / ||Y_c - X_c|| inferred from R^2.

The script is read-only; it prints a per-pair table and a summary.
"""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "__tep" / "affine" / "tables" / "affine_task6_decomposition_svd.csv"


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows = load_rows()
    main_rows = [r for r in rows if r["is_anomaly"].lower() == "false"]

    print(f"n_main = {len(main_rows)}\n")
    header = (
        f"{'model_a':30s} {'R2':>6s} {'rho':>5s} {'w':>5s} "
        f"{'C_p(k)':>7s} {'C_R(k)':>7s} {'C_cd(k)':>8s} "
        f"{'LB':>6s} {'UB':>6s} {'gap_UB':>8s}"
    )
    print(header)
    print("-" * len(header))

    rec = []
    for r in main_rows:
        Tp = float(r["Pred_delta_total_energy"])
        TR = float(r["Residual_total_energy"])
        Tcd = float(r["Centered_delta_total_energy"])
        w_data = Tp / Tcd
        Cp = float(r["Pred_delta_energy_5pct_h"])
        CR = float(r["Residual_energy_5pct_h"])
        Ccd = float(r["Centered_delta_energy_5pct_h"])
        UB = w_data * Cp + (1.0 - w_data) * CR
        LB = max(w_data * Cp, (1.0 - w_data) * CR)
        R2 = float(r["E_R2"])
        rho_sq_implied = (TR / Tcd) / max(1.0 - R2, 1e-12) if Tcd > 0 else float("nan")
        rho = math.sqrt(rho_sq_implied) if rho_sq_implied > 0 else float("nan")
        rec.append(
            {
                "model_a": r["model_a"],
                "R2": R2,
                "rho": rho,
                "w": w_data,
                "Cp": Cp,
                "CR": CR,
                "Ccd": Ccd,
                "LB": LB,
                "UB": UB,
                "gap_UB": UB - Ccd,
                "gap_LB": Ccd - LB,
            }
        )
        print(
            f"{r['model_a']:30s} {R2:6.3f} {rho:5.2f} {w_data:5.3f} "
            f"{Cp:7.3f} {CR:7.3f} {Ccd:8.3f} "
            f"{LB:6.3f} {UB:6.3f} {UB - Ccd:+8.4f}"
        )

    def summ(key: str) -> str:
        xs = [r[key] for r in rec]
        return (
            f"mean={statistics.mean(xs):+.4f} "
            f"median={statistics.median(xs):+.4f} "
            f"min={min(xs):+.4f} max={max(xs):+.4f}"
        )

    print()
    print("Per-pair gap statistics (n = {}):".format(len(rec)))
    print("  UB - C_cd:", summ("gap_UB"))
    print("  C_cd - LB:", summ("gap_LB"))

    violations = [r for r in rec if r["gap_UB"] < -1e-3 or r["gap_LB"] < -1e-3]
    if violations:
        print("\nBound violations (numerical / cross-term residual):")
        for r in violations:
            print(f"  {r['model_a']}: UB-C={r['gap_UB']:+.4f}, C-LB={r['gap_LB']:+.4f}")
    else:
        print("\nAll 26 pairs satisfy LB <= C_cd <= UB (within 1e-3).")

    print()
    print("rho summary:")
    print("  rho:", summ("rho"))
    print()
    print("If R^2 = 1 - eps, then 1 - w = eps * rho^2 (residual share).")


if __name__ == "__main__":
    main()
