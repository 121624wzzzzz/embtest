#!/usr/bin/env python3
"""Write __tep common-k spectrum CSVs with D/P/R/P_abs columns from compute output."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEP = REPO / "__tep" / "affine" / "tables"
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

BASE_COLS_E = [
    "model_a",
    "model_b",
    "hidden_dim",
    "vocab",
    "identity_R2",
    "full_affine_R2",
    "delta_over_Yc_energy",
    "full_affine_gain_over_delta",
]
BASE_COLS_U = ["matrix_kind"] + BASE_COLS_E


def spectrum_fieldnames(base: list[str]) -> list[str]:
    fields = list(base)
    for k in RANKS:
        fields.extend(
            [
                f"D_energy_at_{k}",
                f"P_energy_at_{k}",
                f"R_energy_at_{k}",
                f"P_abs_delta_energy_at_{k}",
            ]
        )
    return fields


def row_for_side(raw: dict[str, str], *, u: bool) -> dict[str, str]:
    out: dict[str, str] = {}
    if u:
        out["matrix_kind"] = raw.get("matrix_kind", "lm_head")
    for c in BASE_COLS_E:
        out[c] = raw[c]
    for k in RANKS:
        out[f"D_energy_at_{k}"] = raw[f"D_energy_at_{k}"]
        out[f"P_energy_at_{k}"] = raw[f"P_energy_at_{k}"]
        out[f"R_energy_at_{k}"] = raw[f"R_energy_at_{k}"]
        out[f"P_abs_delta_energy_at_{k}"] = raw[f"P_abs_delta_energy_at_{k}"]
    return out


def write_summary(spec_path: Path, summary_path: Path) -> None:
    with spec_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    summary_rows = []
    for k in RANKS:
        d = [float(r[f"D_energy_at_{k}"]) for r in rows]
        p = [float(r[f"P_energy_at_{k}"]) for r in rows]
        r = [float(r[f"R_energy_at_{k}"]) for r in rows]
        pa = [float(r[f"P_abs_delta_energy_at_{k}"]) for r in rows]
        summary_rows.append(
            {
                "k": k,
                "D_median": statistics.median(d),
                "P_median": statistics.median(p),
                "R_median": statistics.median(r),
                "P_abs_delta_median": statistics.median(pa),
                "D_mean": statistics.mean(d),
                "P_mean": statistics.mean(p),
                "R_mean": statistics.mean(r),
                "P_abs_delta_mean": statistics.mean(pa),
            }
        )
    fields = [
        "k",
        "D_median",
        "P_median",
        "R_median",
        "P_abs_delta_median",
        "D_mean",
        "P_mean",
        "R_mean",
        "P_abs_delta_mean",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--e-in", type=Path, default=Path("/tmp/bi_clean_e_dpr.csv"))
    parser.add_argument("--u-in", type=Path, default=Path("/tmp/bi_clean_u_dpr.csv"))
    args = parser.parse_args()

    e_raw = list(csv.DictReader(args.e_in.open(newline="", encoding="utf-8")))
    u_raw = list(csv.DictReader(args.u_in.open(newline="", encoding="utf-8")))

    e_out = TEP / "e" / "affine_pred_delta_common_spectrum.csv"
    u_out = TEP / "u" / "unembed_pred_delta_common_spectrum.csv"
    e_fields = spectrum_fieldnames(BASE_COLS_E)
    u_fields = spectrum_fieldnames(BASE_COLS_U)

    e_rows = [row_for_side(r, u=False) for r in e_raw]
    u_rows = [row_for_side(r, u=True) for r in u_raw]

    for path, fields, rows in [
        (e_out, e_fields, e_rows),
        (u_out, u_fields, u_rows),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"WROTE {path} ({len(rows)} rows, {len(fields)} cols)")

    write_summary(e_out, TEP / "e" / "affine_pred_delta_common_spectrum_summary.csv")
    write_summary(u_out, TEP / "u" / "unembed_pred_delta_common_spectrum_summary.csv")
    print("updated summaries with R_median / R_mean")


if __name__ == "__main__":
    main()
