#!/usr/bin/env python3
"""Augment W-rank budget sweep CSVs with raw-scale explain fractions.

Under a fixed-bias adapter model, rank-r low-rank terms explain centered delta D,
while bias h explains the rank-1 row mean shift.  Raw-scale gains are:

    gain_raw = mean/raw + (D/raw) * gain_centered

where gain_centered is w_rank_update_gain or rank_affine_update_gain.
"""

from __future__ import annotations

import argparse
import csv
import statistics as stats
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "affine" / "tables" / "final" / "model_level_e_u_affine_lora_summary.csv"

DEFAULT_SWEEPS = {
    "embed": ROOT / "affine" / "tables" / "e" / "affine_w_rank_budget_clean.csv",
    "lm_head": ROOT / "affine" / "tables" / "u" / "unembed_w_rank_budget_clean.csv",
}

SUMMARY_RANKS = (1, 2, 4, 8, 16, 32, 64, 128)


def load_mean_shifts(summary_path: Path) -> tuple[dict[str, float], dict[str, float]]:
    e_map: dict[str, float] = {}
    u_map: dict[str, float] = {}
    with summary_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            e_map[row["model_a"]] = float(row["E_mean_shift_over_raw_delta"])
            u_map[row["model_a"]] = float(row["U_mean_shift_over_raw_delta"])
    return e_map, u_map


def mean_shift_for_row(
    row: dict[str, str], side: str, e_map: dict[str, float], u_map: dict[str, float]
) -> float:
    if "mean_shift_over_raw_delta" in row and row["mean_shift_over_raw_delta"]:
        return float(row["mean_shift_over_raw_delta"])
    model_a = row["model_a"]
    return u_map[model_a] if side == "U" else e_map[model_a]


def augment_row(
    row: dict[str, str], *, side: str, e_map: dict[str, float], u_map: dict[str, float]
) -> dict[str, Any]:
    out = dict(row)
    ms = mean_shift_for_row(row, side, e_map, u_map)
    d_over_raw = 1.0 - ms
    w_gain = float(row["w_rank_update_gain"])
    aff_gain = float(row["rank_affine_update_gain"])
    full_gain = float(row["full_affine_update_gain"])

    w_raw = ms + d_over_raw * w_gain
    aff_raw = ms + d_over_raw * aff_gain
    full_raw = ms + d_over_raw * full_gain

    out["mean_shift_over_raw_delta"] = ms
    out["centered_delta_over_raw_delta"] = d_over_raw
    out["w_rank_update_gain_over_raw"] = w_raw
    out["rank_affine_update_gain_over_raw"] = aff_raw
    out["full_affine_update_gain_over_raw"] = full_raw
    out["rank_affine_gain_over_w_gain_raw"] = aff_raw / w_raw if w_raw > 0 else ""
    out["affine_wins_centered"] = int(aff_gain > w_gain)
    out["affine_wins_raw"] = int(aff_raw > w_raw)
    return out


def summarize(rows: list[dict[str, Any]], ranks: tuple[int, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rank_w in ranks:
        sub = [r for r in rows if int(r["rank_w"]) == rank_w]
        if not sub:
            continue
        wg = [float(r["w_rank_update_gain"]) for r in sub]
        ag = [float(r["rank_affine_update_gain"]) for r in sub]
        wgr = [float(r["w_rank_update_gain_over_raw"]) for r in sub]
        agr = [float(r["rank_affine_update_gain_over_raw"]) for r in sub]
        ms = [float(r["mean_shift_over_raw_delta"]) for r in sub]
        wins_c = sum(int(r["affine_wins_centered"]) for r in sub)
        wins_r = sum(int(r["affine_wins_raw"]) for r in sub)
        out.append(
            {
                "rank_w": rank_w,
                "n": len(sub),
                "mean_shift_over_raw_median": stats.median(ms),
                "w_gain_centered_median": stats.median(wg),
                "aff_gain_centered_median": stats.median(ag),
                "w_gain_raw_median": stats.median(wgr),
                "aff_gain_raw_median": stats.median(agr),
                "aff_over_w_centered_median": stats.median(
                    [a / w if w > 0 else float("inf") for a, w in zip(ag, wg)]
                ),
                "aff_over_w_raw_median": stats.median(
                    [a / w if w > 0 else float("inf") for a, w in zip(agr, wgr)]
                ),
                "affine_wins_centered": wins_c,
                "affine_wins_raw": wins_r,
            }
        )
    return out


def process_sweep(
    sweep_path: Path,
    *,
    side: str,
    e_map: dict[str, float],
    u_map: dict[str, float],
    out_path: Path,
    summary_path: Path,
) -> None:
    with sweep_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    augmented = [augment_row(r, side=side, e_map=e_map, u_map=u_map) for r in rows]
    fieldnames = list(augmented[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(augmented)

    summary = summarize(augmented, SUMMARY_RANKS)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"WROTE {out_path} ({len(augmented)} rows)")
    print(f"WROTE {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY_CSV)
    args = parser.parse_args()
    e_map, u_map = load_mean_shifts(args.summary)

    for side, sweep in DEFAULT_SWEEPS.items():
        tag = "e" if side == "embed" else "u"
        out = sweep
        summary_out = (
            ROOT / "affine" / "tables" / tag / f"{tag}_w_rank_budget_summary.csv"
        )
        process_sweep(
            sweep,
            side="U" if side == "lm_head" else "E",
            e_map=e_map,
            u_map=u_map,
            out_path=out,
            summary_path=summary_out,
        )


if __name__ == "__main__":
    main()
