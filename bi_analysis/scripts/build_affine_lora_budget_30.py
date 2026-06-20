#!/usr/bin/env python3
"""Synchronize BI-clean 30 model and aggregate tables from __tep final data."""

from __future__ import annotations

import csv
import statistics as stats
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "__tep" / "affine" / "tables" / "final" / "model_level_e_u_affine_lora_summary.csv"
OUT = ROOT / "bi_analysis" / "tables"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]], group_fn: Callable[[dict[str, str]], str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group in sorted({group_fn(row) for row in rows}):
        selected = [row for row in rows if group_fn(row) == group]
        summary: dict[str, Any] = {"group": group, "n": len(selected)}
        for column in rows[0]:
            if not column.startswith(("E_", "U_")):
                continue
            values = []
            for row in selected:
                try:
                    values.append(float(row[column]))
                except (KeyError, TypeError, ValueError):
                    pass
            if values:
                summary[f"{column}_mean"] = stats.mean(values)
                summary[f"{column}_median"] = stats.median(values)
        for side in ("E", "U"):
            for rank in (1, 8):
                column = f"{side}_aff_vs_W_ratio_r{rank}"
                summary[f"{side}_aff_wins_r{rank}"] = sum(float(row[column]) > 1 for row in selected)
        output.append(summary)
    return output


def main() -> None:
    rows = read_rows(SOURCE)
    if len(rows) != 30 or len({row["model_a"] for row in rows}) != 30:
        raise SystemExit("source must contain exactly 30 unique BI-clean pairs")
    synced = [{**row, "analysis_scope": "clean"} for row in rows]
    write_rows(OUT / "affine_lora_budget_summary.csv", synced)
    write_rows(
        OUT / "affine_lora_by_tied_summary.csv",
        summarize(rows, lambda row: f"tied={row['tie_word_embeddings']}"),
    )
    write_rows(
        OUT / "affine_lora_by_family_size_summary.csv",
        summarize(rows, lambda row: row["family"]),
    )
    print("WROTE BI-clean 30 model, tied, and family tables")


if __name__ == "__main__":
    main()
