#!/usr/bin/env python3
"""Verify current Task1-6 outputs against the BI-clean 30 reference statistics."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

TOL = 1e-6


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def excluded(model_a: str) -> bool:
    return model_a == "Gemma-3-1B" or model_a.startswith("Gemma-4-")


def check(label: str, got: float, expected: float, tol: float = TOL) -> bool:
    ok = math.isfinite(got) and abs(got - expected) <= tol
    print(f"  [{'OK' if ok else 'FAIL'}] {label}: got={got:.9f} expected={expected:.9f}")
    return ok


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    results = root / "cross_model_geometry" / "results"
    golden = json.loads((root / "__tep" / "data" / "computed_stats.json").read_text())
    ok = True

    t1 = load_csv(results / "task1_base_instruct" / "summary.csv")
    t1_clean = [r for r in t1 if not excluded(r["model_a"])]
    print("=== Task1 / BI-clean ===")
    ok &= check("BI-full rows", len(t1), golden["task1"]["n_full"])
    ok &= check("BI-clean rows", len(t1_clean), golden["task1"]["n_clean"])
    ok &= check(
        "clean E cosine mean",
        mean([float(r["gcorr_E_cos_mean"]) for r in t1_clean]),
        golden["task1"]["clean_E_cos_mean"],
    )

    t2 = load_csv(results / "task2_model_series" / "summary.csv")
    same = [float(r["gcorr_E_cos_mean"]) for r in t2 if r["hidden_dim_a"] == r["hidden_dim_b"]]
    diff = [float(r["gcorr_E_cos_mean"]) for r in t2 if r["hidden_dim_a"] != r["hidden_dim_b"]]
    print("=== Task2 ===")
    ok &= check("rows", len(t2), golden["task2"]["n"])
    ok &= check("same-hidden cosine mean", mean(same), golden["task2"]["same_hidden_cos_mean"])
    ok &= check("different-hidden cosine mean", mean(diff), golden["task2"]["diff_hidden_cos_mean"])

    t3 = load_csv(results / "task3_cross_scale_groups" / "summary.csv")
    print("=== Task3 ===")
    ok &= check("rows", len(t3), golden["task3"]["n"])
    ok &= check(
        "negative Euclidean GCorr count",
        sum(float(r["gcorr_E_euc_mean"]) < 0 for r in t3),
        golden["task3"]["negative_euc_count"],
    )

    t4 = load_csv(results / "task4_moe_cross_family" / "summary.csv")
    t5 = load_csv(results / "task5_affine_subsampled" / "summary_pair.csv")
    print("=== Task4 / Task5 ===")
    ok &= check("Task4 rows", len(t4), golden["task4"]["n"])
    ok &= check("Task5 unique rows", len(t5), golden["task5"]["n_unique"])
    ok &= check("Task5 R2_E mean", mean([float(r["R2_E"]) for r in t5]), golden["task5"]["R2_E_mean"])

    t6 = load_csv(results / "task6_base_instruct_full_vocab" / "summary_pair_base_instruct_full_vocab.csv")
    t6_clean = [r for r in t6 if not excluded(r["model_a"])]
    print("=== Task6 / BI-clean ===")
    ok &= check("BI-full rows", len(t6), golden["task6"]["n_full"])
    ok &= check("BI-clean rows", len(t6_clean), golden["task6"]["n_clean"])
    ok &= check(
        "clean E_R2 mean",
        mean([float(r["E_R2"]) for r in t6_clean]),
        golden["task6"]["clean_E_R2_mean"],
    )
    ok &= check(
        "clean A-I rank95/h mean",
        mean([float(r["A_delta_A_delta_rank_95"]) / float(r["hidden_dim_a"]) for r in t6_clean]),
        golden["task6"]["clean_A_delta_rank95_h_mean"],
    )

    print("\n全部 BI-clean 30 口径检查通过。" if ok else "\n存在不一致。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
