#!/usr/bin/env python3
"""一键复核 gcorr + affine 两模块关键数字（只读 ijcai_clean/results/）。"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

TOL = 1e-3


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def is_anomaly(model_a: str) -> bool:
    return model_a == "Gemma-3-1B" or model_a.startswith("Gemma-4-")


def close(a: float, b: float, tol: float = TOL) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


def check(label: str, got: float, expected: float, tol: float = TOL) -> bool:
    ok = close(got, expected, tol)
    mark = "OK" if ok else "FAIL"
    print(f"  [{mark}] {label}: got={got:.6f} expected={expected:.6f}")
    return ok


def main() -> int:
    root = repo_root()
    results = root / "ijcai_clean" / "results"
    golden_path = root / "__tep" / "data" / "computed_stats.json"
    if not golden_path.exists():
        print(f"缺少基准文件: {golden_path}", file=sys.stderr)
        return 1

    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    ok_all = True

    # --- gcorr: Task1 ---
    t1 = load_csv(results / "task1_base_instruct" / "summary.csv")
    cos = [float(r["gcorr_E_cos_mean"]) for r in t1]
    main_cos = [float(r["gcorr_E_cos_mean"]) for r in t1 if not is_anomaly(r["model_a"])]
    print("=== gcorr / Task1 ===")
    ok_all &= check("n=31", len(t1), golden["task1"]["n"])
    ok_all &= check("gcorr_E_cos median", median(cos), golden["task1"]["gcorr_E_cos"]["median"])
    ok_all &= check("main group cos mean (n=26)", mean(main_cos), 0.995, tol=0.002)

    # --- gcorr: Task2 ---
    t2 = load_csv(results / "task2_model_series" / "summary.csv")
    same_h = [float(r["gcorr_E_cos_mean"]) for r in t2 if r["hidden_dim_a"] == r["hidden_dim_b"]]
    diff_h = [float(r["gcorr_E_cos_mean"]) for r in t2 if r["hidden_dim_a"] != r["hidden_dim_b"]]
    print("=== gcorr / Task2 ===")
    ok_all &= check("n=110", len(t2), golden["task2"]["n"])
    ok_all &= check("same hidden cos mean", mean(same_h), 0.873067, tol=0.002)
    ok_all &= check("diff hidden cos mean", mean(diff_h), 0.375936, tol=0.002)

    # --- gcorr: Task3 ---
    t3 = load_csv(results / "task3_cross_scale_groups" / "summary.csv")
    neg_euc = sum(1 for r in t3 if float(r["gcorr_E_euc_mean"]) < 0)
    print("=== gcorr / Task3 ===")
    ok_all &= check("n=176", len(t3), golden["task3"]["n"])
    ok_all &= check("negative euc count", neg_euc, golden["task3"]["negative_euc_count"])
    ok_all &= check("cos mean", mean([float(r["gcorr_E_cos_mean"]) for r in t3]), golden["task3"]["gcorr_E_cos"]["mean"])

    # --- gcorr: Task4 ---
    t4 = load_csv(results / "task4_moe_cross_family" / "summary.csv")
    print("=== gcorr / Task4 ===")
    ok_all &= check("n=21", len(t4), golden["task4"]["n"])
    qwen_iter = max(
        float(r["gcorr_E_cos_mean"])
        for r in t4
        if "Qwen3.5" in r["model_a"] and "Qwen3.6" in r["model_b"]
        or "Qwen3.5" in r["model_b"] and "Qwen3.6" in r["model_a"]
    )
    ok_all &= check("Qwen3.5↔3.6 cos max", qwen_iter, 0.953, tol=0.002)

    # --- affine: Task5 ---
    t5 = load_csv(results / "task5_affine_subsampled" / "summary_pair.csv")
    t5_t1 = [float(r["R2_E"]) for r in t5 if "task1" in r["source_tasks"]]
    print("=== affine / Task5 ===")
    ok_all &= check("n_pair=338", len(t5), golden["task5"]["n_pair"])
    ok_all &= check("all R2_E mean", mean([float(r["R2_E"]) for r in t5]), golden["task5"]["R2_E"]["mean"])
    ok_all &= check("task1 subset R2_E mean", mean(t5_t1), golden["task5"]["task1_subset_R2_E"]["mean"])

    # --- affine: Task6 ---
    t6 = load_csv(results / "task6_base_instruct_full_vocab" / "summary_pair_base_instruct_full_vocab.csv")
    r2_all = [float(r["E_R2"]) for r in t6]
    r2_main = [float(r["E_R2"]) for r in t6 if not is_anomaly(r["model_a"])]
    rank95_h = [
        float(r["A_delta_A_delta_rank_95"]) / float(r["hidden_dim_a"])
        for r in t6
        if not is_anomaly(r["model_a"])
    ]
    ed_rank95_h = [
        float(r["E_delta_E_delta_rank_95"]) / float(r["hidden_dim_a"])
        for r in t6
        if not is_anomaly(r["model_a"])
    ]
    print("=== affine / Task6 ===")
    ok_all &= check("n=31", len(t6), golden["task6"]["n"])
    ok_all &= check("main E_R2 mean", mean(r2_main), golden["task6"]["E_R2_main"]["mean"])
    ok_all &= check("all E_R2 median", median(r2_all), golden["task6"]["E_R2_all"]["median"])
    ok_all &= check("main AI rank95/h mean", mean(rank95_h), golden["task6"]["main_AI_rank95_h"]["mean"], tol=0.005)
    ok_all &= check("main E_delta rank95/h mean", mean(ed_rank95_h), golden["task6"]["main_E_delta_rank95_h"]["mean"], tol=0.005)

    # --- cross: GCorr vs R², Task5 vs Task6 ---
    t1_by_pair = {(r["model_a"], r["model_b"]): float(r["gcorr_E_cos_mean"]) for r in t1}
    cos31, r2_31 = [], []
    for r in t6:
        k = (r["model_a"], r["model_b"])
        if k in t1_by_pair:
            cos31.append(t1_by_pair[k])
            r2_31.append(float(r["E_R2"]))
    t5_by_pair = {(r["model_a"], r["model_b"]): float(r["R2_E"]) for r in t5 if "task1" in r["source_tasks"]}
    r2_t5, r2_t6 = [], []
    for r in t6:
        k = (r["model_a"], r["model_b"])
        if k in t5_by_pair:
            r2_t5.append(t5_by_pair[k])
            r2_t6.append(float(r["E_R2"]))

    print("=== cross-module ===")
    ok_all &= check("pearson(gcorr_cos, E_R2)", pearson(cos31, r2_31), golden["merged_corr_cos_R2"], tol=0.002)
    ok_all &= check("pearson(Task5 R2, Task6 R2)", pearson(r2_t5, r2_t6), golden["task5_vs_task6_R2_corr"], tol=0.002)

    print()
    if ok_all:
        print("全部关键指标通过。")
        return 0
    print("存在未通过项，请对照 __tep/data/computed_stats.json。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
