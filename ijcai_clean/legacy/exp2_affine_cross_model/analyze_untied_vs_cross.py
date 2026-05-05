#!/usr/bin/env python3
"""
分析 tied / untied：
1. 模型内部 E→U：tied 与 untied 分别的 R2_EU
2. 跨模型：tied–tied、untied–untied 的 R2_E / R2_U
3. 对比 tied vs untied。
"""
import os
import pandas as pd

RESULTS_DIR = "/root/shared-nvme/ijcai/exp2_affine_cross_model/results"
THRESHOLD_TIED = 0.99  # R2_EU >= 0.99 视为 tied，否则 untied


def main():
    intra = pd.read_csv(os.path.join(RESULTS_DIR, "affine_intra_model_EU_results.csv"))
    cross = pd.read_csv(os.path.join(RESULTS_DIR, "affine_cross_model_results.csv"))

    # 1) 定义 tied / untied
    untied_models = set(intra[intra["R2_EU"] < THRESHOLD_TIED]["model"].tolist())
    tied_models = set(intra[intra["R2_EU"] >= THRESHOLD_TIED]["model"].tolist())
    print(f"Tied 模型数: {len(tied_models)},  Untied 模型数: {len(untied_models)}")
    print("Tied 模型:", sorted(tied_models))
    print("Untied 模型:", sorted(untied_models))

    # 2) 模型内部 E→U：tied 与 untied
    intra_tied = intra[intra["model"].isin(tied_models)].copy()
    intra_untied = intra[intra["model"].isin(untied_models)].copy()
    mean_R2_EU_tied = intra_tied["R2_EU"].mean()
    mean_R2_EU_untied = intra_untied["R2_EU"].mean()
    print(f"\n【模型内部 E→U】")
    print(f"  Tied:   模型数={len(intra_tied)},   R2_EU 均值={mean_R2_EU_tied:.6f}  范围=[{intra_tied['R2_EU'].min():.6f}, {intra_tied['R2_EU'].max():.6f}]")
    print(f"  Untied: 模型数={len(intra_untied)}, R2_EU 均值={mean_R2_EU_untied:.4f}  范围=[{intra_untied['R2_EU'].min():.4f}, {intra_untied['R2_EU'].max():.4f}]")

    # 3) 跨模型：tied–tied 与 untied–untied
    cross_tied = cross[
        cross["model_a"].isin(tied_models) & cross["model_b"].isin(tied_models)
    ].copy()
    cross_untied = cross[
        cross["model_a"].isin(untied_models) & cross["model_b"].isin(untied_models)
    ].copy()
    print(f"\n【跨模型 E–E / U–U】")
    if len(cross_tied) > 0:
        print(f"  Tied–tied:   对数={len(cross_tied)},  R2_E 均值={cross_tied['R2_E'].mean():.4f},  R2_U 均值={cross_tied['R2_U'].mean():.4f}")
    else:
        print("  Tied–tied: 0 对")
    if len(cross_untied) > 0:
        mean_R2_E = cross_untied["R2_E"].mean()
        mean_R2_U = cross_untied["R2_U"].mean()
        print(f"  Untied–untied: 对数={len(cross_untied)},  R2_E 均值={mean_R2_E:.4f},  R2_U 均值={mean_R2_U:.4f}")
    else:
        mean_R2_E = mean_R2_U = float("nan")

    # 4) 跨系列同规模
    cross_family_path = os.path.join(RESULTS_DIR, "affine_cross_family_same_scale_results.csv")
    if os.path.isfile(cross_family_path):
        cf = pd.read_csv(cross_family_path)
        cf_untied = cf[cf["model_a"].isin(untied_models) & cf["model_b"].isin(untied_models)]
        cf_tied = cf[cf["model_a"].isin(tied_models) & cf["model_b"].isin(tied_models)]
        print(f"\n【跨系列同规模】tied–tied: {len(cf_tied)} 对, untied–untied: {len(cf_untied)} 对")
        if len(cf_untied) > 0:
            print(f"  untied–untied R2_E={cf_untied['R2_E'].mean():.4f}, R2_U={cf_untied['R2_U'].mean():.4f}")
        if len(cf_tied) > 0:
            print(f"  tied–tied     R2_E={cf_tied['R2_E'].mean():.4f}, R2_U={cf_tied['R2_U'].mean():.4f}")

    # 5) 对比结论
    print("\n" + "=" * 60)
    print("Tied vs Untied 对比")
    print("=" * 60)
    print(f"  模型内 E→U:   Tied R²≈{mean_R2_EU_tied:.4f} (近 1)  >>  Untied R²={mean_R2_EU_untied:.4f}")
    if len(cross_tied) > 0 and len(cross_untied) > 0:
        print(f"  跨模型 R2_E:  Tied–tied={cross_tied['R2_E'].mean():.4f},  Untied–untied={mean_R2_E:.4f}")
        print(f"  跨模型 R2_U:  Tied–tied={cross_tied['R2_U'].mean():.4f},  Untied–untied={mean_R2_U:.4f}")

    # 6) 写出汇总表（含 tied / untied 全部）
    out = os.path.join(RESULTS_DIR, "untied_comparison_summary.csv")
    rows = [
        {"type": "intra_EU", "tied_untied": "untied", "n": len(intra_untied), "mean_R2": mean_R2_EU_untied, "note": "untied 模型内 E→U"},
        {"type": "intra_EU", "tied_untied": "tied", "n": len(intra_tied), "mean_R2": mean_R2_EU_tied, "note": "tied 模型内 E→U"},
    ]
    if len(cross_untied) > 0:
        rows.append({"type": "cross_E", "tied_untied": "untied", "n": len(cross_untied), "mean_R2": mean_R2_E, "note": "untied-untied 跨模型 E–E"})
        rows.append({"type": "cross_U", "tied_untied": "untied", "n": len(cross_untied), "mean_R2": mean_R2_U, "note": "untied-untied 跨模型 U–U"})
    if len(cross_tied) > 0:
        rows.append({"type": "cross_E", "tied_untied": "tied", "n": len(cross_tied), "mean_R2": cross_tied["R2_E"].mean(), "note": "tied-tied 跨模型 E–E"})
        rows.append({"type": "cross_U", "tied_untied": "tied", "n": len(cross_tied), "mean_R2": cross_tied["R2_U"].mean(), "note": "tied-tied 跨模型 U–U"})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
