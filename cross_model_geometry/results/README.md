# Results Layout

`cross_model_geometry/results/` 保存主线实验输出。默认约定：

- `task1_*` 到 `task6_*` 是可复现的正式 Task 输出。
- `_analysis/` 只放从正式结果派生出来的 posthoc 表或人工排查清单。
- 专项分析不要写入这里；例如全库 E/U checkpoint 几何审计放在仓库根目录的 `analysis_eu_geometry/`。

## 正式任务目录

| 目录 | 来源任务 | 主要内容 |
|------|----------|----------|
| `task1_base_instruct/` | Task1 | Base-Instruct GCorr summary、bootstrap 与 metadata。 |
| `task2_model_series/` | Task2 | 同系列 pair plan、generated pairs、GCorr summary。 |
| `task3_cross_scale_groups/` | Task3 | 跨系列 / 跨规模桶 pair plan、generated pairs、GCorr summary。 |
| `task4_moe_cross_family/` | Task4 | MoE / 跨 family pair plan、generated pairs、GCorr summary。 |
| `task5_affine_subsampled/` | Task5 | Task1-4 pair 并集的子采样仿射 R² 和 intra E->U 结果。 |
| `task6_base_instruct_full_vocab/` | Task6 | Base-Instruct 全词表仿射、A-I 诊断和 SVD 能量报告。 |

## 派生分析目录

`_analysis/` 中的文件应满足两个条件：

1. 可以从 `task*/` 正式输出重新派生。
2. 文件名或 README 能说明来源和用途。

当前这里保留的是非 Gemma 的 posthoc 分析表，例如异常组汇总、hidden dim mismatch 清单、负 GCorr pair 清单等。E/U checkpoint 几何审计见 `../../analysis_eu_geometry/`。

## 注意

`bootstrap_results.csv` 体积可能较大，通常由 `.gitignore` 忽略。需要复核完整 bootstrap 时以本地文件为准；需要轻量阅读时优先看各目录下的 `summary*.csv` 和 `metadata.json`。
