# Posthoc Analysis Tables

`_analysis/` 保存从 `ijcai_clean/results/task*/` 正式结果派生出来的轻量排查表。这里不是主线 Task 输出目录，也不放专项 debug 实验的默认输出。

Gemma / E/U checkpoint 几何审计已迁到仓库根目录 `analysis_eu_geometry/`。

## 当前文件

| 文件 | 来源 | 用途 |
|------|------|------|
| `main_group_base_instruct_summary.csv` | Task1 summary + Task6 full-vocab summary | 主分析组 Base-Instruct pair 的筛选汇总。 |
| `anomaly_group_base_instruct.csv` | Task1 summary + Task6 full-vocab summary | 异常组 Base-Instruct pair 的筛选汇总，用于解释 Gemma 等特殊现象。 |
| `task2_hidden_dim_mismatch_pairs.csv` | Task2 pair plan / summary | Task2 中 hidden dim 不一致或因此无法直接比较的 pair 清单。 |
| `task3_negative_gcorr_pairs.csv` | Task3 summary | Task3 中 GCorr 为负或异常偏低的 pair 清单。 |
| `task5_low_intra_EU_untied.csv` | Task5 `summary_intra_EU.csv` | untied 模型内部 E->U 仿射解释力偏低的模型清单。 |

## 使用约定

新增文件时请优先满足：

1. 可以从 `task*/summary*.csv`、`pair_plan.csv` 或 `metadata.json` 重新派生。
2. 文件名能说明来源任务和筛选条件。
3. 如果是专项 debug 实验，放到独立工作区，而不是这里。
