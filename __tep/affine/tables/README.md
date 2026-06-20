# Affine Tables：BI-clean 30

本目录全部现行表统一覆盖 BI-clean 30，不保留较早的小规模 BI 表。

## Final

| 文件 | 行数 | 内容 |
|---|---:|---|
| `final/model_level_e_u_affine_lora_summary.csv` | 30 | 每 pair 的 E/U decomposition、谱和 W-rank 摘要 |
| `final/model_level_e_u_by_tied_summary.csv` | 2 | tied / untied 聚合 |
| `final/model_level_e_u_by_family_summary.csv` | 8 | family 聚合 |

## E / U

| 文件 | 行数 | 内容 |
|---|---:|---|
| `e/affine_w_rank_budget_clean.csv` | 240 | E 侧30对×8个标准 rank |
| `u/unembed_w_rank_budget_clean.csv` | 240 | U 侧30对×8个标准 rank |
| `e/affine_w_rank_budget_summary.csv` | 8 | E 侧 rank 聚合 |
| `u/unembed_w_rank_budget_summary.csv` | 8 | U 侧 rank 聚合 |
| `e/affine_w_rank_budget_boundaries.csv` | 30 | E 侧逐 pair 边界 |
| `u/unembed_w_rank_budget_boundaries.csv` | 30 | U 侧逐 pair 边界 |
| `e/affine_task6_decomposition_svd.csv` | 30 | E 侧 D/P/R 与谱 |
| `u/unembed_task6_decomposition_svd.csv` | 30 | U 侧 D/P/R 与谱 |
| `e/affine_pred_delta_common_spectrum.csv` | 30 | E 侧 common-k 谱 |
| `u/unembed_pred_delta_common_spectrum.csv` | 30 | U 侧 common-k 谱 |

Hybrid 表未列入当前数据：此前结果只覆盖部分 BI-clean pair，必须完整重跑30对后才能恢复。
