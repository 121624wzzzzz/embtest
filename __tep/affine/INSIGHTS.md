# Affine / Low-Rank Update：BI-clean 30 结论

## 口径

- 35 对注册 BI pair 排除 5 对异常，保留30对。
- tied 17 对；untied 13 对。
- tied pair 的 E/U 是同一矩阵；untied pair 分别报告 E 与 U。
- 普通 full-matrix R2 很高但不是主要证据，核心指标是 update-scale `P/D` 和同参数 W-rank 比较。

## Update-scale

| 分组 | n | E P/D median | U P/D median |
|---|---:|---:|---:|
| 全部 | 30 | 0.1080 | 0.3098 |
| tied | 17 | 0.2960 | 0.2960 |
| untied | 13 | 0.0491 | 0.3188 |

untied 模型中，U/lm_head 的全局 affine component 明显强于 E/input embedding；tied 模型因为共享矩阵，两侧结果一致。

## 同参数 W-rank 比较

| W rank | E affine wins | U affine wins | E ratio median | U ratio median |
|---:|---:|---:|---:|---:|
| 1 | 16/30 | 27/30 | 1.388 | 2.846 |
| 2 | 16/30 | 26/30 | 1.251 | 2.279 |
| 4 | 15/30 | 24/30 | 1.111 | 1.802 |
| 8 | 15/30 | 22/30 | 1.005 | 1.431 |
| 16 | 13/30 | 20/30 | 0.866 | 1.207 |
| 32 | 8/30 | 15/30 | 0.779 | 1.006 |
| 64 | 3/30 | 7/30 | 0.629 | 0.855 |
| 128 | 1/30 | 1/30 | 0.447 | 0.763 |

结论边界明确：E 侧优势主要位于很小的 rank budget；U 侧优势更强、持续到更大的 budget，但也不会无限保持。

## 谱结论

BI-clean 30 上，`A-I` 的 mean rank95/h 为 0.4528，原始 `E_instruct-E_base` 为 0.7837。全局仿射变换相对单位阵的偏移比直接矩阵差异更集中。

## 数据入口

- `tables/final/model_level_e_u_affine_lora_summary.csv`：30行逐 pair 总表。
- `tables/e/affine_w_rank_budget_clean.csv`：E 侧30×8。
- `tables/u/unembed_w_rank_budget_clean.csv`：U 侧30×8。
- `tables/e/affine_task6_decomposition_svd.csv`、`tables/u/unembed_task6_decomposition_svd.csv`：30对分解。

旧 hybrid 数据未覆盖30对，因此不再作为当前证据。
