# Scripts：BI-clean 30

所有全量入口统一使用 `--all-clean`，含义为35个注册 BI pair 排除5个异常 pair 后的30对。

| 脚本 | 用途 |
|---|---|
| `compute_affine_pred_delta_svd.py` | D/P/R decomposition 与谱指标；默认排除5个异常 pair |
| `evaluate_w_rank_budget.py` | W-form 与 affine-only 同参数比较 |
| `augment_w_rank_budget_raw_scale.py` | 补充 raw-scale 指标 |
| `evaluate_pred_delta_common_spectrum.py` | D/P/R common-k spectrum |
| `evaluate_affine_effective_subspace.py` | 有效 affine 子空间 |
| `evaluate_hybrid_affine_w_budget.py` | hybrid；恢复结果前必须完整运行 `--all-clean` |
| `verify_metrics.py` | 验证当前 BI-clean 30 统计与 Task1-6 行数 |

不要在结果文件中混合部分模型补算。新结果必须先验证包含30个唯一 `model_a`。
