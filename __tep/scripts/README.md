# Scripts

当前主线脚本：

| 脚本 | 用途 |
|------|------|
| [`compute_affine_pred_delta_svd.py`](compute_affine_pred_delta_svd.py) | E/U 侧 D/P/R decomposition 与谱指标 |
| [`evaluate_w_rank_budget.py`](evaluate_w_rank_budget.py) | 同 W-form LoRA rank 预算比较 affine-only（含 raw 尺度 gain） |
| [`augment_w_rank_budget_raw_scale.py`](augment_w_rank_budget_raw_scale.py) | 给已有 W-rank sweep 补 raw 尺度可解释比例与汇总 |
| [`evaluate_pred_delta_common_spectrum.py`](evaluate_pred_delta_common_spectrum.py) | common-k spectrum：D vs P |
| [`evaluate_hybrid_affine_w_budget.py`](evaluate_hybrid_affine_w_budget.py) | affine + W residual hybrid oracle |
| [`verify_metrics.py`](verify_metrics.py) | 早期关键数字复核 |

旧参数匹配、证明诊断和边界验证脚本在 [`archive/`](archive/)。
