# Tables

顶层只放索引；CSV 按用途分区。

| 子目录 | 用途 |
|--------|------|
| [`final/`](final/) | 论文主线优先引用的三张总表 |
| [`e/`](e/) | E/input embedding 侧 decomposition、spectrum、budget、hybrid 表 |
| [`u/`](u/) | U/lm_head 侧 decomposition、spectrum、budget、hybrid 表 |
| [`archive/`](archive/) | 早期 exploratory / support 表 |

## Final

| 文件 | 内容 |
|------|------|
| [`final/model_level_e_u_affine_lora_summary.csv`](final/model_level_e_u_affine_lora_summary.csv) | 每个模型一行，含 tied、E/U R2、D/P/R、谱集中性、LoRA budget、hybrid 指标 |
| [`final/model_level_e_u_by_tied_summary.csv`](final/model_level_e_u_by_tied_summary.csv) | tied vs untied 聚合 |
| [`final/model_level_e_u_by_family_size_summary.csv`](final/model_level_e_u_by_family_size_summary.csv) | 按模型族/尺寸聚合 |

## E / Input Embedding

| 文件 | 内容 |
|------|------|
| [`e/affine_task6_decomposition_svd.csv`](e/affine_task6_decomposition_svd.csv) | E 侧 D/P/R/mean decomposition 与 rank95/eff-rank |
| [`e/affine_w_rank_budget_sweep_all_main.csv`](e/affine_w_rank_budget_sweep_all_main.csv) | E 侧 affine-only vs W-form LoRA sweep |
| [`e/affine_pred_delta_common_spectrum.csv`](e/affine_pred_delta_common_spectrum.csv) | E 侧 common-k spectrum（D/P/R/P_abs，30 对） |
| [`e/affine_hybrid_w_budget.csv`](e/affine_hybrid_w_budget.csv) | E 侧 affine + W residual hybrid sweep |

## U / lm_head

| 文件 | 内容 |
|------|------|
| [`u/unembed_task6_decomposition_svd.csv`](u/unembed_task6_decomposition_svd.csv) | U 侧 D/P/R/mean decomposition 与 rank95/eff-rank |
| [`u/unembed_w_rank_budget_sweep_all_main.csv`](u/unembed_w_rank_budget_sweep_all_main.csv) | U 侧 affine-only vs W-form LoRA sweep |
| [`u/unembed_pred_delta_common_spectrum.csv`](u/unembed_pred_delta_common_spectrum.csv) | U 侧 common-k spectrum（D/P/R/P_abs，30 对） |
| [`u/unembed_hybrid_w_budget.csv`](u/unembed_hybrid_w_budget.csv) | U 侧 affine + W residual hybrid sweep |
