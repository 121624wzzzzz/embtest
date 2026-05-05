# IJCAI 代码功能与实验结果摘要

## 实验 1：全局几何相似性

入口：`src/exp1_global_geometry/run_exp1_v4.py`。

功能：读取模型 E/U 矩阵，按同 tokenizer token id 或跨 tokenizer token 字符串对齐，采样最多 20,000 个 token 和 5,000,000 个 token pair，对 E 和 U 分别计算 cosine、euclidean、squared euclidean 的距离向量 Pearson 相关，即 GCorr。

结果口径：原始合并结果为 297 组模型对比，其中 Base-Instruct 19 组、主系列内两两 189 组、跨系列 108 组。Base-Instruct 使用 10 次 bootstrap，其余主要使用 100 次 bootstrap。默认 `n_tokens=20000`、`n_pairs=5000000`，跨 tokenizer 时 token 数由公共 token 集决定。

主要观察：

- Base-Instruct 对整体 GCorr 很高，说明微调前后 E/U 几何基本保持。
- 同系列不同规模仍保留较强结构相似性，但规模差越大通常越弱。
- 跨系列相似性明显下降。
- 欧氏距离指标在多个场景下比 cosine 更稳定，适合作为正文重点指标之一。

## 实验 2：跨模型仿射关系

入口：`src/exp2_affine_cross_model/run_affine_cross_model.py`。

功能：对 Qwen3、Qwen2.5、Llama、Gemma2 四个系列内的模型两两拟合 E_B ≈ E_A @ A^T + b 和 U_B ≈ U_A @ A_U^T + b_U；同时做每个模型内部 E->U 仿射拟合。公共 token 至少 5,000，拟合最多采样 24,000 行。

结果：跨模型 189 对，模型内 E->U 39 个。按系列均值：

- Gemma2：E/U 完全 tied，因此 E 与 U 的结果一致。
- Llama：R2_E_mean=0.3835，R2_U_mean=0.4235，U-U 略强。
- Qwen2.5：R2_E_mean=0.3129，R2_U_mean=0.3173，U-U 略强但差距小。
- Qwen3：R2_E_mean=0.3206，R2_U_mean=0.3901，U-U 明显更强。

tied/untied 分组显示：untied 模型内 E->U 平均 R2 约 0.2922，而 tied 模型内 E->U 为 1；untied 跨模型 U-U 平均 R2 约 0.5313，高于 E-E 的 0.4726。

## 模型情况

实验 1 覆盖 Qwen3、Qwen2.5、Llama、Gemma2、Mistral、Yi、DeepSeek、Qwen3-MoE。实验 2 主分析聚焦 Qwen3、Qwen2.5、Llama、Gemma2 四个系列。逐模型 hidden dim、vocab、声明 tied、实测 tied 和最终标签见 `results_summary/exp1_model_tag_audit.csv`。

## 可疑点与注意事项

- tied/untied 不能按模型名字推断，必须使用 `actual_tied`。
- 若模型没有显式 U，代码会使用 `U=E.copy()`，这会导致 `actual_tied=True`；报告中应说明这是按权重可见性得到的实测标签。
- `docs/source_notes/` 中保留了原始分析草稿，其中可能有“待确认”表述；clean 仓的正式结论以本文件和 `docs/model_tag_audit.md` 为准。

## 关键结果文件

- `results_summary/exp1_global_summary.csv`
- `results_summary/exp1_model_tag_audit.csv`
- `results_summary/affine_cross_model_summary.csv`
- `results_summary/affine_intra_model_EU_summary.csv`
- `results_summary/untied_comparison_summary.csv`
