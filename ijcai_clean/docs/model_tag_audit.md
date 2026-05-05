# IJCAI tied / untied 标签审计

## 判定规则

`ijcai` 的模型来自 HuggingFace 权重，因此最终标签以实际矩阵为准：

- 声明值：`config.tie_word_embeddings`，在结果 CSV 中为 `is_tied_a/is_tied_b`。
- 实测值：`actual_tied = np.allclose(E, U, rtol=1e-5, atol=1e-5)`，在结果 CSV 中为 `actual_tied_a/actual_tied_b`。
- 本审计采用 `actual_tied` 作为最终 tied / untied 标签；声明值只作为辅助证据。

关键代码位于 `src/exp1_global_geometry/run_exp1_v4.py`：加载 `model.embed_tokens.weight` / `transformer.wte.weight` 作为 E，加载 `lm_head.weight` / `output.weight` 作为 U；如果找不到 U 则使用 `U = E.copy()`，随后计算 `np.allclose(E, U)`。

## 审计输出

- `results_summary/exp1_model_tag_audit.csv`：逐模型列出 family、hidden dim、vocab、声明 tied、实测 tied、最终标签和异常说明。
- `results_summary/untied_comparison_summary.csv`：实验 2 中 tied/untied 分组的 R² 汇总。

## 主要结论

- Gemma2 系列在当前结果中 E/U 实测 tied，模型内 E->U 的 R² 为 1。
- Qwen 与 Llama 存在 tied 和 untied 混合情况，必须按具体模型的 `actual_tied` 标注，不能只按模型家族或 Base/Instruct 名称推断。
- 如果 `is_tied` 与 `actual_tied` 不一致，报告与论文表述应优先使用 `actual_tied`，并在脚注说明声明值与实际权重的差异。
