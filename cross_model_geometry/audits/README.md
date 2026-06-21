# Audit Metadata

`cross_model_geometry/audits/` 保存模型抽取和 tied/untied 检查产生的审计元数据。这里的 JSON 不是 Task1-6 实验结果，也不是论文指标输出；它们用于确认哪些模型已经抽取、E/U 维度是否合理、权重键名来自哪里，以及 tied 声明和实际矩阵状态是否一致。

## 文件说明

| 文件 | 用途 | 常见阅读场景 |
|------|------|--------------|
| `all_models_summary.json` | 每个已抽取模型的标准化维度、标准化 key、原始 embedding / lm_head key、`tie_word_embeddings` 声明。 | 检查某个模型的 E/U 来源、vocab / hidden size、lm_head 是否独立存在。 |
| `model_tied_untied_audit.json` | 按 tied / untied / physical matrix 数量汇总模型列表。 | 快速查看当前模型池中 tied 与 untied 的分布。 |
| `model_tied_matrix_summary.json` | 每模型的 tied 状态、矩阵维度、物理矩阵数量和估算体积。 | 排查大模型矩阵体量、物理 E/U 是否共享、磁盘规模。 |

## 和主线实验的关系

主线实验通常直接读取：

- `extracts/<model>.safetensors`
- `extracts/<model>.info.json`
- `configs/*.yaml`

`all_models_summary.json` 是把每个模型的抽取信息聚合到一个文件中，方便人工审计。实验代码计算 `actual_tied` 时仍以真实 E/U 矩阵比较为准，而不是只相信 JSON 中的配置声明。

## 更新来源

这些文件来自仓库级工具或一次性审计脚本：

- `all_models_summary.json` 由 `tools/get_model_useful.py` 的抽取流程聚合生成。
- `model_tied_untied_audit.json` 和 `model_tied_matrix_summary.json` 是 tied / untied 专项审计产物，用于辅助解释模型池状态。

如果重新下载或重新抽取模型，优先检查 `extracts/*.info.json` 与本目录 JSON 是否同步。
