# Affine / W-rank Budget：BI-clean 30

当前30对总体结果：

| 指标 | E | U |
|---|---:|---:|
| P/D median | 0.1080 | 0.3098 |
| affine wins @ W-rank 1 | 16/30 | 27/30 |
| affine wins @ W-rank 8 | 15/30 | 22/30 |

untied 13对的 P/D median 为 E 0.0491、U 0.3188，说明 U/lm_head 更新中的全局 affine component 更强。完整逐 pair 数据见 `affine_lora_budget_summary.csv`；标准 rank sweep 见 `../../__tep/affine/tables/`。

Hybrid 旧结果未覆盖全部30对，因此不在当前报告中使用。
