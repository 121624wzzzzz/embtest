# BI Analysis：BI-clean 30

本目录统一使用35对注册 Base→Instruct pair 排除5对异常后的 **BI-clean 30**（17 tied + 13 untied）。不再维护更早的小规模分析层级。

## 入口

- `bi_pairs.yaml`：35对注册信息；30对标记 `clean`，5对标记 `excluded`。
- `tables/affine_lora_budget_summary.csv`：30行 E/U affine 与 W-rank 总表。
- `tables/affine_effective_subspace.csv`：17 tied×1 + 13 untied×2 = 43行。
- `tables/task6_decomposition_{e,u}.csv`：E/U 各30行。
- `tables/dpr_common_spectrum_{e,u}.csv`：E/U 各30行。
- `notes/02_affine_effective_update_insight.md`：有效 affine 子空间解释。
- `notes/03_full_vocab_affine_geometry.md`：Task6 full-vocab 解释。

`__tep/` 与本目录现在使用相同的 BI-clean 30 口径。旧的部分 pair 中间表和不完整 hybrid 结果已经移除。
