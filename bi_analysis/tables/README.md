# bi_analysis/tables：BI-clean 30

| 文件 | 行数 | 内容 |
|---|---:|---|
| `affine_lora_budget_summary.csv` | 30 | 每 pair 的 E/U affine、谱和 W-rank 摘要 |
| `affine_lora_by_tied_summary.csv` | 2 | tied / untied 聚合 |
| `affine_lora_by_family_size_summary.csv` | 8 | family 聚合 |
| `affine_effective_subspace.csv` | 43 | tied 一侧、untied E/U 两侧的有效子空间指标 |
| `delta_w_r2_one_minus.csv` | 30 | identity/full-affine 的变化量口径 |
| `task6_decomposition_e.csv` | 30 | E 侧 D/P/R 分解 |
| `task6_decomposition_u.csv` | 30 | U 侧 D/P/R 分解 |
| `dpr_common_spectrum_e.csv` | 30 | E 侧 common-k 谱 |
| `dpr_common_spectrum_u.csv` | 30 | U 侧 common-k 谱 |

标准 W-rank 全量表位于 `../../__tep/affine/tables/{e,u}/`，统一为30对×8档。旧 hybrid 结果未覆盖30对，已撤下。

重建主表：

```bash
python3 bi_analysis/scripts/build_affine_lora_budget_30.py
```
