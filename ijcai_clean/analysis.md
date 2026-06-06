# 项目分析备忘

本文档用于保留对当前 embedding / unembedding 几何项目的阶段性分析结论。它偏向研究判断和结果解读；具体指标定义、脚本入口和自动生成结果分别见 `docs/methods_and_metrics.md`、`README.md` 与 `results/`。全仓分析口径、BI 分层和特殊案例排错统一见 [`../docs/ANALYSIS_SCOPES_AND_SPECIAL_CASES.md`](../docs/ANALYSIS_SCOPES_AND_SPECIAL_CASES.md)。

## 项目主线

当前项目围绕不同大语言模型的 embedding matrix `E` 与 unembedding / lm_head matrix `U` 展开，核心问题是：不同模型、同系列模型、Base-Instruct 模型之间的词向量空间是否存在稳定几何关系。

已有任务大致分为：

- Task1：Base-Instruct pair 的 GCorr 几何相关性。
- Task2：同一模型系列内部的 GCorr。
- Task3：跨模型系列、跨规模桶的 GCorr。
- Task4：MoE / 跨 family 的 GCorr。
- Task5：对 Task1-4 pair 并集做仿射关系分析，拟合 `Y ~= X A + b`。
- Task6 / Base-Instruct full-vocab 诊断：对 `configs/base_instruct_pairs.yaml` 中的 Base-Instruct pair 使用完整词表 id 对齐，进一步分析仿射矩阵 `A`、`A-I` 和 `E_instruct - E_base` 的 SVD 低秩结构。

## Task5 仿射关系的理解

Task5 的核心实验是给定两个矩阵 `X` 和 `Y`，拟合：

```text
Y ~= X A + b
```

其中 `A` 是线性变换矩阵，`b` 是平移项。主要用 `R2` 和 `rel_err` 衡量拟合效果。`R2` 越接近 1，说明两个空间之间越接近一个全局仿射关系。

对 Base-Instruct 这类同 tokenizer、同 hidden dimension 的 pair，更适合使用完整词表 id 对齐，而不是采样 token 行。当前 Task6 脚本 `scripts/run_task6_base_instruct_full_vocab_affine.py` 已经按这个方式重算，并输出到：

- `results/task6_base_instruct_full_vocab/summary_pair_base_instruct_full_vocab.csv`
- `results/task6_base_instruct_full_vocab/base_instruct_full_vocab_affine_report.md`

## BI Affine 研究解释

BI full-vocab affine 的研究性叙事已经移到 [`../bi_analysis/notes/03_full_vocab_affine_geometry.md`](../bi_analysis/notes/03_full_vocab_affine_geometry.md)，包括：

- Base-Instruct full-vocab `R2_E` 分组结果。
- `A` 与单位阵的关系。
- `E_delta = E_instruct - E_base` 与 `A_delta = A - I` 的 SVD 低秩分析。
- `rank / hidden_dim`、`effective_rank / hidden_dim`、归一化后的线性关系。
- `energy_at_1pct_h / 5pct_h / 10pct_h` 等相对 hidden dimension 指标。

本文档只保留代码项目层面的任务入口和结果位置说明；BI 口径、异常排除和论文叙事统一在 `bi_analysis/` 与 [`../docs/ANALYSIS_SCOPES_AND_SPECIAL_CASES.md`](../docs/ANALYSIS_SCOPES_AND_SPECIAL_CASES.md) 中维护。
