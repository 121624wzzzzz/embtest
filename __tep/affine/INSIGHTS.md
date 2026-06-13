# Affine / Low-Rank Update：最终洞察

> 本文件沿用早期 affine final 表口径（BI-main-26）：BI-full 35 排除 `Gemma-3-1B` 与 `Gemma-4-*` 5 对后得 BI-clean 30，再去掉 extended 4 对（MoE/DeepSeek）即为 n=26。当前全仓 BI-full 35 / BI-clean 30 / BI-main-26 关系见 [`../../docs/分析口径与特殊案例.md`](../../docs/分析口径与特殊案例.md)。
> 记 `D=Y_c-X_c`，`P=X_c(A-I)`，`R=D-P`。核心指标是 `P/D = ||P||_F^2 / ||D||_F^2`。

## 1. 故事线

最初的观察是：Base→Instruct 的 embedding 可以被高 R2 affine map 拟合，而且 `A-I` 的谱比 naive delta 更集中。后续补算把这个故事修正得更稳：

1. **普通 R2 很高，但不够说明问题。** `identity_R2 = 1-||D||^2/||Y_c||^2` 已经接近 1，所以 `full_affine_R2` 的绝对提升很小。
2. **真正要看 update-scale。** 用 `P/D` 衡量完整 affine component 在小差分 `D` 中占多少。
3. **E 侧有边界。** E 的 `P/D` median 只有 **0.120**，residual 常常主导；affine 优势主要在低 W-rank 参数预算。
4. **U 侧救回主叙事。** U/lm_head 的 `P/D` median 为 **0.315**；在 untied 模型中为 **0.462**（main 9 对，旧口径；含 extended 4 对后 BI-clean 30 的 untied 13 对中位降至 **0.319**，因扩展的 4 对 MoE 的 U P/D 偏低，见 [`../../docs/分析口径与特殊案例.md`](../../docs/分析口径与特殊案例.md) §4.1）。
5. **tied/untied 是结构性解释。** tied 模型 E/U 是同一个矩阵，结果自然一致；untied 模型中，E 侧 affine 弱而 U 侧 affine 强。
6. **hybrid 给出实用意义。** `affine low-rank(P) + W low-rank(R)` 在 U 侧小到中等 rank 预算下非常稳定。

## 2. 关键数字

| 分组 | n | E `P/D` median | U `P/D` median | E `P` rank95/h median | U `P` rank95/h median | E rW1 wins | U rW1 wins | E hybrid stable | U hybrid stable |
|------|---:|---------------:|---------------:|----------------------:|----------------------:|------------:|------------:|----------------:|----------------:|
| tied | 17 | 0.296 | 0.296 | 0.453 | 0.453 | 15/17 | 15/17 | 15/17 | 15/17 |
| untied | 9 | 0.049 | **0.462** | 0.518 | **0.198** | 0/9 | **9/9** | 0/9 | **9/9** |

按模型族：

| family | n | E `P/D` med | U `P/D` med | U/E med | E rW8 wins | U rW8 wins |
|--------|---:|------------:|------------:|--------:|-----------:|-----------:|
| Qwen2.5 | 7 | 0.073 | **0.486** | 6.54x | 1/7 | 2/7 |
| Qwen3 | 5 | 0.296 | 0.308 | 1.00x | 3/5 | 5/5 |
| Qwen3.5 | 4 | 0.117 | 0.120 | 1.00x | 3/4 | 4/4 |
| Llama-3.1 | 2 | 0.065 | **0.445** | 7.16x | 0/2 | 2/2 |
| Llama-3.2 | 2 | 0.405 | 0.405 | 1.00x | 2/2 | 2/2 |
| Gemma-2 | 3 | 0.320 | 0.320 | 1.00x | 2/3 | 2/3 |
| Gemma-3 | 3 | 0.101 | 0.101 | 1.00x | 3/3 | 3/3 |

## 3. E 侧结论

E/input embedding 侧的结论要写得克制：

- full affine R2 高，但 identity R2 也高。
- `P/D` median **0.120**，说明完整 centered delta 中 residual 往往很大。
- `P` 自身谱确实比 `D` 更集中：common-k 下 `C_P(k) > C_D(k)`，但对完整 delta 的绝对解释量要乘上 `P/D`。
- 同 W-rank 参数预算下，affine-only wins：
  - rW=1: **15/26**
  - rW=2: **15/26**
  - rW=4: **14/26**
  - rW=8: **14/26**
- 因此 E 侧适合写成：**low-rank affine component exists, but the advantage is budget-window and family dependent.**

## 4. U/lm_head 侧结论

U 侧是当前最强结果：

- `P/D` mean/median = **0.323 / 0.315**，明显高于 E 的 **0.202 / 0.120**。
- affine-only wins：
  - rW=1: **24/26**
  - rW=2: **24/26**
  - rW=4: **22/26**
  - rW=8: **20/26**
- hybrid wins：
  - rW=1: **24/26**
  - rW=2/4/8/16/32/64: **26/26**
- U 侧 small-budget stable hybrid：**24/26**；两个例外是 `Qwen2.5-1.5B`、`Qwen2.5-3B` 在 rW=1 退回 pure W，但 rW>=2 也被 hybrid 救回。

论文主句可以写：

> The affine structure is substantially stronger on the unembedding side. In untied models, the input embedding update is mostly residual-like, whereas the lm_head update contains a large, spectrally concentrated global affine component.

## 5. tied/untied 解释

tied 是这条故事线的关键调节变量：

- **tied**：E=U，共享同一个矩阵，所以 E/U 指标一致；大多数 tied 模型 affine 表现好，但 Qwen2.5-1.5B/3B 是低 rank 预算例外。
- **untied**：E/U 分裂；E 侧 `P/D` median **0.049**，U 侧 **0.462**。这解释了为什么只看 E 会觉得 affine 拉胯，而补上 U 后故事变强。

这不是纯尺寸规律，而是尺寸、模型族和 tied/untied 的耦合。大尺寸 Qwen2.5、Llama-3.1、Qwen3/3.5 后段多为 untied，因此它们最能展示 E/U 非对称。

## 6. 推荐写法

强版本：

> Base-to-Instruct vocabulary-side updates exhibit a low-dimensional global affine component. This component is modest and family-dependent in input embeddings, but becomes substantially stronger in unembeddings, especially for untied architectures. Weight tying collapses the two views; once untied, the lm_head update is much more affine and spectrally concentrated than the input embedding update.

边界版本：

> High affine R2 alone is not the evidence for low rank, because Base and Instruct matrices are already close. The meaningful evidence comes from update-scale decomposition and parameter-matched low-rank budgets.

## 7. 主要证据文件

| 文件 | 用途 |
|------|------|
| [`tables/final/model_level_e_u_affine_lora_summary.csv`](tables/final/model_level_e_u_affine_lora_summary.csv) | 每模型总表 |
| [`tables/final/model_level_e_u_by_tied_summary.csv`](tables/final/model_level_e_u_by_tied_summary.csv) | tied/untied 汇总 |
| [`tables/final/model_level_e_u_by_family_size_summary.csv`](tables/final/model_level_e_u_by_family_size_summary.csv) | 模型族/尺寸汇总 |
| [`tables/e/affine_w_rank_budget_sweep_all_main.csv`](tables/e/affine_w_rank_budget_sweep_all_main.csv) | E 侧 W-rank budget |
| [`tables/u/unembed_w_rank_budget_sweep_all_main.csv`](tables/u/unembed_w_rank_budget_sweep_all_main.csv) | U 侧 W-rank budget |
| [`tables/e/affine_hybrid_w_budget_summary.csv`](tables/e/affine_hybrid_w_budget_summary.csv) | E 侧 hybrid summary |
| [`tables/u/unembed_hybrid_w_budget_summary.csv`](tables/u/unembed_hybrid_w_budget_summary.csv) | U 侧 hybrid summary |

## 8. 阅读顺序

1. 本文件。
2. [`analysis/tasks/task6_pred_delta_probe.md`](analysis/tasks/task6_pred_delta_probe.md)：完整实验流水线和 U 侧补算。
3. [`analysis/theory/affine_decomposition_proof.md`](analysis/theory/affine_decomposition_proof.md)：理论分解。
4. [`tables/README.md`](tables/README.md)：结果表索引。
