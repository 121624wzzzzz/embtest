# BI Full-Vocab Affine Geometry

本文档承接 `ijcai_clean` Task6 / Base-Instruct full-vocab 诊断，专门记录 BI affine 关系的研究性解释：全局仿射 `A`、`A-I`、`E_instruct - E_base` 的谱结构、归一化 rank / effective-rank，以及 energy@相对 hidden dimension。

口径说明：

- 全仓 BI 35/30/26 口径、排除规则与特殊案例见 [`../../docs/分析口径与特殊案例.md`](../../docs/分析口径与特殊案例.md)。
- 本文中的 “BI-clean / non-excluded” 指排除 Gemma-3-1B 与 Gemma-4 四对后的 30 对。
- 原始 Task6 结果见 `../ijcai_clean/results/task6_base_instruct_full_vocab/`。

## 1. Full-Vocab Affine 主要结果

完整词表分析显示，大部分 Base-Instruct pair 的 embedding 空间之间存在很强的仿射关系：

```text
all Base-Instruct: R2_E mean ~= 0.9434
Qwen only:         R2_E mean ~= 0.9963
Llama only:        R2_E mean ~= 0.9880
Gemma only:        R2_E mean ~= 0.8305
DeepSeek only:     R2_E mean ~= 1.0000
```

全体 35 组的 `R2_E` 均值约为 `0.9434`。Qwen、Llama 和 DeepSeek V3/V3.1 的 Base-Instruct 变化基本可以被一个全局仿射变换很好解释。Gemma 组均值明显偏低，主要来自 Gemma-3-1B 和 Gemma-4 系列的异常表现。

异常处理只保留结论摘要：

| case | 现象 | 处理 |
|------|------|------|
| `Gemma-3-1B -> Gemma-3-1B-Instruct` | full-vocab affine `R2_E≈0.3751`，row cosine 约 `0.279`，明显脱离其他 Gemma-3 大模型 | 排除出 BI-clean；原因见 [`gemma3_1b_rewrite`](../../docs/分析口径与特殊案例.md#61-gemma3_1b_rewrite) |
| `Gemma-4-* -> *-Instruct` | `R2_E≈0.6682-0.7759`，Base E near-unit row-norm gauge，raw Euclidean 受径向标尺影响 | 排除出 BI-clean；原因见 [`gemma4_checkpoint_gauge`](../../docs/分析口径与特殊案例.md#62-gemma4_checkpoint_gauge) |

## 2. A 矩阵与单位阵的关系

最初的直觉是：Base 到 Instruct 的变化可能接近“单位阵附近的小旋转 / 小扰动”。当前结果部分支持这个直觉，但需要更精确表述。

对 Qwen / Llama，`A` 通常非常接近单位方向：

- `identity_cosine` 接近 1。
- `rel_A_minus_I_over_I` 较小。
- `offdiag_norm_over_A` 较小，说明坐标混合有限。
- `R2` 很高，说明全局仿射关系足够解释大部分变化。

但 `A` 并不严格是正交旋转。`A^T A` 与 `I` 仍有可测误差，所以更准确的说法是：

> Base-Instruct 的 embedding 空间变化通常接近单位阵附近的低幅度全局仿射扰动，而不是严格的正交旋转。

## 3. SVD 低秩分析

当前重点比较两类差异矩阵：

```text
E_delta = E_instruct - E_base
A_delta = A - I
```

对二者都做 SVD 能量分析，关注：

- `rank_50 / rank_80 / rank_90 / rank_95 / rank_99`
- `effective_rank`
- `energy_at_k`
- `energy_at_1pct_h / 5pct_h / 10pct_h`

这里的能量指 squared singular value energy。`rank_95` 表示解释 95% 能量所需的最小秩，`effective_rank` 是 entropy-based 有效秩。

## 4. 秩与 Hidden Dimension 的关系

绝对 rank 明显随 hidden dimension 增大而增大，因此不能只看 rank 原值。更合理的比较方式是看：

```text
rank / hidden_dim
effective_rank / hidden_dim
energy@相对 hidden dimension
```

全体 35 组中：

```text
E_delta rank95 mean ~= 2780, 约 0.800 * hidden_dim
A-I     rank95 mean ~= 1626, 约 0.473 * hidden_dim

E_delta rank99 mean ~= 3321, 约 0.950 * hidden_dim
A-I     rank99 mean ~= 2333, 约 0.690 * hidden_dim

E_delta effective_rank mean ~= 1868, 约 0.486 * hidden_dim
A-I     effective_rank mean ~= 1060, 约 0.271 * hidden_dim
```

排除 Gemma-3-1B 和 Gemma-4 后：

```text
E_delta rank95 / hidden_dim mean ~= 0.784
A-I     rank95 / hidden_dim mean ~= 0.453

E_delta rank99 / hidden_dim mean ~= 0.946
A-I     rank99 / hidden_dim mean ~= 0.675

E_delta effective_rank / hidden_dim mean ~= 0.472
A-I     effective_rank / hidden_dim mean ~= 0.254
```

结论：

> hidden dimension 决定绝对 rank 的规模；归一化后，`A-I` 的秩和有效秩系统性低于 `E_instruct - E_base`。直接 embedding 差异更高秩、更分散，而最优仿射变换相对单位阵的偏移集中在更低维的子空间。

## 5. 归一化后的线性关系

把 rank 或 effective rank 除以 hidden dimension 后，`A-I` 与 `E_delta` 的关系不是一个全局稳定线性律，但在同一模型族内更接近线性。

主分析组中：

```text
(A-I effective_rank / h) ~= -0.012 + 0.515 * (E_delta effective_rank / h)
R2 ~= 0.553
```

同一模型族内，`effective_rank / h` 的线性关系更强：

```text
Qwen2.5: R2 ~= 0.915
Qwen3:   R2 ~= 0.710
Qwen3.5: R2 ~= 0.961
Llama:   R2 ~= 0.999
Gemma-2: R2 ~= 0.823
```

因此可以写成：

> 同一模型族内，`A-I` 的归一化有效秩与 `E_instruct - E_base` 的归一化有效秩有较明显线性关系；但跨模型族混合后，family 差异会显著扰动这个关系。

`rank95 / rank99` 不如 `effective_rank` 稳定，原因是高阈值 rank 容易接近 hidden dimension 上限，出现饱和效应。

## 6. Energy@相对 Hidden Dimension

为了避免固定 `k` 对不同 hidden dimension 不公平，当前脚本新增了：

```text
energy_at_1pct_h
energy_at_5pct_h
energy_at_10pct_h
```

含义是：top `1% / 5% / 10%` hidden dimensions 的奇异方向解释了多少能量。

排除 Gemma-3-1B 和 Gemma-4 后，主分析组中位数为：

```text
energy@1%h:
E_delta median ~= 0.141
A-I     median ~= 0.329

energy@5%h:
E_delta median ~= 0.352
A-I     median ~= 0.544

energy@10%h:
E_delta median ~= 0.464
A-I     median ~= 0.696
```

这个结果很关键：

> 在相同的相对维度预算下，`A-I` 的能量明显比 `E_delta` 更集中。比如只使用 top `5%` hidden dimensions，`A-I` 中位数已经解释约 `54.4%` 能量，而 `E_delta` 约为 `35.2%`。

这比固定 `energy@100 / 500 / 1000` 更适合跨 hidden dimension 比较。

## 7. 推荐指标优先级

1. `energy_at_1pct_h / 5pct_h / 10pct_h`：最公平地比较不同 hidden dimension 下的谱集中程度。
2. `effective_rank / hidden_dim`：最适合概括为单个“有效维度”。
3. `rank80 / rank90 / rank95`：适合作为累计能量阈值的辅助证据。
4. `rank99`：保留，但不要作为主结论，因为容易接近满秩上限。
5. 固定 `energy@100 / 500 / 1000`：可用于补充，但不同 hidden dimension 间公平性较弱。

## 8. 阶段性研究结论

当前最稳的研究表述是：

> 对大多数非异常 Base-Instruct pair，embedding 空间之间存在高质量全局仿射关系。该仿射矩阵 `A` 通常接近单位阵方向，但不是严格正交旋转。与直接 embedding 差异 `E_instruct - E_base` 相比，`A-I` 的谱能量更集中、有效秩更低，说明 instruct tuning 引入的全局空间变换可以被更少的主方向概括。

更短版本：

> Base-Instruct 的 embedding 差异本身偏高秩，但其最优全局仿射变换相对单位阵的偏移是更低秩、更集中的。

## 9. 后续可补充

- 画 `energy threshold -> rank / hidden_dim` 的完整曲线，而不是只看 50/80/90/95/99。
- 画 `top p% hidden dim -> cumulative energy` 曲线，比较 `E_delta` 和 `A-I`。
- 对每个模型族分别拟合 `effective_rank / h` 的线性关系，避免跨 family 混合造成解释偏差。
- 若需要说明 Gemma / DeepSeek-V4 异常或排除原因，直接引用 [`../../docs/分析口径与特殊案例.md`](../../docs/分析口径与特殊案例.md)，避免在本文档重复维护排错叙述。
