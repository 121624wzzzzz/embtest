# Task6 补充：仿射分解谱检验

## 问题

为了检验「$E_\Delta$ 比 $A-I$ 更分散」的来源，计算以下分解项：

$$
Y-X = (Y_c-X_c) + \mathbf{1}(\mu_Y-\mu_X)^\top
$$

以及中心化仿射分解：

$$
Y_c-X_c = X_c(A-I) + R,\qquad R:=Y_c-X_cA.
$$

对应谱对象：

| 名称 | Gram |
|------|------|
| raw delta | $(Y-X)^\top(Y-X)$ |
| centered delta | $(Y_c-X_c)^\top(Y_c-X_c)$ |
| pred delta | $(A-I)^\top X_c^\top X_c(A-I)$ |
| residual | $R^\top R$ |
| mean shift | $n(\mu_Y-\mu_X)^\top(\mu_Y-\mu_X)$ |

注意 raw delta 与 centered delta 的关系是精确正交分解：

$$
(Y-X)^\top(Y-X)=(Y_c-X_c)^\top(Y_c-X_c)+n(\mu_Y-\mu_X)^\top(\mu_Y-\mu_X),
$$

因为 $(Y_c-X_c)$ 的行和为 0。

## 结果（主分析组 n=26）

| 指标 | raw $E_\Delta$ | centered delta | pred $X_c(A-I)$ | $A-I$ | residual |
|------|----------------|----------------|-------------------|-------|----------|
| rank95 / h mean | 0.769 | **0.781** | 0.425 | 0.426 | **0.807** |
| rank95 / h median | 0.825 | **0.840** | 0.464 | 0.471 | **0.865** |
| eff_rank / h mean | 0.418 | **0.494** | 0.195 | 0.203 | **0.630** |
| energy@5%h mean | 0.387 | 0.350 | 0.610 | 0.600 | **0.265** |

能量占比：

| 项 | mean | median |
|----|------|--------|
| residual / centered-delta energy | **0.798** | **0.880** |
| mean-shift / raw-delta energy | 0.065 | 0.033 |

贴近度：

| 比较 | MAE(rank95/h) | corr(rank95/h) | corr(energy@5%h) |
|------|---------------|----------------|------------------|
| centered vs raw | **0.012** | — | — |
| residual vs centered | **0.027** | **0.946** | **0.840** |
| pred vs centered | 0.356 | 0.320 | 0.491 |

## Residual-aware 低秩效果（主分析组 n=26）

若只看 `A-I` 或 `pred = X_c(A-I)` 自身的谱，前 5% hidden 方向能量约 **0.60–0.61**；但若目标改成“用低秩 affine pred 去解释真实 centered delta”，必须乘上 pred 在 centered delta 中的能量占比

$$
w=\frac{\|X_c(A-I)\|_F^2}{\|Y_c-X_c\|_F^2}.
$$

| 口径 | mean | median | 含义 |
|------|------|--------|------|
| `A-I` energy@5%h | **0.600** | 0.628 | 参数偏移自身的低秩压缩效果 |
| `pred=X_c(A-I)` energy@5%h | **0.610** | 0.671 | 仿射预测项自身的低秩压缩效果 |
| `w = pred / centered-delta energy` | 0.202 | 0.120 | 完整仿射项在真实更新中的能量占比 |
| `w * C_pred(5%h)` | **0.150** | 0.083 | 低秩 affine pred 能解释的 centered-delta 能量 |
| raw 口径：`low pred / raw delta` | 0.132 | 0.080 | 再把 mean shift 计入分母 |
| raw 口径：`mean shift + low pred` | 0.197 | 0.108 | 若均值平移作为 rank-1 项单独保留 |

结论：`A-I` 在**参数空间 / 仿射预测项空间**确实更可压缩；但如果评价目标是完整 raw/centered delta，低秩 `A-I` 单独解释的能量会被 residual 占比显著稀释。这个结果支持更精确的叙事：`A-I` 是低秩的全局仿射更新表示，而不是完整 token-level delta 的低秩重构。

## LoRA-style 参数预算解释

如果适配器形式是 hidden-space 的共享低秩矩阵

$$
X \mapsto X(I+UV^\top)+b,\qquad U,V\in\mathbb{R}^{h\times r},
$$

那么它的目标对象正是 $A-I$，参数量约为 $2hr$（另加 bias $h$）。在这个口径下，比较的不是“rank-$r$ 的 `A-I` 能否重构完整 token-level delta”，而是：

> 同样 rank / 近似同样 LoRA 参数量下，低秩 hidden-space adapter 能保留多少全局仿射更新能量？

主组中取 $r=\lceil0.05h\rceil$，对应矩阵参数量约为 full $h\times h$ 的 **10%**。此时：

| 对象 | energy@5%h mean | median | 逐对优势 |
|------|-----------------|--------|----------|
| `A-I` | **0.600** | **0.628** | 26/26 高于 raw delta |
| raw `E_delta` | 0.387 | 0.397 | — |

因此，在“增加相同或近似相同 hidden-space LoRA 参数”的适配设定下，`A-I` 的谱集中性意味着 rank-$r$ adapter 会优先捕获更大比例的全局更新能量。这个结论与上一节不矛盾：上一节说明低秩 `A-I` 不等于完整 delta 重构；本节说明若目标是设计共享低秩仿射 adapter，`A-I` 是更自然也更可压缩的对象。

### Rank-budget 实际检验

派生脚本 [`../../../scripts/archive/evaluate_lora_rank_budget.py`](../../../scripts/archive/evaluate_lora_rank_budget.py) 用现有 Task6 分解量计算 rank-$r$ hidden-space affine adapter 的实际预测效果。取 $r=\lceil0.05h\rceil$，即约 **10%** full $h\times h$ 参数量：

| 指标 | mean | median | 说明 |
|------|------|--------|------|
| identity `Y≈X` 的 `R²` | 0.9889 | 0.9963 | Base 与 Instruct 本来很近，绝对 R² 提升会被压缩 |
| rank-$r$ affine adapter `R²` | **0.9907** | **0.9967** | 只用约 10% full 矩阵参数 |
| full affine `R²` | 0.9914 | 0.9969 | 上界：完整 $A-I$ |
| rank-$r$ 捕获的 update-error 降低 | 0.150 | 0.083 | 相对 identity 的真实更新误差降低 |
| rank-$r$ / full-affine 可改进量 | **0.610** | **0.671** | rank-$r$ adapter 捕获 full affine gain 的比例 |

逐对范围：rank-$r$ adapter 捕获 full affine 可改进量的 **20.7%–95.1%**，无 0 或负例。低端多为 full affine gain 本身很小的 pair（如 Llama-3.1、Qwen3.5-9B）；Gemma-2 与部分 Qwen2.5 pair 可达 **87%–95%**。

同 rank 的 token-level delta SVD 不是同参数量比较：rank-$r$ 的 `vocab×h` delta 需要 $r(n+h)$ 参数，而 rank-$r$ hidden adapter 只需约 $2hr$ 参数。主组中前者参数量是后者的 **8.3×–121.8×**（mean **37.8×**）。因此，LoRA-style 参数预算下应比较 hidden-space $A-I$，而不是 raw token delta。

### 参数量匹配：hidden affine vs token-level delta

进一步用 [`../../../scripts/archive/evaluate_param_matched_adapters.py`](../../../scripts/archive/evaluate_param_matched_adapters.py) 做参数量近似匹配：仍取 hidden affine rank $r_{\text{aff}}=\lceil0.05h\rceil$，参数量为

$$
P_{\text{aff}}=h+2hr_{\text{aff}},
$$

其中 $h$ 是 bias/mean shift，$2hr_{\text{aff}}$ 是 rank-$r$ hidden linear map 的两侧因子。token-level delta 的 rank-$r_\Delta$ 矩阵参数量为

$$
P_{\Delta}=h+r_\Delta(n+h),
\qquad
r_{\Delta}\approx\frac{2hr_{\text{aff}}}{n+h}.
$$

这里有一个离散化细节：token-level delta 的最小非零 rank 是 1。如果用“最接近同参数量”的整数 rank，主组 26 对在 5%h 档的 `delta_params / affine_params` median 为 **1.02**，但 mean 为 **1.13**，range 为 **0.81–2.33**，且 **18/26** 组会超出 hidden affine 参数量。这个口径可以看趋势，但不是严格不超预算比较。

| 指标 | hidden affine | param-matched token delta |
|------|---------------|---------------------------|
| `R²` mean | **0.99069** | 0.98979 |
| `R²` median | **0.99674** | 0.99658 |
| update-error gain mean | **0.150** | 0.120 |
| update-error gain median | **0.083** | 0.064 |
| 胜出 pair 数 | **15/26** | 11/26 |

如果强制 token-level delta 不超过 hidden affine 的参数量，即 $r_\Delta=\lfloor2hr_{\text{aff}}/(n+h)\rfloor$，5%h 档仍是 hidden affine 胜出 **15/26**；mean update gain 为 **0.150 vs 0.109**，median gain ratio 为 **1.17**（3 组因预算不足以购买 rank-1 token delta，no-overbudget delta 为 rank 0）。这说明“最近整数”口径没有制造 hidden affine 的优势；相反，它在不少低预算组给了 token-delta 额外参数。

按系列看，hidden affine 胜出集中在 Gemma-2/3、Llama-3.2、Qwen3 小中模型、Qwen3.5 小中模型；token-delta 胜出集中在 Qwen2.5 大模型、Llama-3.1、Qwen3/3.5 大模型。结论应写成：

> 在近似相同参数量下，hidden-space affine adapter 平均更优、且在多数 pair 上胜出，但并非无条件碾压；当 full affine gain 本身很小或 token-level residual 有强低秩结构时，param-matched token delta 可以更好。

这比单看 `A-I` 谱更强：它说明 `A-I` 的可压缩性在多数主组 pair 中确实转化为参数匹配的适配收益；同时也给出了边界，避免把“全局 affine 低秩”误写成“任何同参低秩更新都不如 affine”。

### 多 rank sweep

同一脚本支持一次读入模型后评估多个 hidden affine rank。完整主组结果见 [`../../tables/archive/affine_param_matched_adapter_rank_sweep_all_main.csv`](../../tables/archive/affine_param_matched_adapter_rank_sweep_all_main.csv)。本次 sweep 取

$$
r_{\text{aff}}/h\in\{0.1\%,0.2\%,0.5\%,1\%,2\%,5\%,10\%\}.
$$

注意极低 rank 下 token-level delta 的 $r_\Delta$ 最小为 1，因此“最近整数 rank”会让 token-delta 参数量可能远大于 hidden affine；例如 0.1%h 档的 `delta/affine` 参数比例 median 为 **7.98×**。从 1%h 起，参数量更接近（median 约 **1.19×**），比较更稳。

| `r_aff/h` | affine wins | mean gain ratio | median gain ratio | mean affine gain | mean delta gain | median param ratio |
|-----------|-------------|-----------------|-------------------|------------------|-----------------|--------------------|
| 0.1% | 13/26 | 0.96 | 0.98 | 0.080 | 0.079 | 7.98 |
| 0.2% | 14/26 | 1.20 | 1.18 | 0.091 | 0.079 | 4.23 |
| 0.5% | 15/26 | 1.55 | 1.50 | 0.107 | 0.080 | 1.85 |
| 1% | 15/26 | **1.76** | **1.63** | 0.120 | 0.084 | 1.19 |
| 2% | 15/26 | **1.78** | **1.52** | 0.133 | 0.094 | 1.02 |
| 5% | 15/26 | 1.51 | 1.25 | 0.150 | 0.120 | 1.02 |
| 10% | 14/26 | 1.40 | 1.05 | 0.163 | 0.146 | 0.98 |

同一 CSV 同时给出严格 no-overbudget 口径（$r_\Delta=\lfloor2hr_{\text{aff}}/(n+h)\rfloor$）：

| `r_aff/h` | affine wins | zero delta ranks | finite mean gain ratio | finite median gain ratio | mean affine gain | mean no-overbudget delta gain | median param ratio |
|-----------|-------------|------------------|------------------------|--------------------------|------------------|-------------------------------|--------------------|
| 0.1% | 25/26 | 25 | 0.39 | 0.39 | 0.080 | 0.000 | 0.13 |
| 0.2% | 24/26 | 24 | 0.46 | 0.46 | 0.091 | 0.000 | 0.08 |
| 0.5% | 19/26 | 18 | 0.73 | 0.42 | 0.107 | 0.003 | 0.04 |
| 1% | 17/26 | 13 | 1.10 | 0.44 | 0.120 | 0.013 | 0.33 |
| 2% | 16/26 | 6 | 1.81 | 1.33 | 0.133 | 0.052 | 0.90 |
| 5% | 15/26 | 3 | 1.37 | 1.17 | 0.150 | 0.109 | 0.93 |
| 10% | 14/26 | 1 | 1.30 | 1.04 | 0.163 | 0.144 | 0.97 |

这张表的作用不是说 rank-0 token delta 是一个有意义的强 baseline，而是指出参数预算的离散事实：在很多低预算设置里，hidden affine 已经能购买非零 rank，而 token-level delta 由于矩阵是 $n\times h$，连 rank 1 都买不起。若硬把 token delta 设为 rank 1，它就已经不是同预算比较。

分组现象稳定：

- hidden affine 在 Gemma-2/3、Llama-3.2、Qwen3 小中模型、Qwen3.5 小中模型上稳定胜出。
- param-matched token delta 在 Llama-3.1、Qwen2.5 大模型、Qwen3/3.5 大模型上稳定胜出。
- 随着 rank 增大，token-delta baseline 也会吃到更多 residual 的低秩结构，因此 affine 的相对优势从 1–2%h 的峰值回落。

因此更稳的结论是：hidden affine adapter 的优势在**低到中等 rank 预算**下最明显，尤其当 Base→Instruct 的变化主要表现为共享仿射扰动时；但若 token-level residual 自身低秩，参数匹配的 delta adapter 会追上甚至超过。

### W-form LoRA rank budget

上面的 $r_{\text{aff}}/h$ sweep 仍有一个问题：$2\%h$ 对大模型已经是很高的 hidden affine rank，而实际 LoRA 很少超过 rank 32 或 64。因此进一步用 W-form LoRA 的 rank 作为预算主轴。脚本 [`../../../scripts/evaluate_w_rank_budget.py`](../../../scripts/evaluate_w_rank_budget.py) 从 $r_W=1$ 开始扫到 128，其中 W-form token delta 的参数量为

$$
P_W=h+r_W(n+h),
$$

hidden affine 取同预算内最大 rank：

$$
r_{\text{aff}}=\left\lfloor\frac{r_W(n+h)}{2h}\right\rfloor,
\qquad
P_{\text{aff}}=h+2hr_{\text{aff}}\le P_W.
$$

完整 sweep 见 [`../../tables/e/affine_w_rank_budget_sweep_all_main.csv`](../../tables/e/affine_w_rank_budget_sweep_all_main.csv)，逐模型边界见 [`../../tables/e/affine_w_rank_budget_boundaries.csv`](../../tables/e/affine_w_rank_budget_boundaries.csv)。这里把“稳定不如 W”定义为：从某个 $r_W$ 开始，当前 rank 与后面两个 rank 连续三次满足 affine gain < W gain。

需要注意，$r_W=1$ 虽然常常能购买十几到几十维 hidden affine rank，但这在 $h=4096/5120/8192$ 的模型上仍只是 **0.1%–0.7%h** 的极低 rank。对于 Qwen2.5-14B、Llama-3.1-8B、Qwen3.5-9B 等组，`A-I`/affine prediction 的 rank50 本身达到数百，因此 $r_{\text{aff}}\approx 15$ 或 30 只能捕获 full affine gain 的很小一部分；这不是与“`A-I` 比 delta 更集中”矛盾，而是说明“更集中”主要是相对 delta，而不是说绝对 rank-16 就足够。

更精确地说，令 $D=Y_c-X_c$，$P=X_c(A-I)$，则 centered least-squares 下

$$
\|D\|_F^2=\|P\|_F^2+\|R\|_F^2,
\qquad
\text{full\_affine\_gain}=\frac{\|P\|_F^2}{\|D\|_F^2}.
$$

rank-$r_{\text{aff}}$ affine 的 update gain 是

$$
\frac{\sum_{i\le r_{\text{aff}}}\sigma_i(P)^2}{\|D\|_F^2}
=
\text{full\_affine\_gain}\cdot C_P(r_{\text{aff}}),
$$

而 $r_W=1$ 的 W-form delta gain 是 $C_D(1)=\sigma_1(D)^2/\|D\|_F^2$。因此 affine 要赢 W1，需要

$$
C_P(r_{\text{aff}})>\frac{C_D(1)}{\text{full\_affine\_gain}}.
$$

例如 Qwen2.5-14B 在 $r_W=1$ 时 $r_{\text{aff}}=15$，但 full_affine_gain 只有 **0.047**，$C_P(15)$ 只有 **0.060**，实际 affine gain 只有 **0.0028**；W1 gain 是 **0.0121**，所以需要 $C_P(15)>0.255$ 才能赢。这里不是 rank 数少一维的问题，而是 affine 分量本身在完整 delta 中占比小，并且其前 15 个 prediction 方向还不够集中。

| `r_W` | affine wins | mean gain ratio | median gain ratio | mean affine gain | mean W gain | median budgeted `r_aff` |
|-------|-------------|-----------------|-------------------|------------------|-------------|-------------------------|
| 1 | 15/26 | 2.09 | 1.88 | 0.129 | 0.079 | 30.5 |
| 2 | 15/26 | 1.60 | 1.63 | 0.141 | 0.096 | 62.0 |
| 4 | 14/26 | 1.28 | 1.45 | 0.153 | 0.117 | 125.0 |
| 8 | 14/26 | 1.07 | 1.23 | 0.164 | 0.142 | 250.0 |
| 16 | 13/26 | 0.91 | 0.96 | 0.174 | 0.172 | 501.0 |
| 32 | 8/26 | 0.78 | 0.86 | 0.184 | 0.209 | 975.5 |
| 64 | 3/26 | 0.66 | 0.72 | 0.193 | 0.259 | 1564.0 |
| 128 | 1/26 | 0.55 | 0.58 | 0.198 | 0.326 | 2058.0 |

边界分布：

- **11/26** 组从 $r_W=1$ 起就连续稳定不如 W：Llama-3.1、Qwen2.5 的 1.5B 及以上、Qwen3/3.5 的较大模型等。
- **12/26** 组在 $r_W\le16$ 内已经稳定不如 W（在上面 11 组之外，Gemma-2-2B 于 $r_W=4$ 过界）。
- **18/26** 组在 $r_W\le32$ 内稳定不如 W。
- **23/26** 组在 $r_W\le64$ 内稳定不如 W。
- 只有 **Llama-3.2-3B** 到 $r_W=128$ 仍未稳定落后，ratio@128 仍为 **1.06**。

因此，如果用更接近常见 LoRA 的 W-rank 口径，结论应进一步收紧：

> hidden affine 的优势主要存在于很低 W-rank 预算，尤其 $r_W=1$ 到 $8$；到 $r_W=16$ 已经接近均势，到 $r_W=32/64$ 时多数模型的 W-form low-rank delta 已经稳定追上或超过 affine。这个结果不削弱 $A-I$ 更谱集中的观察，而是说明 affine 的低秩优势有一个明确的参数预算窗口。

按模型族与尺寸排序后，规律更清楚（family-ordered CSV 见 [`../../tables/e/affine_w_rank_budget_family_ordered.csv`](../../tables/e/affine_w_rank_budget_family_ordered.csv)）：

- **Qwen2.5**：只有 0.5B 有低 rank 优势；1.5B 及以上从 $r_W=1$ 起稳定不如 W。
- **Qwen3**：0.6B/1.7B 有低 rank 优势，4B 优势窗口最大；8B/14B 从 $r_W=1$ 起不佳。
- **Qwen3.5**：0.8B/2B/4B 只在低 rank 有优势；9B 从 $r_W=1$ 起不佳。
- **Llama**：Llama-3.1 的 8B/70B 都不佳；Llama-3.2 的 1B/3B 是最强 affine 优势组。
- **Gemma-2/3**：整体多为低 rank 优势型，没有像 Llama-3.2 那样长窗口，也没有像 Qwen2.5 中大模型那样从 $r_W=1$ 起全面失败。

这里的“低 rank”是以 W-form rank $r_W$ 为主轴，而不是 hidden affine 的 $r_{\text{aff}}$。因此如果 $r_W=1,2,4$ 都是 affine 占优，就已经是很强的实际结论；这对应常见 LoRA 的超低 rank 预算，而不是一个人为放大的 hidden-rank 比较。主组中 **14/26** 在 $r_W=1,2,4$ 全部 affine 胜出，而且这 14 组也都在 $r_W=8$ 胜出。它们是：

Qwen2.5-0.5B；Qwen3-0.6B/1.7B/4B；Qwen3.5-0.8B/2B/4B；Llama-3.2-1B/3B；Gemma-2-9B/27B；Gemma-3-4B/12B/27B。

因此可以把结果分成三层写：

1. **超低 W-rank 强优势**：$r_W=1,2,4$ 甚至 8 都胜出，这是最有实践意义的 affine 优势。
2. **长窗口优势**：优势能延续到 $r_W=32/64$，主要是 Llama-3.2 与 Qwen3-4B。
3. **不适合 affine**：从 $r_W=1$ 起就不如 W，主要是 Llama-3.1、Qwen2.5 中大模型、Qwen3/3.5 大模型。

### Common-k spectrum: delta vs affine prediction

为了避免 rank 预算换算造成混淆，另用 [`../../../scripts/evaluate_pred_delta_common_spectrum.py`](../../../scripts/evaluate_pred_delta_common_spectrum.py) 对同一个 $k$ 直接比较两条累计谱：

$$
D=Y_c-X_c,\qquad P=X_c(A-I).
$$

输出见 [`../../tables/e/affine_pred_delta_common_spectrum.csv`](../../tables/e/affine_pred_delta_common_spectrum.csv)，聚合表见 [`../../tables/e/affine_pred_delta_common_spectrum_summary.csv`](../../tables/e/affine_pred_delta_common_spectrum_summary.csv)。三列含义：

- $C_D(k)$：`D` 的 top-$k$ 能量占 `D` 总能量比例。
- $C_P(k)$：`P` 的 top-$k$ 能量占 `P` 总能量比例。
- `P_abs_delta(k)`：$\|P\|_F^2/\|D\|_F^2\cdot C_P(k)$，也就是 affine prediction 的 top-$k$ 最终解释完整 delta 的比例。

主组 26 对的 median / mean：

| k | median $C_D(k)$ | median $C_P(k)$ | median `P_abs_delta(k)` | mean $C_D(k)$ | mean $C_P(k)$ | mean `P_abs_delta(k)` |
|---|-----------------|-----------------|--------------------------|---------------|---------------|------------------------|
| 1 | 0.021 | 0.105 | 0.017 | 0.079 | 0.168 | 0.060 |
| 2 | 0.036 | 0.161 | 0.025 | 0.096 | 0.214 | 0.071 |
| 4 | 0.051 | 0.227 | 0.036 | 0.117 | 0.268 | 0.085 |
| 8 | 0.067 | 0.320 | 0.048 | 0.142 | 0.325 | 0.099 |
| 16 | 0.089 | 0.421 | 0.060 | 0.172 | 0.385 | 0.113 |
| 32 | 0.132 | 0.520 | 0.072 | 0.210 | 0.446 | 0.126 |
| 64 | 0.200 | 0.600 | 0.080 | 0.259 | 0.510 | 0.138 |
| 128 | 0.304 | 0.691 | 0.086 | 0.327 | 0.580 | 0.150 |
| 256 | 0.421 | 0.764 | 0.092 | 0.422 | 0.663 | 0.163 |
| 512 | 0.574 | 0.850 | 0.101 | 0.554 | 0.761 | 0.176 |
| 1024 | 0.744 | 0.933 | 0.111 | 0.711 | 0.865 | 0.188 |
| 2048 | 0.918 | 0.989 | 0.119 | 0.867 | 0.954 | 0.198 |

这张表把矛盾点拆清楚了：$P$ 自己的谱确实比 $D$ 更集中，例如 median $C_P(16)=0.421$，高于 median $C_D(16)=0.089$；但由于 $\|P\|_F^2/\|D\|_F^2$ 的 median 只有约 0.12，$P$ 的 top-16 对完整 delta 的解释比例 median 只有 **0.060**。因此“affine prediction 谱更集中”和“affine top-k 对完整 delta 解释很多”不是同一句话。

补充：直接求 rank-$a$ affine LoRA 拟合完整变化量

$$
\min_{\operatorname{rank}(B)\le a}\|D-X_cB\|_F^2
$$

与上面的 $P=X_c(A-I)$ 截断是同一个 reduced-rank regression 解。令 $X_c=U\Sigma V^\top$，则 $X_cB=UC$ 且 $\operatorname{rank}(C)=\operatorname{rank}(B)$；目标分解为

$$
\|D-UC\|_F^2=\|U^\top D-C\|_F^2+\|(I-UU^\top)D\|_F^2.
$$

因此最优 rank-$a$ 解就是对 $U^\top D$ 做 top-$a$ SVD；对应的 fitted update $UU^\top D$ 正是 full affine prediction $P$。所以当前 `P_abs_delta(k)` 已经是在直接 rank-$k$ affine 拟合完整 $D$ 时能解释的 delta 能量比例，而不是一个更弱的“先拟合 full affine 再随便截断”的替代量。数值上用 $G_P=(A-I)^\top X_c^\top X_c(A-I)$ 与 direct reduced-rank regression Gram $(X_c^\top D)^\top (X_c^\top X_c)^{-1}(X_c^\top D)$ 校验，top-$k$ 能量差在 $10^{-8}$ 量级。

### Hybrid: affine LoRA + W-form residual LoRA

进一步测试两种 LoRA 形式叠加是否有参数优势。脚本 [`../../../scripts/evaluate_hybrid_affine_w_budget.py`](../../../scripts/evaluate_hybrid_affine_w_budget.py) 使用一个可解释的分解式 oracle：

$$
D=Y_c-X_c=P+R,\qquad P=X_c(A-I).
$$

hybrid adapter 用 rank-$a$ hidden affine LoRA 近似 $P$，再用 rank-$q$ W-form LoRA 近似 residual $R$，参数量为

$$
P_{\text{hybrid}}=h+2ha+q(n+h).
$$

这不是实际训练结果，而是谱分解下的构造性估计；同预算比较中允许退化为 pure W fallback，因此 hybrid 不会被强行写成比 pure W 差。输出见 [`../../tables/e/affine_hybrid_w_budget.csv`](../../tables/e/affine_hybrid_w_budget.csv)，summary 见 [`../../tables/e/affine_hybrid_w_budget_summary.csv`](../../tables/e/affine_hybrid_w_budget_summary.csv)，族内排序见 [`../../tables/e/affine_hybrid_w_budget_family_ordered.csv`](../../tables/e/affine_hybrid_w_budget_family_ordered.csv)。

同 pure W rank-$r_W$ 参数预算下，best hybrid 的解释量与 pure W 的比值如下：

| pure W rank | hybrid better | mean gain ratio | median gain ratio | match 参数比 mean | match 参数比 median | median 参数节省 |
|-------------|---------------|-----------------|-------------------|-------------------|---------------------|-----------------|
| 1 | 15/26 | 2.32 | 1.88 | 0.47 | 0.11 | 88.9% |
| 2 | 17/26 | 1.86 | 1.65 | 0.43 | 0.09 | 91.3% |
| 4 | 17/26 | 1.57 | 1.51 | 0.41 | 0.08 | 91.7% |
| 8 | 17/26 | 1.38 | 1.35 | 0.42 | 0.13 | 87.5% |
| 16 | 17/26 | 1.25 | 1.22 | 0.47 | 0.28 | 72.2% |
| 32 | 17/26 | 1.16 | 1.14 | 0.56 | 0.47 | 52.8% |
| 64 | 17/26 | 1.11 | 1.08 | 0.66 | 0.65 | 34.7% |

这里的 `match 参数比` 是“达到 pure W rank-$r_W$ 的解释量，hybrid 最少需要多少参数 / pure W 参数”。例如 $r_W=8$ 时，主组 median 只需 **12.5%** 的 pure W 参数即可达到同样解释量；但这主要由 17 个有效 hybrid 组贡献，剩余 9 组退化为 pure W，参数比为 1。

按模型族看：

- hybrid 明显有效：Qwen3 小中模型、Qwen3.5 0.8B/2B/4B、Llama-3.2、Gemma-2/3 大多数组、Qwen2.5-0.5B。
- hybrid 还能救回一部分 affine-only 不佳但 full affine gain 较高的模型：Qwen2.5-1.5B/3B。在 $r_W=8$，它们达到 pure W 解释量只需约 **33%** 参数。
- hybrid 基本无效、退回 pure W：Qwen2.5-7B/14B/32B/72B、Qwen3-8B/14B、Qwen3.5-9B、Llama-3.1-8B/70B。

因此，两种形式叠加的结果比单独 affine 更积极：即使 full affine 不能解释全部 delta，只要 $P$ 的低秩谱足够集中，hybrid 可以先用很便宜的 hidden affine rank 捕获 $P$ 的主方向，再用少量 W-rank 补 residual，从而在 17/26 组上取得明显参数优势。

小参数预算下的稳定性更适合作为主结论。若只看 $r_W\in\{1,2,4,8\}$，hybrid 在 **15/26** 组上四档全部同预算优于 pure W，并且这 15 组四档都能用更少参数 match pure W 的解释量。小预算稳定性表见 [`../../tables/e/affine_hybrid_small_budget_stability.csv`](../../tables/e/affine_hybrid_small_budget_stability.csv)。按族看：

- Qwen2.5：1/7 稳定小预算优势（仅 0.5B）。
- Qwen3：3/5 稳定小预算优势（0.6B/1.7B/4B）。
- Qwen3.5：3/4 稳定小预算优势（0.8B/2B/4B）。
- Llama-3.1：0/2。
- Llama-3.2：2/2。
- Gemma-2：3/3。
- Gemma-3：3/3。

### Unembedding / lm_head 侧补算

上面新做的 W-rank budget、common-k spectrum 和 hybrid sweep 最初只针对 input embedding `embed`。原始 Task6 full-vocab affine 表确实同时算了 `E_R2` 与 `U_R2`，但这些新预算表需要单独对 `lm_head` 再跑一次。脚本现已统一支持 `--matrix-kind embed|lm_head`，`lm_head` 侧读取 `standardized_sources.lm_head`；对 tied 模型这会落到共享权重，对 untied 模型则读取真实 unembedding。

U 侧主表：

- W-rank sweep：[`../../tables/u/unembed_w_rank_budget_sweep_all_main.csv`](../../tables/u/unembed_w_rank_budget_sweep_all_main.csv)
- W-rank 边界：[`../../tables/u/unembed_w_rank_budget_boundaries.csv`](../../tables/u/unembed_w_rank_budget_boundaries.csv)
- 族内排序：[`../../tables/u/unembed_w_rank_budget_family_ordered.csv`](../../tables/u/unembed_w_rank_budget_family_ordered.csv)
- common-k spectrum：[`../../tables/u/unembed_pred_delta_common_spectrum.csv`](../../tables/u/unembed_pred_delta_common_spectrum.csv)
- common-k summary：[`../../tables/u/unembed_pred_delta_common_spectrum_summary.csv`](../../tables/u/unembed_pred_delta_common_spectrum_summary.csv)
- hybrid sweep：[`../../tables/u/unembed_hybrid_w_budget.csv`](../../tables/u/unembed_hybrid_w_budget.csv)
- hybrid summary：[`../../tables/u/unembed_hybrid_w_budget_summary.csv`](../../tables/u/unembed_hybrid_w_budget_summary.csv)
- hybrid 小预算稳定性：[`../../tables/u/unembed_hybrid_small_budget_stability.csv`](../../tables/u/unembed_hybrid_small_budget_stability.csv)

U 侧比 E 侧更支持 affine/affine-hybrid 叙事。按 update scale 计，full affine prediction 的能量占比

$$
\frac{\|X_c(A-I)\|_F^2}{\|Y_c-X_c\|_F^2}
$$

在 E 侧 mean/median 为 **0.202 / 0.120**，在 U 侧为 **0.323 / 0.315**。也就是说，unembedding 的 Base→Instruct 更新中，全局 affine component 占完整 centered delta 的比例明显更高。

同 W-rank 参数预算下，affine-only 胜出数也明显增加：

| $r_W$ | E affine wins | U affine wins |
|------:|--------------:|--------------:|
| 1 | 15/26 | **24/26** |
| 2 | 15/26 | **24/26** |
| 4 | 14/26 | **22/26** |
| 8 | 14/26 | **20/26** |
| 16 | 13/26 | **18/26** |
| 32 | 8/26 | **13/26** |
| 64 | 3/26 | **6/26** |
| 128 | 1/26 | 1/26 |

U 侧 common-k 谱也更强。median 口径下：

| k | U median $C_D(k)$ | U median $C_P(k)$ | U median `P_abs_delta(k)` |
|---|------------------:|------------------:|---------------------------:|
| 1 | 0.061 | 0.151 | 0.044 |
| 8 | 0.161 | 0.383 | 0.119 |
| 16 | 0.208 | 0.490 | 0.155 |
| 64 | 0.306 | 0.692 | 0.216 |
| 512 | 0.633 | 0.917 | 0.273 |
| 2048 | 0.893 | 0.993 | 0.306 |

因此，U 侧不仅 `P` 自身比 `D` 更集中，而且 `P` 在完整 update 中的绝对占比也不小；这比 E 侧“谱更集中但绝对贡献被 residual 稀释”的情况更有利。

hybrid 结果更明显。与 pure W rank-$r_W$ 同参数预算相比：

| pure W rank | E hybrid better | U hybrid better | U median gain ratio | U median match 参数比 |
|-------------|----------------:|----------------:|--------------------:|----------------------:|
| 1 | 15/26 | **24/26** | 2.93 | 0.093 |
| 2 | 17/26 | **26/26** | 2.29 | 0.070 |
| 4 | 17/26 | **26/26** | 1.89 | 0.077 |
| 8 | 17/26 | **26/26** | 1.56 | 0.111 |
| 16 | 17/26 | **26/26** | 1.35 | 0.168 |
| 32 | 17/26 | **26/26** | 1.21 | 0.341 |
| 64 | 17/26 | **26/26** | 1.13 | 0.531 |

若只看 $r_W\in\{1,2,4,8\}$ 的小预算稳定性，U 侧 **24/26** 组四档都同预算优于 pure W，且四档都能用更少参数 match pure W 的解释量。两个例外是 Qwen2.5-1.5B 与 Qwen2.5-3B：它们在 $r_W=1$ 退回 pure W fallback；但从 $r_W=2$ 起 hybrid 也能超过 pure W。

按族内尺寸看：

- **Qwen2.5**：U 侧明显比 E 侧好。1.5B/3B 的 affine-only 仍差；7B/14B/32B/72B 有低 rank 优势但窗口较短，通常到 $r_W=3$–11 后稳定不如 W。
- **Qwen3**：U 侧全尺寸强于 E 侧，4B/8B/14B 都有长窗口优势，稳定落后边界在 $r_W\approx 76$–82。
- **Qwen3.5**：0.8B/2B/4B/9B 全部有低 rank 优势，9B 不再像 E 侧那样从 $r_W=1$ 起失败。
- **Llama**：Llama-3.1 在 U 侧被救回，8B 有低 rank 优势、70B 有长窗口优势；Llama-3.2 仍是强优势组。
- **Gemma-2/3**：U 侧整体仍有优势，但 Gemma-2-2B 窗口很短，$r_W=4$ 开始稳定不如 W。

因此，最终写法应区分 E 与 U：input embedding 侧的 affine 优势是“模型族依赖、低预算窗口内成立”；unembedding 侧的 affine/hybrid 优势更稳定，尤其 hybrid 在小到中等 W-rank 预算下几乎全组有效。

## 结论

这个检验给出三点结论：

1. **raw 与 centered 几乎同谱**：mean shift 不是 $E_\Delta$ 高 rank95 的主因；它平均只占 raw delta 能量 6.5%。
2. **pred 贴近 $A-I$，不贴近 centered/raw delta**：$X_c^\top X_c$ 对 $A-I$ 的重加权不是谱分散主因。
3. **residual 与 centered delta 同谱且占能量高**：residual / centered-delta energy 平均 0.798，中位 0.880；centered delta 的高 rank95 更像 residual 谱，而不是 pred 谱。

这修正了一个容易误写的推断：高 $R^2$ 不等于残差相对 $Y_c-X_c$ 很小。Task6 的 $R^2$ 分母是 $\|Y_c\|_F^2$，而微调差分 $\|Y_c-X_c\|_F^2$ 远小于 $\|Y_c\|_F^2$；因此 $R$ 可以对拟合质量很小，却对“差分谱”很大。

$$
\frac{\|R\|_F^2}{\|Y_c\|_F^2}\ll 1
\quad\not\Rightarrow\quad
\frac{\|R\|_F^2}{\|Y_c-X_c\|_F^2}\ll 1.
$$

## 写作判断

更稳的论文表述应是：

> In the high-$R^2$ BI regime, the affine parameter shift $A-I$ is empirically spectrally concentrated. The high $R^2$ does not imply this low rank; instead, it makes the low-rank $A-I$ meaningful as a global Base→Instruct affine update. The raw embedding difference $E_\Delta=Y-X$ is dominated, in spectral shape, by the residual of the centered affine fit when measured relative to the small update scale $\|Y_c-X_c\|_F$. Thus $A-I$ is not merely a reparameterization of $E_\Delta$; it parameterizes the globally coherent affine component, while $E_\Delta$ mixes that component with high-dimensional residual variation.

这可以支持「$A-I$ 比 naive delta 更好」的叙事，但证明逻辑不是“强仿射推出同谱低秩”，而是：

1. Task6 先实证观察到参数偏移 $A-I$ 谱集中；
2. 强仿射说明这个低秩 $A-I$ 是高解释率的全局线性主项，而不是任意拟合参数；
3. naive delta 混入了相对 update 尺度很大的高维残差；
4. 因此 $A-I$ 是更干净的全局更新表示，而 $E_\Delta$ 是主项 + 高维残差的混合。

## 输出

- 计算脚本：[`../../../scripts/compute_affine_pred_delta_svd.py`](../../../scripts/compute_affine_pred_delta_svd.py)
- W-rank 脚本：[`../../../scripts/evaluate_w_rank_budget.py`](../../../scripts/evaluate_w_rank_budget.py)
- common-k 谱脚本：[`../../../scripts/evaluate_pred_delta_common_spectrum.py`](../../../scripts/evaluate_pred_delta_common_spectrum.py)
- hybrid 脚本：[`../../../scripts/evaluate_hybrid_affine_w_budget.py`](../../../scripts/evaluate_hybrid_affine_w_budget.py)
- 结果表：[`../../tables/e/affine_task6_decomposition_svd.csv`](../../tables/e/affine_task6_decomposition_svd.csv)
