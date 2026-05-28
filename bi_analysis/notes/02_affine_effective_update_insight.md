# Insight：Affine 为何能解释 U 侧 BI 更新

> 从「几何对齐现象」推进到「affine 为什么真的能解释并产生作用」。  
> 数据：`affine_effective_subspace.csv`（43 行，cuda:7）；**non-excluded 30 对**（untied 13 对为主论据）。  
> η=95% 有效子空间，除非另注。

---

## Affine 形式（全文统一记号）

Base→Instruct 词表侧权重记为 $X$（Base）、$Y$（Instruct），形状 $n\times d$（$n$=词表行数，$d$=hidden dim）。变化矩阵：

$$
\boxed{
\Delta W = Y - X
}
$$

我们检验 $\Delta W$ **是否可部分由 X 诱导的 affine 形式**（线性部分 + 样本维 bias）表示——即 $\Delta W$ 有多大比例的能量落在 $Z=[X,\mathbf{1}_n]$ 张成的（低维有效）子空间里，**并不要求**全局等于一个 affine 变换：

$$
\boxed{
\Delta W \approx X(A-I) + \mathbf{1}_n b^\top \quad\text{（部分能量；主指标 } R^2_{\mathrm{aff},K}\text{）}
}
$$

其中 $A\in\mathbb{R}^{d\times d}$，$b\in\mathbb{R}^d$，$\mathbf{1}_n\in\mathbb{R}^n$ 为全 1 列向量。等价地，令设计矩阵

$$
\boxed{
Z = [X,\,\mathbf{1}_n] \in \mathbb{R}^{n\times(d+1)},\qquad
\Theta = \begin{bmatrix} M \\ b^\top \end{bmatrix},\;\; M=A-I
}
$$

则

$$
\boxed{
\Delta W \approx Z\Theta = XM + \mathbf{1}_n b^\top
}
$$

**E/U 口径**：untied 模型 $X_E=\texttt{embed}$、$X_U=\texttt{lm\_head}$ 分列；tied 模型 $X_E=X_U$（共享矩阵，只算 E 侧一行）。

**centered 分解**（`__tep/affine` / `bi_analysis` 线 A）：$D=Y_c-X_c$，$P=X_c(A-I)$，$R=D-P$；主指标 $P/D=\|P\|^2/\|D\|^2$。  
**raw 变化量**（本文有效子空间 + 线 C）：$\Delta W=Y-X$；$\|\Delta W\|^2=\|D\|^2+n\|\mu_Y-\mu_X\|^2$（mean-shift 占 raw Δ 通常 2–8%）。  
**勿只读一侧**：untied U 在 raw 上 $R^2_{\mathrm{aff},K}\approx 33\%$、centered 上 $P/D\approx 32\%$ **同档**（mean-shift 占 raw Δ 中位 **0.01%**（E）/ **6.9%**（U））；E 两侧均 $\approx 5\%$。见 [`tables/README.md`](../tables/README.md) 线 A「Raw ΔW 对照」。

---

## 核心命题

$$
\boxed{
\Delta W \text{ 的总变化中，有相当一部分可以被 } [X,\mathbf{1}_n] \text{ 诱导的有效 affine 子空间解释；并且这些可解释部分集中在 } \Delta W \text{ 的主变化方向上。}
}
$$

不只是说「U 的主变化子空间 affine-aligned」，而是：

$$
\boxed{
\text{这种 affine-alignment 直接决定 affine update 能捕捉多少 } \Delta W \text{ 的有效变化能量。}
}
$$

---

## 1. Affine 的作用 = 把 ΔW 投影到 col(Z)

由上节 $\Delta W \approx Z\Theta$，令 $M=A-I$，则 $\Delta W \approx XM + \mathbf{1}_n b^\top = Z\Theta$。  
在 Frobenius 范数下，最优 affine 拟合即 $\Delta W$ 在 $\operatorname{col}(Z)$ 上的正交投影：

$$
\boxed{
\widehat{\Delta W}_{\mathrm{aff}} = P_Z \Delta W
}
$$

$$
\boxed{
P_Z = Z(Z^\top Z)^+ Z^\top
}
$$

**affine 能解释的比例**（全 $\operatorname{col}(Z)$ 投影，记 $R^2_{\mathrm{aff,full}}$）：

$$
\boxed{
R^2_{\mathrm{aff,full}}
=
R^2_{\mathrm{aff}}
=
\frac{\|\widehat{\Delta W}_{\mathrm{aff}}\|_F^2}{\|\Delta W\|_F^2}
=
\frac{\|P_Z \Delta W\|_F^2}{\|\Delta W\|_F^2}
}
$$

$$
\boxed{
\text{affine 的效果取决于 } \Delta W \text{ 有多少能量落在 } \operatorname{col}(Z)=\operatorname{col}([X,\mathbf{1}_n]) \text{ 里。}
}
$$

---

## 2. 为何要看低维有效子空间 Q_K

### 2.1 名义满秩 ≠ 有结构

深度学习里的权重矩阵 $X$、设计矩阵 $Z=[X,\mathbf{1}_n]$ **在代数上常常是满秩的**（例如 $\mathrm{rank}(Z)=\min(n,d+1)=d+1$）。  
但 **「满秩」本身几乎没有判别力**：

- 满秩只说明没有精确线性依赖，**不说明**能量均匀分布在所有方向上；
- 实际 SVD 谱几乎总是 **陡降**的：少数奇异值承载绝大部分 Frobenius 能量，其余方向多为噪声级小奇异值；
- 因此 meaningful 的问题不是「是否在 $\operatorname{col}(Z)$ 里」（一个 $(d+1)$ 维子空间相对 $n$ 维样本空间已经很大），而是：**$\Delta W$ 是否落在 $Z$ 的那些真正承载能量的低维有效方向上**——即 **有效秩（effective rank）**，而非名义秩。

### 2.2 若看全 col(Z)，结论可能平凡

对 $Z\in\mathbb{R}^{n\times(d+1)}$ 做 SVD：$Z=Q\Lambda R^\top$。左奇异向量 $q_j\in\mathbb{R}^n$ 是样本维上的方向。

若在 **样本维** $n$ 上 $\operatorname{rank}(Z)=n$（极端情况下 $d+1\ge n$ 且满秩），则 $\operatorname{col}(Z)=\mathbb{R}^n$，$P_Z=I$，从而 $R^2_{\mathrm{aff}}=1$——**任意** $\Delta W$ 都可被 $Z\Theta$ 精确表示，结论失去结构意义。

我们的设置里 $\mathrm{rank}(Z)\le d+1\ll n$（词表 $n\sim10^5$，$d\sim10^3$），故 $R^2_{\mathrm{aff,full}}$ 本身已非平凡（E 侧 ~5%，U 侧 ~36%）。  
但即便在 $\operatorname{col}(Z)$ 的 $(d+1)$ 维内，$Z$ 的奇异值谱仍高度不平衡：**不能把所有 $d+1$ 个方向一视同仁**，而应只保留累计能量达到阈值 $\eta$（90/95/99%）的前 $K$ 个方向：

$$
Q_K = [q_1,\dots,q_K], \qquad
\frac{\sum_{j=1}^{K}\lambda_j^2}{\sum_j \lambda_j^2} \ge \eta
$$

$Q_K$ 张成的才是 **$X$ 在样本维上的低维有效 affine 子空间**——即 $X$ 真正「活」在哪些 token 方向模式里，而不是 $Z$ 名义列空间的全部 $(d+1)$ 维。

### 2.3 有效子空间与指标

定义 **有效 affine 子空间**及其投影：

$$
\boxed{
\mathcal{S}_{\mathrm{aff},K} = \operatorname{span}(Q_K), \qquad P_K = Q_K Q_K^\top
}
$$

**有效 affine 可解释比例**（非平凡主指标）：

$$
\boxed{
R^2_{\mathrm{aff},K}
=
\frac{\|P_K \Delta W\|_F^2}{\|\Delta W\|_F^2}
=
\frac{\|Q_K Q_K^\top \Delta W\|_F^2}{\|\Delta W\|_F^2}
}
$$

问的是：**$\Delta W$ 的总能量中，有多少落在 $X$ 的有效（低维、高能量）affine 结构里**——既排除了样本维满秩的平凡性，也排除了 $Z$ 内部大量弱奇异方向的干扰。

与随机基线 $K/n$ 的倍率对照（§5–10）进一步检验：对齐是否显著强于「随机方向碰巧落进 $K$ 维子空间」。

---

## 3. 主变化方向如何决定 R²_aff,K

对 $\Delta W$ 做 SVD（样本维左奇异向量 $u_i\in\mathbb{R}^n$）：

$$
\boxed{
\Delta W = \sum_i \sigma_i u_i v_i^\top
}
$$

投影到有效 affine 子空间：

$$
\boxed{
P_K \Delta W = Q_K Q_K^\top \Delta W = \sum_i \sigma_i (Q_K Q_K^\top u_i) v_i^\top
}
$$

因此

$$
\boxed{
\|P_K \Delta W\|_F^2 = \sum_i \sigma_i^2 \|Q_K^\top u_i\|_2^2
}
$$

定义第 $i$ 个主变化方向的对齐度 $a_i(K)=\|Q_K^\top u_i\|_2^2$，则：

$$
\boxed{
R^2_{\mathrm{aff},K}
=
\frac{\sum_i \sigma_i^2 a_i(K)}{\sum_i \sigma_i^2}
}
$$

**理论核心**：

$$
\boxed{
\text{affine 的最终解释能力}
=
\sum_i
\underbrace{\sigma_i^2}_{\Delta W \text{ 第 } i \text{ 方向能量}}
\times
\underbrace{a_i(K)}_{\text{该方向对 } \mathcal{S}_{\mathrm{aff},K} \text{ 的对齐度}}
}
$$

→ $\Delta W$ 的主变化子空间是否落在 $\mathcal{S}_{\mathrm{aff},K}$ 里，**直接决定** affine 能解释多少总变化。

---

## 4. Rank-L 分解：r_L × A_{L,K}

前 $L$ 个奇异方向的谱能量占比：

$$
\boxed{
r_L = \frac{\sum_{i=1}^{L}\sigma_i^2}{\|\Delta W\|_F^2}
}
$$

前 $L$ 个主变化方向的**加权** affine 对齐度：

$$
\boxed{
A_{L,K} = \frac{\sum_{i=1}^{L}\sigma_i^2 a_i(K)}{\sum_{i=1}^{L}\sigma_i^2}
}
$$

前 $L$ 个方向对**总** $\Delta W$ 的 affine 解释贡献：

$$
\boxed{
R^{2,(L)}_{\mathrm{aff},K}
=
\frac{\sum_{i=1}^{L}\sigma_i^2 a_i(K)}{\|\Delta W\|_F^2}
=
r_L \cdot A_{L,K}
}
$$

$$
\boxed{
\text{只有 } r_L \text{ 高不够，只有 } A_{L,K} \text{ 高也不够；两者乘积才是对总变化的实际 affine 解释量。}
}
$$

随机基线：对单位随机方向 $u$，$\mathbb{E}\|Q_K^\top u\|_2^2 = K/n$；故 $A_{L,K}/(K/n) \gg 1$ 表示对齐**显著非随机**。

---

## 5. 实测：untied U vs E（中位数，η=95%）

**数据**：[`affine_effective_subspace.csv`](../tables/affine_effective_subspace.csv)（43 行）；untied **13 对** E/U 各 13 行；tied 17 对仅 E 行。Aff/LoRA 见 [`affine_lora_budget_summary.csv`](../tables/affine_lora_budget_summary.csv)（30 行）。

### 5.1 主变化子空间是否 affine-aligned？

| 指标 | E | U | 解读 |
|------|:---:|:---:|------|
| A₅,K | 0.048 | **0.829** | U 前 5 主方向 ~83% 落在 Q_K |
| A₁₀,K | 0.048 | **0.772** | 前 10 仍 ~77% |
| A₅/(K/n) | 1.6× | **30.3×** | U 对齐 ≈30 倍随机；E ≈随机 |
| a₁/(K/n) | 1.6× | **31.7×** | rank-1 方向倍率同档 |
| 13/13 U>E | — | ✓ | a₁、A₅、A₁₀、R²_aff,K 全部成立 |

**结论**：U 侧 ΔW 的**主变化子空间高度 affine-aligned**；E 侧主方向几乎不对齐（≈随机）。

### 5.2 谱集中度 r_L：我们关心的是「主成分」，不是全局极低秩

| 侧 | r₁ | r₅ | r₁₀ |
|---|:---:|:---:|:---:|
| **E** | 0.7% | 1.9% | 2.9% |
| **U** | 11.7% | **22.2%** | **26.1%** |

U 确实**不是** r₅≥70% 那种全局极低秩：前 5–10 个奇异方向只覆盖总能量的一部分（r₅≈22%，r₁₀≈26%）。  
**但这与我们的研究问题一致**——我们探究的不是「整个 $\Delta W$ 能否用 rank-5 精确重构」，而是：

> **BI 更新中，那些高效、可压缩的主变化成分，能否用 affine（$X(A-I)+\mathbf{1}b^\top$）这一结构化形式表达？**

在这个目标下：

- **不要求** R²_aff,K → 100%；主成分之外本就可能存在 token 级细粒度残差；
- **要求** 主变化方向与 $Q_K$ 显著对齐（A_{L,K} 高、倍率 ≫ K/n），且 affine 对总能量有 **substantial 占比**（R²_aff,K ~33% 已属不错）；
- r_L 中等意味着：affine 主要解释 **leading 子空间** 及其部分高阶延伸，而非全部 $\Delta W$——这正是「高效表达主成分」的合理预期。

（Qwen2.5 四模型 r₅≈49–52%，属更强的 near-rank-1 特例，不是 general 要求。）

### 5.3 分解链：R^(L) 与 R²_aff,K 各回答什么

| 量 | E | U | 公式 |
|----|:---:|:---:|------|
| r₅ | 1.9% | 22.2% | 前 5 方向能量占比 |
| A₅,K | 0.048 | 0.829 | 前 5 **内部**对齐 |
| **R^(5)_aff,K = r₅·A₅,K** | **0.07%** | **13.3%** | 前 5 对**总 ΔW** 的贡献 |
| r₁₀ | 2.9% | 26.1% | |
| **R^(10)_aff,K** | **0.10%** | **16.0%** | |
| **R²_aff,K（全谱）** | **3.6%** | **33.0%** | 含 u₆+ 的贡献 |

→ **R^(L)** 衡量「前 L 个主成分里 affine 解释了多少**总** ΔW」；**R²_aff,K** 衡量「**全部**方向加权后的 affine 解释占比」。  
U 上 A_{L,K} 很高 → 主成分内部几乎全是 affine 型；r_L 中等 → R^(5)≈13% 是预期内的；更高阶 u₆+ 亦部分对齐 → 全谱 R²_aff,K 升至 **33%**，**已满足「affine 高效表达主变化成分」的判断**。

### 5.4 全谱与 raw/centered 对照（30 对 untied 13，η=95% 中位）

数据来源：[`affine_effective_subspace.csv`](../tables/affine_effective_subspace.csv)、[`affine_lora_budget_summary.csv`](../tables/affine_lora_budget_summary.csv)。

| 指标 | E | U | 口径 |
|------|:---:|:---:|------|
| $R^2_{\mathrm{aff},K}$ | **3.6%** | **33.0%** | raw $\Delta W$，有效子空间 $Q_K$ |
| $R^2_{\mathrm{aff,full}}$ | **4.9%** | **36.0%** | raw $\Delta W$，全 $\operatorname{col}(Z)$ |
| $P/D$（centered 线性） | **4.9%** | **31.9%** | $D=Y_c-X_c$，$P=X_c(A-I)$ |
| mean-shift / raw $\Delta W$ | **0.01%** | **6.9%** | $n\|\mu_Y-\mu_X\|^2/\|\Delta W\|^2$ |

U 侧：**约 1/3 raw $\Delta W$ 能量**落在有效 affine 子空间；与 centered $P/D\approx 32\%$ **同档**（mean-shift 很小，去中心化几乎不改变结论）。  
**脚注**：historical main 9 untied alone 的 U $P/D$ 中位 **46.2%**；extended 4 并入 30 对后降至 **31.9%**（DeepSeek-V3 等 MoE 对拉低中位）。

残差占比（U）：

$$
\boxed{
\frac{\|\Delta W - \widehat{\Delta W}_{\mathrm{aff},K}\|_F^2}{\|\Delta W\|_F^2}
\approx 1 - R^2_{\mathrm{aff},K} \approx 67\%
}
$$

affine 未解释全部，但已捕捉 substantial 结构化成分。

### 5.5 全谱 vs 主方向：两者合论

$$
\boxed{
R^2_{\mathrm{aff},K} \approx 33\% \;\Rightarrow\; \Delta W_U \text{ 总变化中约三分之一可由有效 affine 子空间解释}
}
$$

$$
\boxed{
A_{5,K}\approx 0.83,\; A_{10,K}\approx 0.77 \;\Rightarrow\; \text{这 33\% 集中在主变化方向，而非随机散落}
}
$$

**两者合起来才是完整论证**：全谱 $R^2_{\mathrm{aff},K}$ 给出**总量**；$A_{L,K}$ 与 $R^{2,(L)}_{\mathrm{aff},K}=r_L A_{L,K}$ 说明**解释对象**是 leading 子空间。  
U 上 $A_{L,K}$ 很高但 $r_L$ 仅中等 → $R^{(5)}_{\mathrm{aff},K}\approx 13\%$、$R^{(10)}\approx 16\%$ **不会**接近 100%，而更高阶 $u_{6+}$ 亦部分对齐 → 全谱升至 **33%**。

### 5.6 Bias vs 线性：affine 解释不是纯 bias

| 侧 | $a_1(K)$ | $B_5$ | $P/D$（centered $X_c(A-I)$，无 bias） |
|---|:---:|:---:|:---:|
| **U** | 0.889 | 0.340 | **31.9%** |
| **E** | 0.047 | 0.006 | **4.9%** |

→ U 的 affine 解释**混有** $\mathbf{1}_n b^\top$ 型 bias（$B_5\approx 34\%$）与 **$X(A-I)$ 型线性**（$P/D\approx 32\%$）；不是单一 bias 故事。

---

## 6. 为何这构成「affine 有好效果」

若用有效子空间投影（或等价 affine 参数）近似：

$$
\boxed{
\widehat{\Delta W}_{\mathrm{aff},K} = P_K \Delta W = Q_K Q_K^\top \Delta W
}
$$

或用 full affine 参数 $\widehat{M},\widehat{b}$ 构造 $\widehat{\Delta W}_{\mathrm{aff}} = X\widehat{M}+\mathbf{1}_n\widehat{b}^\top$，则

$$
\boxed{
\frac{\|\widehat{\Delta W}_{\mathrm{aff},K}\|_F^2}{\|\Delta W\|_F^2} \approx R^2_{\mathrm{aff},K}
}
$$

$$
\boxed{
\frac{\|\Delta W - \widehat{\Delta W}_{\mathrm{aff},K}\|_F^2}{\|\Delta W\|_F^2} \approx 1 - R^2_{\mathrm{aff},K}
}
$$

对 untied U（中位）：$R^2_{\mathrm{aff},K}\approx 33\%$，残差 $\approx 67\%$。

**好效果**不是说解释了 100%，而是：

1. **解释量 substantial**：总变化约 **1/3** 落在有效 affine 子空间  
2. **解释对象重要**：$A_{5,K}\approx 0.83$、$A_{10,K}\approx 0.77$，且 $\gg K/n$（**~30×**）→ 捕捉的是**主变化方向**，非边缘噪声  
3. **E/U 机制分离**：E 侧 $R^2_{\mathrm{aff},K}\approx 4\%$、$A_{L,K}\approx$ 随机、$r_L$ 极散 → U 的 affine 效果是**结构性的**，非全体 BI 共性

$$
\boxed{
\text{affine 解释的是结构最强、最主要的变化方向，而不是随机残差。}
}
$$

---

## 7. 与 E 侧对照（一句话）

| 维度 | E（untied embed） | U（untied lm_head） |
|---|---|---|
| 主变化子空间 | 不存在（r₁≈0.7%） | 存在（r₅≈22%） |
| Affine 对齐 | $A_{5,K}\approx 5\%$，$\approx$ 随机 | $A_{5,K}\approx 83\%$，$\approx 30\times$ 随机 |
| Affine 解释总量 | $R^2_{\mathrm{aff},K}\approx 4\%$ | $R^2_{\mathrm{aff},K}\approx 33\%$ |
| Centered $P/D$ | $\approx 5\%$ | $\approx 32\%$ |
| 机制 | 高维 diffuse token 微调 | 结构化 affine update |

---

## 8. 最终主结论（理论 + 数据合一）

$$
\boxed{
U \text{ 侧 } \Delta W \text{ 并非全局极低秩，但其主变化子空间高度 affine-aligned；因此 affine 不能解释全部变化，却能解释总变化中约 } 33\% \text{，且主要作用于最结构化的主变化方向。}
}
$$

$$
\boxed{
U \text{ 侧存在显著的有效 affine update：全谱 } R^2_{\mathrm{aff},K}\approx 33\%\text{，前 }5\text{–}10\text{ 主方向 } A_{L,K}\approx 0.77\text{–}0.83\text{（}\sim 30\times K/n\text{）。}
}
$$

英文（可直接引用）：

$$
\boxed{
\text{Although } \Delta W_U \text{ is not extremely low-rank, a substantial fraction (}\sim 33\%\text{) of its total update energy is affine-explainable, and this component is concentrated on the leading update directions (}A_{5,K}\approx 0.83\text{, }A_{10,K}\approx 0.77\text{, }\sim 30\times\text{ random baseline).}}
}
$$

---

## 9. 推论：U 的 Δ 功率谱更集中 → affine 可解释占比更高

上述分解

$$
R^2_{\mathrm{aff},K}=\frac{\sum_i\sigma_i^2 a_i(K)}{\sum_i\sigma_i^2}
$$

说明：**谱越集中在「高 aᵢ 的方向」上，R²_aff,K 越高**。因此「U 谱更密 + U affine 可解释量更高」不是两个独立现象，而是**同一机制的两面**。

### 9.1 Raw ΔW 谱（样本维 SVD，untied 13 对中位）

| 侧 | r₁ | r₅ | r₁₀ | R²_aff,K |
|---|:---:|:---:|:---:|:---:|
| **E** | 0.7% | 1.9% | 2.9% | **3.6%** |
| **U** | 11.7% | **22.2%** | **26.1%** | **33.0%** |

U 的 ΔW **更集中**（r₅ 约为 E 的 12×）；同时 R²_aff,K 约为 E 的 **9×**。  
若 E 侧 aᵢ(K) 还 ≈ 随机（A₅,K≈5%），则 σᵢ² 分散 + aᵢ 低 → R² 必然极低——与实测一致。

### 9.2 Centered D/P 谱（线 A，untied 13 对中位；[`affine_lora_budget_summary.csv`](../tables/affine_lora_budget_summary.csv)）

| 侧 | $P$ rank95/$h$ ↓ | $D$ rank95/$h$ | $P$ 5% 能量/$h$ ↑ | $P/D$ |
|---|:---:|:---:|:---:|:---:|
| **untied E** | 0.578 | 0.856 | 0.257 | **0.049** |
| **untied U** | **0.363** | 0.813 | **0.738** | **0.319** |

- U 侧 **$P$ 比 $D$ 更集中**（$P$ rank95/$h$ 更低；$P$ 的 5% 能量/$h$ 更高）
- 同时 $P/D$ 高 → centered 线性 affine 占 $D$ 的比例大；与 raw $R^2_{\mathrm{aff},K}\approx 33\%$ 同向

### 9.3 理论链条（一句话）

$$
\boxed{
\underbrace{\text{U：}\sigma_i^2\text{ 更集中}}_{r_L\uparrow}
\times
\underbrace{a_i(K)\text{ 更高}}_{A_{L,K}\uparrow}
\;\Rightarrow\;
R^2_{\mathrm{aff},K}\uparrow
\quad\text{（E 侧两项均弱）}
}
$$

---

## 10. 实验验证：Aff/LoRA 同解释量优势（主实验）与 Hybrid（补充）

理论预测：若 $\Delta W$ 的主变化落在 $[X,\mathbf{1}_n]$ 有效子空间，则 **hidden-dim Aff/LoRA**（作用在 $A-I$）应以更低 rank 达到与 **W-form / Vocab LoRA**（作用在 $n\times d$ 词表矩阵）相同的解释量。

**参数量级**（Task6 口径，bias $h$ 另计）：

$$
P_W = h + r_W(n+h), \qquad P_{\mathrm{aff}} = h + 2h\,r_{\mathrm{aff}}
$$

同参数量下 W-rank 与 aff-rank 近似关系：

$$
r_{\mathrm{aff}} \approx r_W \cdot \frac{n+h}{2h} \approx r_W \cdot \frac{n}{2d}
$$

**Vocab LoRA rank=1 时，Aff/LoRA 参匹配秩常 ~8–61（untied U 中位 ~15）**——**1 对 $O(n/d)$**，见 [`tables/README.md`](../tables/README.md)。

### 10.1 n/d 与 W-rank=1 对应的 aff 参匹配秩（U 侧，**30 对 / untied 13**）

完整 13 行 untied 表见 [`tables/AFFINE_LORA_BUDGET.md`](../tables/AFFINE_LORA_BUDGET.md)。代表行：

| 模型 | tier | $n/d$ | $(n+d)/2d$ | aff rank @W1 |
|------|:---:|---:|---:|---:|
| Qwen3.5-35B-A3B (untied U) | extended | 121 | 61 | **61** |
| Qwen3.5-0.8B (tied) | main | 242 | 122 | **121** |
| Qwen3-30B-A3B (untied U) | extended | 74 | 37 | **37** |
| Qwen3.5-9B (untied U) | main | 61 | 31 | **30** |
| Qwen3-8B (untied U) | main | 37 | 19 | **19** |
| Llama-3.1-8B (untied U) | main | 31 | 16 | **16** |
| Qwen2.5-14B/32B (untied U) | main | 30 | 15 | **15** |
| DeepSeek-V3/V3.1 (untied U) | extended | 18 | 9 | **9** |
| Llama-3.1-70B (untied U) | main | 16 | 8 | **8** |

*注：`U_aff_rank_budget_r1` = W rank=1 参数量可买到的 aff rank（$\lfloor(n+h)/(2h)\rfloor$），与 $(n+d)/2d$ 一致。*

**untied U（13 对）**：aff rank @W1 中位 **15**（范围 8–61）；$(n+d)/2d$ 中位 **15.3**。  
→ Vocab LoRA **rank=1** 参预算下，Aff/LoRA hidden rank 通常 **~16 乃至 30–60+**（MoE 大词表）。

### 10.2 aff/W 胜率（centered gain，30 对）

| 分组 | n | aff/W @r1 | wins r1 | aff/W @r4 | wins r4 |
|------|---:|---:|---:|---:|---:|
| **untied U** | **13** | **2.18×** | **12/13** | **1.79×** | **10/13** |
| **untied E** | 13 | 0.34× | 1/13 | 0.33× | 0/13 |
| **tied（E=U）** | 17 | **3.00×** | **15/17** | **1.81×** | **14/17** |

→ 与 R²_aff,K、A_{L,K} 一致。**主实验**见 [`tables/affine_lora_budget_summary.csv`](../tables/affine_lora_budget_summary.csv)（30 行）。

**边界例外（不影响主结论）**：r1 唯一 U 侧 aff/W<1 的是 **DeepSeek-V3**（≈0.75）。原因复合、属交界案例而非反证：(1) U 侧 **P/D≈0.12** 低于 untied U 中位 ~0.32，仿射在 Δ 里绝对占比小；(2) $d=7168$、$n/d≈18$ → W rank=1 参匹配 aff rank 仅 **9**（Qwen3-8B 为 19），低秩 aff 对 $P$ 的压缩上限低；(3) 有效子空间与 E/U 分裂仍成立（U 的 R²_aff,K、P/D 仍高于 E）。**12/13 胜率 + 系统性 E/U 分裂**已足够；个别边界模型作脚注即可。DeepSeek-V3.1 同族 aff/W@r1≈1.15，亦支持「边界」而非「类型反转」的读法。

### 10.3 Hybrid（P + R 拆分）——对主 narrative **价值有限**

Hybrid = rank-$a$ hidden affine 近似 $P$ + rank-$q$ W-form 近似 $R=D-P$，在同参数预算下与 pure W 比解释量。

**为何不必作为主证据**：

1. **结论已被 aff vs W 覆盖**：U 侧 P/D 高、R²_aff,K 高 → 主变化已在 affine 子空间；hybrid 只是把 $D=P+R$ **事后拆分**，不增加「U affine-friendly」的新信息。
2. **对 U 侧增量小**：24/26 hybrid stable，但 affine-only 已 24/26 @ rW=1,2；高 P/D 的 Qwen2.5 untied U 上 aff/W @r1 仅 ~1.1×，hybrid 与 pure aff 差别不大。
3. **主要救场在 E / tied 例外**：E 侧 R 主导、hybrid 帮 E 低预算；Qwen2.5-1.5B/3B tied 在 rW=1 偏 W，hybrid 在 rW≥2 救回——均属**边界**，非 U affine 主结论核心。

**定位**：hybrid 可作为「$P$ 低秩 + $R$ 低秩」的构造性脚注；**主叙事依赖 §5 有效子空间 + §10.1–10.2 aff/W**，不必突出 hybrid。

### 10.4 tied ≈ untied U：同一套 affine-friendly 性质

tied 模型 E=U 共享矩阵，故 **tied 的 E 指标 = untied 的 U 指标**（同一 ΔW）。有效子空间验证（30 对 non-excluded，η=95%）：

| 分组 | r₅ | A₅,K | A₅/(K/n) | R²_aff,K |
|------|:---:|:---:|:---:|:---:|
| **tied E**（17） | 0.216 | 0.817 | **85×** | **0.322** |
| **untied U**（13） | 0.222 | 0.829 | **30.3×** | **0.330** |
| untied E（13） | 0.019 | 0.048 | 1.6× | 0.036 |

**tied E 与 untied U 在 r_L、A_{L,K}、R²_aff,K 上同档**（~32–33%），而 untied E 完全异类。  
倍率 tied 更高（85× vs 30×）因 K/n 更小，但 A₅,K 绝对值接近（0.82 vs 0.83）。

含义：**affine-friendly 不是「untied U 独有」，而是「与 lm_head 侧 / 共享权重矩阵相关的性质」**；untied 分裂后该性质**只留在 U**，tied 则在 E 上也能看到。

---

## 11. 建议主结论（可直接引用）

**中文（完整版）**：

> 我们考察 $\Delta W$ **有多大一部分**可由 $X$ 诱导的 affine 形式 $X(A-I)+\mathbf{1}_n b^\top$ 解释（非全局恒等）。由于完整 $\operatorname{span}([X,\mathbf{1}_n])$ 可能因满秩而平凡地表达任意变化，我们转而度量 $\Delta W$ 在 **[X, 1] 低维有效 affine 子空间** $Q_K$ 上的投影能量 $R^2_{\mathrm{aff},K}$。  
>  
> 在 untied U 侧，$\Delta W$ 的前 5–10 个主变化方向与 $Q_K$ 高度对齐（$A_{5,K}\approx 0.83$，$A_{10,K}\approx 0.77$，约为随机基线 $K/n$ 的 **30 倍**）；全谱 $R^2_{\mathrm{aff},K}\approx 33\%$，表明总变化能量中约三分之一可由有效 affine 子空间解释——**对「affine 能否高效表达主变化成分」这一问题，已是 substantial 且令人满意的占比**（并不要求全局 rank-5 重构）。  
>  
> $r_5\approx 22\%$ 说明我们捕捉的是 **leading 子空间** 而非全部 $\Delta W$；在此范围内 $r_L\cdot A_{L,K}$ 分解成立：$R^{(5)}_{\mathrm{aff},K}\approx 13\%$、$R^{(10)}\approx 16\%$，更高阶方向亦部分对齐，使全谱 $R^2_{\mathrm{aff},K}$ 高于 $R^{(10)}_{\mathrm{aff},K}$。  
>  
> 更重要的是，affine 可解释成分集中在主变化方向上；raw $\Delta W$ 上 $R^2_{\mathrm{aff},K}\approx 33\%$ 与 centered $P/D\approx 32\%$ 同档——affine 捕捉的是结构化、功能上重要的更新，而非随机残差。  
>  
> E 侧无此模式：谱极散、对齐 $\approx$ 随机、$R^2_{\mathrm{aff},K}\approx 4\%$。  
>  
> **Aff/LoRA 实验**（untied U **12/13** 优于 W @ r1；Vocab rank-1 参匹配 aff rank 中位 **15**）提供独立验证。

**中文（短版）**：

> U 侧（及 tied）：主变化子空间高度 affine-aligned → 全谱 ~33% 可解释；Aff/LoRA 12/13 untied U 优于 Vocab LoRA @ r1（rank-1 W 参匹配 aff rank ~15）。E 侧不支持。

**English（短版）**：

> *On the untied U side (and tied models, which mirror U-like properties), a substantial fraction (~33%) of the total update energy is affine-explainable; the leading directions are strongly affine-aligned (~30× random baseline); Aff/LoRA wins 12/13 at equal rank budget. The E side shows no comparable structure.*

---

## 12. 证据链地图

```text
理论：R²_aff,K = Σ σᵢ² aᵢ(K) / Σ σᵢ²
              ↓
U：r_L↑（谱更集中）× aᵢ↑（对齐）  →  R²_aff,K↑（~33%）
E：r_L↓ + aᵢ≈随机               →  R²_aff,K↓（~4%）
              ↓
rank-L：R^(L) = r_L · A_{L,K}
              ↓
实验：aff/LoRA @30对  untied U 12/13  |  Vocab r=1 → Aff r~8–61
（hybrid 补充，非主证据）
              ↓
tied E ≈ untied U（R²_k~32%, A₅~0.82）  ≠  untied E
              ↓
centered P/D（U ~32%）← 与 raw R² 同档印证
```

---

## 13. 变化量与可解释性解耦（untied 13 对）

一个可能的质疑：**U 的 affine 可解释占比更高，是否只是因为 U 的 BI 变化量（‖ΔW‖）更大？**

**不是。** 变化量大小与 affine 结构强度是**解耦**的：

| 对比 | 成立对数 | 含义 |
|------|:---:|------|
| U 的 `1−R²_identity` > E（U 变化更大） | **10/13** | 多数对 U 确实改动更多 |
| U 的 P/D > E | **13/13** | **全部** untied 对 U 仿射占比更高 |
| U 的 R²_aff,K > E | **13/13** | **全部** untied 对 U 有效子空间解释更高 |

关键反例（**U 变化更小或相近，但 affine 结构仍显著更强**）：

| 模型 | U/E 变化量比 | U/E P/D | U/E R²_aff,K |
|------|:---:|:---:|:---:|
| **DeepSeek-V3** | **0.4×**（U 更小） | 2.0× | 2.7× |
| DeepSeek-V3.1 | 0.8× | 1.8× | 2.6× |
| **Qwen3-30B-A3B** | **1.0×**（几乎相同） | **7.6×** | **9.7×** |
| Qwen3.5-35B-A3B | 1.1× | 10.3× | 13.3× |

→ DeepSeek：U 侧 ΔW 绝对量更小，R²_aff,K 仍 ~13% vs E ~5%；Qwen3-30B：E/U 变化量几乎相同，U 的 P/D 与 R²_aff,K 仍高一个数量级。

因此 U 侧更高的 affine 可解释性来自 **ΔW 的结构**（谱更集中、主方向对齐 Q_K），**不能**用「U 改动更大」解释。这也与 §3 的分解 R²_aff,K = Σσᵢ²aᵢ/Σσᵢ² 一致：指标对 ‖ΔW‖ 归一化，且 aᵢ(K) 在 U 侧远高于随机。

---

## 相关文件

- **数据总目录**：[`../tables/README.md`](../tables/README.md)
- 有效子空间：`../tables/affine_effective_subspace.csv`、`../tables/AFFINE_EFFECTIVE_SUBSPACE.md`
- Aff/LoRA（30 对）：`../tables/affine_lora_budget_summary.csv`、`../tables/AFFINE_LORA_BUDGET.md`
- 1−R²：`../tables/DELTA_W_R2_ONE_MINUS.md`
- D/P/R 谱（30 对）：`../tables/dpr_common_spectrum_{e,u}.csv`
- 原始 sweep 明细 / 脚本：`__tep/affine/`
