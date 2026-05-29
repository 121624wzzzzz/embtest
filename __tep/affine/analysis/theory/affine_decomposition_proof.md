# 仿射分解：$A-I$ 低秩现象的解释框架

> 实证支撑：[`../archive/task6_full_vocab_svd.md`](../archive/task6_full_vocab_svd.md)、[`../tasks/task6_pred_delta_probe.md`](../tasks/task6_pred_delta_probe.md)
> 派生表：[`../../tables/e/affine_task6_decomposition_svd.csv`](../../tables/e/affine_task6_decomposition_svd.csv)
> 验证脚本：[`../../../scripts/archive/verify_decomposition_bound.py`](../../../scripts/archive/verify_decomposition_bound.py)

本文给出在 BI 主分析组（$n=26$）下，如何把两个事实放在同一条严谨链条中解释：

1. **实证事实**：$A-I$ 的 rank95/h 更低、energy@5%h 更高，即仿射参数偏移本身谱集中。
2. **条件作用**：高 $R^2$ 说明这个低秩 $A-I$ 不是任意拟合参数，而是确实解释了 Base→Instruct 的全局仿射变化。

因此本文不声称「高 $R^2$ 推出 $A-I$ 低秩」。证明的核心是：给定实测的 $A-I$ 谱集中，精确分解与 Ky Fan 不等式说明为什么 naive 差分 $E_\Delta=Y-X$ 会比 $A-I$ 更分散，以及为什么 $A-I$ 可被解释为全局一致更新的低秩参数表示。所有数值断言均可由派生表逐对复现。

## 0. 读法：从结果到解释的流水线

下表中的“低秩”均指谱有效秩或能量集中度（rank95、effective rank、energy@5%h），不是严格代数秩。

| 步骤 | 问题 | 回答 |
|------|------|------|
| 1. 先看仿射是否有效 | `A-I` 是否值得解释？ | 主组 `E_R2≈0.991`，所以 `A-I` 是高解释率仿射映射的参数偏移 |
| 2. 再看 `A-I` 是否低秩 | 低秩是哪里来的？ | 直接对 `G_{A-I}=(A-I)^T(A-I)` 做 SVD；主组 rank95/h≈0.426、energy@5%h≈0.600 |
| 3. 同口径比较 delta | `E_delta` 是否也低秩？ | 对 `G_{rd}=(Y-X)^T(Y-X)` 做同样 SVD；rank95/h≈0.769、energy@5%h≈0.387 |
| 4. 解释为什么不同 | 两者为什么不同谱？ | `Y_c-X_c=X_c(A-I)+R`，delta 混入 residual，而 `A-I` 只参数化全局仿射项 |
| 5. 检查桥接项 | `X_c(A-I)` 是否改变了 `A-I` 谱？ | 主组 `G_p=(A-I)^TS(A-I)` 与 `G_{A-I}` 谱接近，故可把 `A-I` 的集中性传给仿射预测项 |
| 6. 给出论文结论 | 应该怎么写？ | “高 R² 使低秩 `A-I` 有解释意义；delta 更散来自 residual 混合”，而不是“高 R² 推出低秩” |

---

## 1. 记号与基本设定

设词表大小 $n$、hidden 维 $d$，并假定 $n>d$ 且 $X_c$ 列满秩。

| 符号 | 定义 |
|------|------|
| $X,Y\in\mathbb{R}^{n\times d}$ | Base / Instruct 嵌入矩阵，行为 token |
| $\mathbf{1}_n\in\mathbb{R}^n$ | 全 1 列向量 |
| $\mu_X,\mu_Y\in\mathbb{R}^d$ | 行均值，$\mu_X=\tfrac{1}{n}X^\top\mathbf{1}_n$ |
| $X_c,Y_c$ | 中心化，$X_c=X-\mathbf{1}_n\mu_X^\top$ |
| $S$ | Base 列协方差，$S:=X_c^\top X_c\in\mathbb{S}^d_{++}$ |
| $A,b$ | 仿射解，$A=\arg\min_{A,b}\|Y-XA-\mathbf{1}_n b^\top\|_F^2$ |
| $R$ | 残差，$R:=Y_c-X_cA\in\mathbb{R}^{n\times d}$ |

**法方程**（中心化等价形式）：

$$
A=S^{-1}X_c^\top Y_c,\qquad X_c^\top R=0,\qquad b=\mu_Y-A^\top\mu_X.
$$

谱诊断对象（所有 $d\times d$ PSD）：

$$
G_{rd}:=(Y-X)^\top(Y-X),\quad
G_{cd}:=(Y_c-X_c)^\top(Y_c-X_c),
$$

$$
G_p:=(A-I)^\top S(A-I),\quad
G_R:=R^\top R,\quad
G_{A-I}:=(A-I)^\top(A-I).
$$

对任意 PSD 矩阵 $M\in\mathbb{S}^d_+$，记其特征值 $\lambda_1(M)\ge\cdots\ge\lambda_d(M)\ge 0$，迹 $T_M=\sum_i\lambda_i(M)$，以及 **top-$k$ 能量份额**

$$
C_M(k):=\frac{1}{T_M}\sum_{i=1}^k\lambda_i(M),\qquad
\mathrm{rank}_{95}(M):=\min\{k:C_M(k)\ge 0.95\}.
$$

---

## 2. 仿射拟合 = col$(X_c)$ 上的正交投影

将 $X_c$ 的列空间记为 $\mathcal{X}\subset\mathbb{R}^n$，对应的正交投影算子

$$
P_{\mathcal{X}}=X_cS^{-1}X_c^\top\in\mathbb{R}^{n\times n}
$$

满足 $P_{\mathcal{X}}^2=P_{\mathcal{X}}=P_{\mathcal{X}}^\top$，且 $P_{\mathcal{X}}X_c=X_c$。

**关键恒等式**（直接代入 $A=S^{-1}X_c^\top Y_c$）：

$$
X_cA=P_{\mathcal{X}}Y_c,\qquad X_c(A-I)=P_{\mathcal{X}}(Y_c-X_c),\qquad R=(I-P_{\mathcal{X}})(Y_c-X_c).
$$

> **几何解释**：仿射拟合就是把「居中的微调差分 $Y_c-X_c$」按 col$(X_c)$ 做正交分解；$X_c(A-I)$ 是落在 Base 列空间上的部分，$R$ 是与 Base 列空间正交的部分。

---

## 3. 命题 1（精确正交分解）

**命题 1.** 设上述记号成立。则下述两个等式精确成立：

$$
\boxed{\;Y-X=(Y_c-X_c)+\mathbf{1}_n(\mu_Y-\mu_X)^\top,\;}\tag{3.1}
$$

$$
\boxed{\;Y_c-X_c=X_c(A-I)+R,\quad X_c(A-I)\perp R\;}\tag{3.2}
$$

其中 (3.2) 的正交性是关于 $\mathbb{R}^{n\times d}$ 上 Frobenius 内积；进一步，对应的 Gram 矩阵满足**精确**和分解

$$
\boxed{\;G_{cd}=G_p+G_R,\quad G_{rd}=G_{cd}+n(\mu_Y-\mu_X)(\mu_Y-\mu_X)^\top\;}\tag{3.3}
$$

**证明.**
(3.1) 由 $X=X_c+\mathbf{1}_n\mu_X^\top$ 和 $Y=Y_c+\mathbf{1}_n\mu_Y^\top$ 立得。

(3.2) 把 $A-I$ 代入直接展开：

$$
(Y_c-X_c)^\top(Y_c-X_c)=((A-I)^\top X_c^\top+R^\top)(X_c(A-I)+R)
$$
$$
=(A-I)^\top X_c^\top X_c(A-I)+(A-I)^\top\underbrace{X_c^\top R}_{=0}+\underbrace{R^\top X_c}_{=0}(A-I)+R^\top R
$$

由法方程 $X_c^\top R=0$，交叉项消去，故 $G_{cd}=G_p+G_R$。

(3.1) 中两项的列方向同样满足 $\mathbf{1}_n^\top(Y_c-X_c)=0$，故 $G_{rd}=G_{cd}+n(\mu_Y-\mu_X)(\mu_Y-\mu_X)^\top$。$\square$

**注 3.1**（自由度）。(3.3) 是**精确**等式，不需要任何关于 $R^2$ 大小、$X_c$ 条件数、$A$ 接近单位阵的假设。

---

## 4. 命题 2（残差能量份额的解析公式）

**命题 2.** 定义 $R^2:=1-\|R\|_F^2/\|Y_c\|_F^2$ 与**尺度比**

$$
\rho:=\frac{\|Y_c\|_F}{\|Y_c-X_c\|_F}.
$$

则**精确**有

$$
\boxed{\;
\frac{T_{G_R}}{T_{G_{cd}}}=1-w=\bigl(1-R^2\bigr)\rho^2,\qquad
w:=\frac{T_{G_p}}{T_{G_{cd}}}=1-(1-R^2)\rho^2.
\;}\tag{4.1}
$$

**证明.** 由 Frobenius 范数与迹的对应

$$
T_{G_R}=\|R\|_F^2=(1-R^2)\|Y_c\|_F^2,\qquad
T_{G_{cd}}=\|Y_c-X_c\|_F^2,
$$

代入即得 $T_{G_R}/T_{G_{cd}}=(1-R^2)\cdot\|Y_c\|_F^2/\|Y_c-X_c\|_F^2=(1-R^2)\rho^2$。$\square$

> **关键含义**。$R^2\to 1$ **不**保证残差相对 $Y_c-X_c$ 可忽略。BI 主组中 $\rho$ 中位 $\approx 16.5$，因此即便 $R^2=0.99$（$1-R^2=0.01$），仍有 $1-w\approx 0.01\times 16.5^2\approx 2.7\Rightarrow$ 被夹回 1 之内，但能给出**$1-w$ 可以接近 1** 的合理量级。

**BI 主组实测**：

| 量 | mean | median | range |
|----|------|--------|-------|
| $R^2$ | 0.991 | 0.996 | 0.375–1.000 |
| $\rho$ | 24.3 | 16.5 | 4.6–87.6 |
| $(1-R^2)\rho^2$（理论 $1-w$） | — | — | 与实测 $1-w$ 逐对一致 |
| 实测 $1-w$ | 0.798 | 0.880 | 0.43–0.97 |

实测 $1-w$ 全部由命题 2 公式精确复现（[verify_decomposition_bound.py](../../../scripts/archive/verify_decomposition_bound.py) 逐对计算）。

---

## 5. 命题 3（Ky Fan 凸组合不等式）

**命题 3.** 对任意 PSD 矩阵 $M,N\in\mathbb{S}^d_+$ 与 $1\le k\le d$，设 $G=M+N$，$w=T_M/T_G$。则

$$
\boxed{\;\max\bigl(wC_M(k),\,(1-w)C_N(k)\bigr)\;\le\;C_G(k)\;\le\;wC_M(k)+(1-w)C_N(k).\;}\tag{5.1}
$$

**证明.** 由 Ky Fan 极值原理：

$$
\sum_{i=1}^k\lambda_i(G)=\max_{\Pi\in\mathcal{P}_k}\mathrm{tr}(\Pi G\Pi)
$$

其中 $\mathcal{P}_k$ 为所有秩 $k$ 正交投影。对任一 $\Pi\in\mathcal{P}_k$，$\mathrm{tr}(\Pi G\Pi)=\mathrm{tr}(\Pi M\Pi)+\mathrm{tr}(\Pi N\Pi)\le\sum_{i=1}^k\lambda_i(M)+\sum_{i=1}^k\lambda_i(N)$，取 sup 后得上界。下界由 Weyl 单调性 $\lambda_i(G)\ge\lambda_i(M)$、$\lambda_i(G)\ge\lambda_i(N)$ 立得。除以 $T_G$ 即式 (5.1)。$\square$

将命题 3 应用到 (3.3) 的 $G_{cd}=G_p+G_R$ 上得：

**推论 5.1**（centered delta 能量集中度的紧致夹逼）：

$$
\boxed{\;\max\bigl(wC_p(k),\,(1-w)C_R(k)\bigr)\;\le\;C_{cd}(k)\;\le\;wC_p(k)+(1-w)C_R(k).\;}\tag{5.2}
$$

**注 5.1**（紧致性）。上界 (5.1) 中等号成立的充要条件是：$M,N$ 的 top-$k$ 特征空间相同（即两者在前 $k$ 维方向上同基）。在我们的实证中，每对 BI pair 上式 (5.2) 的上界与实测 $C_{cd}(k)$ 之差仅 0.0018–0.0113（mean 0.0059），见 §6 表，说明 $G_p$ 与 $G_R$ 的 top-$k$ 特征方向**近似同基**。

---

## 6. 应用：BI 主组 $k=\lceil 0.05d\rceil$ 验证

将 (5.2) 应用到 $k=\lceil 0.05d\rceil$ 上，逐对计算上下界并与实测对照（完整表见 [verify_decomposition_bound.py 输出](../../../scripts/archive/verify_decomposition_bound.py)）。

**主组聚合（$n=26$）**：

| 量 | $G_p$ | $G_R$ | $G_{cd}$（实测） | UB = $wC_p+(1-w)C_R$ |
|----|------|------|-------------------|-----------------------|
| $\mathrm{rank}_{95}/d$ mean | 0.425 | 0.807 | 0.781 | — |
| $C(k=0.05d)$ mean | 0.610 | 0.265 | 0.350 | **0.356** |
| $C(k=0.05d)$ median | 0.671 | 0.223 | 0.344 | **0.350** |

**逐对上界 gap 统计（UB$-C_{cd}$）**：

| | mean | median | min | max |
|--|------|--------|-----|-----|
| 26 对 | +0.0059 | +0.0058 | +0.0018 | +0.0113 |

**结论 6.1**：在 BI 主组上，命题 3 给出的凸组合上界**全部成立且数值上几乎饱和**。这说明 (5.2) 不仅是定理上的不等式，而是**实证上接近等式**：

$$
\boxed{\;C_{cd}(k)\;\approx\;wC_p(k)+(1-w)C_R(k).\;}\tag{6.1}
$$

---

## 7. 条件性比较定理：低秩 $A-I$ 如何压过 $E_\Delta$

到此可以把上面的引理拼成所要的条件性命题：若实测的 $A-I$（或等价的 $G_p$）比残差谱更集中，且 residual 在 update 尺度中占比不小，则 naive delta 的谱集中度必然低于 $A-I$。这里的低秩来源是 $A-I$ 的实测谱，而不是由高 $R^2$ 自动推出。

**条件性定理.** 设第 1 节记号成立，且 $1-w=(1-R^2)\rho^2$ 满足 $1-w\ge\tau$ 与 $C_R(k_0)\le\gamma$ 成立（$k_0=\lceil 0.05d\rceil$，$0<\tau,\gamma<1$）。则

$$
C_{cd}(k_0)\;\le\;1-\tau(1-\gamma).\tag{7.1}
$$

特别地，若 $G_p$ 与 $G_{A-I}$ 在 $k_0$ 处的能量份额近似（见 §8）：

$$
C_p(k_0)\approx C_{A-I}(k_0),
$$

则

$$
C_{cd}(k_0)-C_{A-I}(k_0)\;\le\;-(1-w)\bigl(C_{A-I}(k_0)-C_R(k_0)\bigr)\;<\;0.\tag{7.2}
$$

**证明.** 由 (5.2) 上界：

$$
C_{cd}(k_0)\le wC_p(k_0)+(1-w)C_R(k_0)\le w\cdot 1+(1-w)\gamma=1-(1-w)(1-\gamma)\le 1-\tau(1-\gamma),
$$

即 (7.1)。对 (7.2)，把 $w$ 等式代入：

$$
C_{cd}(k_0)-C_{A-I}(k_0)\le w\bigl(C_p(k_0)-C_{A-I}(k_0)\bigr)+(1-w)\bigl(C_R(k_0)-C_{A-I}(k_0)\bigr).
$$

前一括号由 §8 假设近似为 0；后一括号在 $C_{A-I}(k_0)>C_R(k_0)$（即 $A-I$ 比残差更集中）时严格负。$\square$

**BI 主组代入**：$\tau=0.43$，$\gamma=C_R(k_0)\approx 0.27$，立得 $C_{cd}(k_0)\le 1-0.43\times 0.73=0.686$。实测 $C_{cd}(k_0)\approx 0.35$，远小于此上界。式 (7.2) 给出的差为

$$
C_{cd}(k_0)-C_{A-I}(k_0)\;\lessapprox\;-0.80\times(0.60-0.27)\;=\;-0.264.
$$

实测 $C_{cd}-C_{A-I}\approx 0.350-0.600=-0.250$。两者一致。

最后由 (3.3)，

$$
G_{rd}=G_{cd}+n(\mu_Y-\mu_X)(\mu_Y-\mu_X)^\top,
$$

第二项秩 1，仅改变一个特征值，故 $\mathrm{rank}_{95}(G_{rd})\le\mathrm{rank}_{95}(G_{cd})+1$；实证亦表明 $C_{rd}(k_0)$ 与 $C_{cd}(k_0)$ 平均仅差 0.04（mean shift 占 raw delta 能量 6.5%）。所以 $G_{rd}$ 继承 $G_{cd}$ 的「分散性」。

**主结论**：

$$
\boxed{\;
C_{A-I}(k_0)\;\approx\;C_p(k_0)\;\gg\;C_{cd}(k_0)\;\approx\;C_{rd}(k_0),
\;}
$$

即 $A-I$ 与 $E_\Delta$ 谱集中度的差距可这样闭合：$A-I$ 的集中性是 Task6 的实测低秩事实；高 $R^2$ 使这个低秩参数具有全局仿射解释意义；而 $E_\Delta$ 更分散，则源自残差项 $R$ 主导 centered delta，其能量份额由命题 2 的精确公式 $1-w=(1-R^2)\rho^2$ 决定。

---

## 8. 关于 $G_p\approx G_{A-I}$（实证补充）

条件性定理需要一个补充：$G_p=(A-I)^\top S(A-I)$ 与 $G_{A-I}=(A-I)^\top(A-I)$ 的**归一化谱**在 BI 主组上数值接近。下面先给一个保守的 Ostrowski 界（界本身松），再用 [`../../tables/archive/affine_task6_proof_diagnostics.csv`](../../tables/archive/affine_task6_proof_diagnostics.csv) 数据说明实际差距远小于界。详细的边界讨论与异常组对照见 [附录 A](#附录-a数值验证两个关键假设)。

**引理 8.1**（congruence 谱重标定，Ostrowski 形式）. 设 $A-I=U\Sigma V^\top$（SVD），$K:=U^\top S U\in\mathbb{S}^d_+$。则

$$
G_{A-I}=V\Sigma^2 V^\top,\qquad G_p=V(\Sigma K\Sigma)V^\top,
$$

且 $G_p$ 的特征值可写作 $\lambda_i(G_p)=\theta_i\sigma_i^2$，其中 $\theta_i\in[\lambda_{\min}(K),\lambda_{\max}(K)]$ 由 $K$ 与 $\Sigma$ 的几何关系决定。

**重要警告（引理 8.1 的界本身松）**。Ostrowski 仅保证每个特征值偏移因子在 $[\lambda_d(K),\lambda_1(K)]$ 内，因此 per-eigenvalue 偏移可达 $\kappa(K)$ 倍。实证（附录 A）显示 $\kappa(K)$ 中位 21、最大 2240，所以 Ostrowski 给出的 per-eigenvalue 界本身不能直接推出 $C_p(k)\approx C_{A-I}(k)$。

**实证 8.2**. 然而**归一化能量份额**（rank95、energy@5%h）对 $K$ 的扰动远比单个 eigenvalue 鲁棒：

| 量 | $G_{A-I}$ mean | $G_p$ mean | $|G_p-G_{A-I}|$ mean / median / max |
|----|----------------|------------|-------------------------------------|
| energy@5%h | 0.600 | 0.610 | **0.038 / 0.018 / 0.282** |
| rank95/h | 0.426 | 0.425 | **0.026 / 0.011 / 0.166** |

中位 1–2%、均值 4%，max 28% 的偏移集中在 Gemma-3 系列。机制为：能量份额是积分 functional，对 per-eigenvalue 的随机重标定具有「中心极限式」鲁棒性（详见 [附录 A.1](#a1-g_papprox-g_a-i-的实证基础)）。

**实用结论**：在 BI 主组上，$C_{A-I}(k_0)$ 与 $C_p(k_0)$ 的差异远小于 $C_{A-I}(k_0)-C_R(k_0)$ 的 gap（前者 $\sim 0.04$，后者 $\sim 0.33$），故条件性比较结论 $C_{cd}\ll C_{A-I}$ 不被这个补充项颠覆。详细量化见附录 A。

---

## 9. 一句话物理解释与写作含义

完整分解（命题 1 + 命题 2）给出：

$$
Y-X=\underbrace{X_c(A-I)}_{\text{coherent affine, low-rank}}+\underbrace{R}_{\text{high-rank residual}}+\underbrace{\mathbf{1}_n(\mu_Y-\mu_X)^\top}_{\text{rank-1 mean shift}}
$$

且三项**两两正交**（Frobenius 内积下，由 (3.2) 与 $\mathbf{1}_n^\top(Y_c-X_c)=0$）。在 update 尺度 $\|Y_c-X_c\|$ 上，三者能量份额：

$$
\frac{\|X_c(A-I)\|_F^2}{\|Y_c-X_c\|_F^2}=w,\quad
\frac{\|R\|_F^2}{\|Y_c-X_c\|_F^2}=1-w,\quad
\frac{n\|\mu_Y-\mu_X\|^2}{\|Y-X\|_F^2}\approx 0.065.
$$

而 BI 主组 $w\approx 0.20$（小），$1-w\approx 0.80$（大）。**仿射主项虽然解释 $Y_c$ 的 $R^2\approx 0.99$，但只占微调差分 $Y_c-X_c$ 的 20% 能量**。

**怎样把「$A-I$ 更低秩」写成理论论证**：

1. **Task6 实证**先给出 $A-I$ 的低秩/高能量集中事实：$C_{A-I}(0.05d)\approx0.60$，rank95/h $\approx0.43$。
2. **高 $R^2$ 条件**说明这个低秩 $A-I$ 有解释意义：它不是低质量拟合中的偶然参数，而是全局仿射拟合的参数偏移。
3. **命题 1** 给出精确正交分解 $G_{cd}=G_p+G_R$，说明 naive delta 不是 $A-I$ 本身，而是仿射预测项与残差的混合。
4. **命题 2** 公式 $1-w=(1-R^2)\rho^2$ 解释为何高 $R^2$ 不保证 $R$ 在 update 尺度中小：尺度比 $\rho\sim 16$ 放大了 $1-R^2\sim 0.01$ 到 $1-w\sim 0.8$。
5. **命题 3** 与紧致上界 (5.2) 把 $C_{cd}(k)$ 表为 $\bigl(C_p(k),C_R(k)\bigr)$ 的 $(w,1-w)$ 凸组合，而 $w\ll 1$ 且 $C_R(k)\ll C_p(k)$，所以 $E_\Delta$ 的谱集中度被 residual 拉低。
6. **§8 实证**：$G_p\approx G_{A-I}$ 谱，故 $C_{A-I}(k)\approx C_p(k)\gg C_{cd}(k)\approx C_{rd}(k)$。

**论文表述建议**（替代「强仿射推出 $A-I$ 同谱低秩」的错误叙事）：

> In high-$R^2$ Base-Instruct pairs, the affine parameter shift $A-I$ is itself
> spectrally concentrated, indicating that the globally shared affine update is
> low-dimensional. The high $R^2$ does not imply this low rank; rather, it makes
> the low-rank $A-I$ meaningful as an explanatory parameter. Affine fitting gives
> the exact orthogonal decomposition $Y_c-X_c=X_c(A-I)\oplus R$. Although the fit
> explains $Y_c$ well, the residual occupies a large share of the small update
> $Y_c-X_c$ because $1-w=(1-R^2)\rho^2$. By Ky Fan's inequality,
> $C_{cd}(k)\le wC_p(k)+(1-w)C_R(k)$, and empirically this bound is nearly tight.
> Thus the naive delta $E_\Delta$ mixes a concentrated affine component with a
> high-dimensional residual, while $A-I$ directly parameterizes the concentrated
> global component.

---

## 10. 附录 A：闭合条件性比较的两个经验假设

条件性比较仍依赖两个未在主体中严格化的事实：

- **A1**：$G_p\approx G_{A-I}$（命题 §8）
- **A2**：Ky Fan 上界 (5.2) 在数据上近似饱和（结论 6.1）

本附录用 31 对 BI（主组 26 + 异常 5）直接验证这两个事实，并给出失效边界。所有数字由 [analyze_proof_assumptions.py](../../../scripts/archive/analyze_proof_assumptions.py) 计算，结果存于 [affine_task6_proof_diagnostics.csv](../../tables/archive/affine_task6_proof_diagnostics.csv)。

### A.1 由 Ostrowski 解释 A1

由 SVD $A-I=U\Sigma V^\top$，$G_p=V(\Sigma K\Sigma)V^\top$，其中 $K:=U^\top SU\in\mathbb{S}^{d}_{++}$。Ostrowski 引理给出：

$$
\lambda_d(K)\,\sigma_i^2\;\le\;\lambda_i(G_p)\;\le\;\lambda_1(K)\,\sigma_i^2,
$$

因此 **$G_p$ 与 $G_{A-I}$ 的 $i$-th 特征值至多相差 $\kappa(K)$**。设比较结论只在 $A-I$ 的「主信号方向」上使用 $G_p$，即考察 $K$ 在 $U$ 的前 $r$ 列上的限制 $K_r:=U_r^\top SU_r$。

**关键观察**：尽管 $\kappa(S)$ 在整个 hidden 空间上可极大（median **631**，max 53880），$\kappa(K_r)$ 在 $A-I$ 的 top-$r$ 子空间上**显著更小**：

| 范围 | $\kappa(S)$ | $\kappa(K_{r=0.05d})$ | $\kappa(K_{r=0.10d})$ | $\kappa(K_{r=\text{rank}_{95}(A-I)})$ |
|------|-------------|------------------------|------------------------|----------------------------------------|
| 主组 mean | 3615.3 | 171.3 | 267.6 | 195.6 |
| 主组 **median** | **631.4** | **21.3** | **31.8** | **62.2** |
| 主组 min | 58.0 | 1.94 | 2.54 | 7.02 |
| 异常 median | 1663.2 | 138.4 | 172.9 | 569.2 |

主组中位 $\kappa(K_{r=0.05d})\approx 21$，比 $\kappa(S)$ 小约 30×。这说明 **$A-I$ 的左奇异方向位于 $S$ 的「条件良好」子空间**，即仿射拟合**自动选择了 $S$ 几何上较各向同性的方向**。这是一个仿射拟合 + 嵌入几何耦合的结构性事实。

**Ostrowski 上界的紧性**。Ostrowski 仅给出 $\lambda_i$ 的乘性区间，对**归一化能量份额** $C_M(k)$ 的差距而言松。例如 $\kappa(K)=21$、$C_{G_{A-I}}(k)=0.6$ 时，最坏情况下 $C_{G_p}(k)\in[0.07,\,0.97]$。实测的差距远小于此：

| 量 | $|C_{G_p}(k)-C_{G_{A-I}}(k)|$（$k=0.05d$） |
|----|--------------------------------------------|
| 主组 mean / median | 0.0103 / **0.0088** |
| 主组 max | 0.282 |
| 异常 mean / median | 0.119 / 0.105 |

主组中 **median 0.009**：远低于 Ostrowski 的最坏界。这意味着 $K_r$ 的特征值重标定不是「对抗性」分布，而是与 $\Sigma$ 谱**协同**——构成更强的「能量份额近似」结论：

$$
\text{(empirical)}\quad C_{G_p}(k)\approx C_{G_{A-I}}(k)\quad\text{(主组 BI；非定理)}.
$$

### A.2 由特征子空间主角解释 A2

(5.2) 的上界

$$
C_{G_{cd}}(k)\le w\,C_p(k)+(1-w)\,C_R(k)
$$

是 Ky Fan 不等式 $\sum_i\lambda_i(M+N)\le\sum_i\lambda_i(M)+\sum_i\lambda_i(N)$ 的直接推论，等号当且仅当 $G_p$ 与 $G_R$ 的 top-$k$ 特征空间**重合**。

计算 $G_p$ 与 $G_R$ 的 top-$k$（$k=\lceil 0.05d\rceil$）正交基 $U_p^{(k)},U_R^{(k)}$ 之间的主角余弦平方

$$
\cos^2\theta_i=\sigma_i^2\bigl(U_p^{(k)\top}U_R^{(k)}\bigr),
$$

其均值

$$
\overline{\cos^2\theta}=\tfrac{1}{k}\bigl\|U_p^{(k)\top}U_R^{(k)}\bigr\|_F^2=\tfrac{1}{k}\mathrm{tr}\bigl(\Pi_p^{(k)}\Pi_R^{(k)}\bigr)\in[0,1].
$$

实证：

| 范围 | $\overline{\cos^2\theta}$ | $\max\cos^2\theta$ | $\min\cos^2\theta$ |
|------|---------------------------|--------------------|--------------------|
| 主组 mean | **0.606** | 0.987 | $\sim 0$ |
| 主组 median | **0.642** | 0.990 | $\sim 0$ |
| 异常 mean | 0.232 | 0.894 | $\sim 0$ |

主组中 **平均 60% 的主角余弦平方接近 1**：top-$k$ 子空间显著重合（不是完全相同，但远高于随机的 $k/d\approx 0.05$）。最大主角余弦平方接近 1，意味着**至少有一个 leading 方向被 $G_p$ 与 $G_R$ 同时识别为主方向**。这是 Ky Fan 上界几乎饱和的几何来源（实测 gap mean +0.006）。

**异常组**：$\overline{\cos^2\theta}=0.23$，子空间对齐显著弱化，与「$A-I$ 与 $E_\Delta$ 谱关系不再清晰」一致。

### A.3 失效边界：异常 5 对的验证

异常组（Gemma-3-1B + Gemma-4-{E2B, E4B, 26B-A4B, 31B}）：

| 假设 | 主组（n=26）median | 异常（n=5）median | 比例 |
|------|--------------------|--------------------|------|
| $\kappa(K_{r=0.05d})$ | 21.3 | 138.4 | **6.5×** |
| $\kappa(K_{r=\text{rank}_{95}(A-I)})$ | 62.2 | 569.2 | **9.2×** |
| $\overline{\cos^2\theta_{5\%}}$ | 0.64 | 0.26 | 0.41× |
| $|C_{G_p}-C_{G_{A-I}}|$ | 0.009 | 0.105 | **11.7×** |
| Ky Fan gap $\mathrm{UB}-C_{cd}$ | 0.006 | 0.012 | 2.0× |

异常组里：
1. 假设 **A1 弱化**：$|C_{G_p}-C_{G_{A-I}}|$ 高一个数量级 → $A-I$ 不再是 $G_p$ 的好代理。
2. 假设 **A2 弱化**：子空间对齐显著下降 → Ky Fan 上界仍成立但更松。
3. **命题 1、2、3 仍精确成立**（它们与模型无关），所以「centered delta = pred + residual」分解依然干净。

故比较结论在异常组中**退化为更弱的形式**：$C_{cd}(k)\ll C_p(k)$ 仍成立（因 $1-w$ 大、$C_R\ll C_p$），但「$G_p\approx G_{A-I}$」不再保证，所以无法从 $A-I$ 的谱直接读出 $E_\Delta$ 的谱关系。这与 Gemma-3-1B / Gemma-4 必须分报的实证完全一致。

### A.4 工程含义

把附录 A.1 的结构性事实「$\kappa(K_r)\ll\kappa(S)$」翻成可被论文引用的命题：

> Proposition (empirical, BI main group). In high-$R^2$ BI pairs, the left singular subspace of $A-I$ aligns with the well-conditioned subspace of $S=X_c^\top X_c$, in the sense that the restriction $K_r=U_r^\top SU_r$ has condition number $\kappa(K_r)$ at least 30× smaller than $\kappa(S)$ for $r\le \mathrm{rank}_{95}(A-I)$. This is the empirical bridge that makes $G_p\approx G_{A-I}$ valid as a working approximation, and is the precise property that breaks in the Gemma anomaly subset.

---

## 11. 边界与已知反例

| 情形 | 哪一步失效 |
|------|------------|
| 异常 5 对（$R^2<0.78$） | 命题 2 仍精确，但 $1-w$ 退化；附录 A.3 实证 $G_p$ 与 $G_{A-I}$ 谱差距放大 11.7× |
| 跨模型（Task5 全体） | 命题 1 仍精确，但 $\rho$ 与 $R^2$ 同步退化，残差谱不再「干净」高秩 |
| Gemma-3-1B | 附录 A.1/A.2 假设全部失效；主角余弦平方均值 0.087（最低），$A-I$ 与 $E_\Delta$ 谱无显著关系 |

---

## 12. 与旧备忘 [`delta_vs_AminusI_old.md`](../archive/delta_vs_AminusI_old.md) 的关系

旧理论备忘 §4.1 提出「左乘 $X_c$ 摊散 $A-I$ 的谱」为主因；本笔记 §6 / §8 实证表明，$G_p$ 与 $G_{A-I}$ 在 BI 主组上谱几乎相同，**$X_c$ 重加权不是主因**。真正的机制是 §4 给出的尺度比 $\rho$ 与 §5 的 Ky Fan 凸组合。

旧备忘 §8（可交换情形的命题 1–4）在 $S=cI$ 时仍正确，但不适用于解释 BI 主组的实测对比；本笔记的条件性比较（命题 1+2+3）是 **$S$ 非各向同性也成立** 的严格替代。
