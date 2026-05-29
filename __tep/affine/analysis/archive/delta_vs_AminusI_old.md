# 理论备忘：$E_\Delta$ 与 $A-I$ 的谱关系

> 与实证对照：[`laws_old.md`](laws_old.md) §4、[`task6_full_vocab_svd.md`](task6_full_vocab_svd.md)。  
> 指标定义：[`ijcai_clean/docs/methods_and_metrics.md`](../../../../ijcai_clean/docs/methods_and_metrics.md) Task6 SVD 节。  
> 实现：$E_\Delta$ 的 Gram 为 $(Y-X)^\top(Y-X)$（流式行累加）；$A-I$ 的 Gram 为 $(A-I)^\top(A-I)$（`full_vocab_affine.py`）。

---

## 1. 符号设定

| 符号 | 含义 |
|------|------|
| $n$ | 词表大小（行数） |
| $d=h$ | hidden 维 |
| $X,Y\in\mathbb{R}^{n\times d}$ | Base / Instruct 嵌入矩阵（行=token） |
| $\mathbf{1}\in\mathbb{R}^n$ | 全 1 列向量 |
| $\mu_X,\mu_Y\in\mathbb{R}^d$ | 行均值：$\mu_X=\frac{1}{n}X^\top\mathbf{1}$ |
| $X_c,Y_c$ | 中心化：$X_c=X-\mathbf{1}\mu_X^\top$，$Y_c=Y-\mathbf{1}\mu_Y^\top$ |

**仿射拟合（与代码一致）**：最小化 $\|Y_c-X_c A\|_F^2$，得 $A\in\mathbb{R}^{d\times d}$，$b=\mu_Y-\mu_X A$，且

$$
Y \approx \hat Y = XA + \mathbf{1}b^\top,\qquad R := Y-\hat Y.
$$

记 **naive 差分** $E_\Delta := Y-X$（未中心化），**偏移矩阵** $\Delta A := A-I$。

**谱诊断对象（均为 $d\times d$ Gram，与 $n$ 无关的归一化口径）**：

$$
G_\Delta := E_\Delta^\top E_\Delta = (Y-X)^\top(Y-X),\qquad
G_{\Delta A} := (\Delta A)^\top(\Delta A).
$$

对 $G$ 的特征值 $\lambda_1\ge\cdots\ge\lambda_d\ge 0$，定义平方奇异值能量份额 $p_i=\lambda_i/\sum_j\lambda_j$，以及（代码口径）

$$
\mathrm{rank}_{95}(G):=\min\Big\{k:\ \sum_{i=1}^k p_i \ge 0.95\Big\},\qquad
\mathrm{eff\_rank}(G):=\exp\Big(-\sum_{i:p_i>0} p_i\log p_i\Big).
$$

报告中的 $\mathrm{rank}_{95}/h$、$\mathrm{eff\_rank}/h$ 即上述量除以 $d$。$\mathrm{energy@5\%}h$ 为前 $\lceil 0.05d\rceil$ 个主成分的累计能量份额。

---

## 2. 恒等分解与 $R^2\to 1$ 条件

### 2.1 未中心化恒等式

由 $\hat Y = XA+\mathbf{1}b^\top$ 直接展开：

$$
\boxed{Y-X = X(A-I) + \mathbf{1}b^\top + R.}
$$

这是**精确恒等式**，不依赖 $R^2$。$R$ 为行残差矩阵；$R^2\to 1$ 时 $\|R\|_F$ 相对 $\|Y_c\|_F$ 可忽略。

### 2.2 中心化形式

利用 $b=\mu_Y-\mu_X A$ 与 $Y_c\approx X_c A$：

$$
Y_c - X_c \approx X_c(A-I) + R_c,\qquad R_c := Y_c - X_c A = R\ \text{（与未中心化残差相同矩阵）}.
$$

代回 $Y-X = Y_c-X_c + \mathbf{1}(\mu_Y-\mu_X)^\top$：

$$
\boxed{Y-X \approx X_c(A-I) + \mathbf{1}(\mu_Y-\mu_X)^\top + R.}
$$

**$R^2\to 1$ 的可忽略条件（相对谱分析）**：记 $\varepsilon_R := \|R\|_F/\|Y-X\|_F$，$\varepsilon_{\mathrm{aff}}$ 为仿射解释不了的行方向能量占比。当 $\varepsilon_R\ll 1$（主组 median $E\_R^2\approx 0.997$ 时典型 $\varepsilon_R\lesssim 0.05$）时，在 $G_\Delta$ 的谱上可把 $R$ 视作小扰动；**Gemma 异常组** $E\_R^2\sim 0.37\text{–}0.78$ 时此项不可省（§6）。

### 2.3 与 $G_\Delta$、$G_{\Delta A}$ 的关系

在 $R\approx 0$ 下：

$$
G_\Delta \approx \underbrace{(X_c\Delta A)^\top(X_c\Delta A)}_{\text{左乘 }X_c\text{ 的二次型}}
+ \underbrace{n\,(\mu_Y-\mu_X)(\mu_Y-\mu_X)^\top}_{\text{秩 }1\text{ 平移}}
+ \text{交叉项}.
$$

**关键**：即使 $R^2$ 很高，$G_\Delta$ 也**不等于** $G_{\Delta A}$；中间必经过 $X_c$ 与均值平移。**不存在**“$R^2\to 1 \Rightarrow G_\Delta\approx G_{\Delta A}$”的定理。

---

## 3. 代数秩不等式 vs $\mathrm{rank}_{95}$

### 3.1 矩阵秩

对任意 $M,N$：$\mathrm{rank}(M+N)\le \mathrm{rank}(M)+\mathrm{rank}(N)$。应用于 §2.1：

$$
\mathrm{rank}(Y-X) \le \mathrm{rank}(X\Delta A) + \mathrm{rank}(\mathbf{1}b^\top) + \mathrm{rank}(R)
\le \mathrm{rank}(\Delta A) + 1 + \mathrm{rank}(R),
$$

其中第二步在 $X$ **列满秩**（$\mathrm{rank}(X)=d$，$n\gg d$ 时合理）下用 $\mathrm{rank}(X\Delta A)\le \mathrm{rank}(\Delta A)$。

**推论（严格秩）**：高 $R^2$ 时 $\mathrm{rank}(R)$ 小，故 $\mathrm{rank}(E_\Delta)$ **理论上界**由 $\mathrm{rank}(\Delta A)+1$ 控制——即 $E_\Delta$ 的代数秩**不应显著高于** $A-I$ 的秩（再加 1 维平移）。

### 3.2 $\mathrm{rank}_{95}$ 与代数秩的区别

| 概念 | 定义 | 对 $E_\Delta$ vs $A-I$ 的含义 |
|------|------|--------------------------------|
| 代数秩 | 非零奇异值个数 | $E_\Delta$ 上界 $\le \mathrm{rank}(\Delta A)+1+\mathrm{rank}(R)$ |
| $\mathrm{rank}_{95}$ | 累计能量达 95% 的最小 $k$ | 衡量**能量是否集中在少数主方向**；满秩矩阵也可 $\mathrm{rank}_{95}\ll d$ |
| $\mathrm{eff\_rank}$ | 能量分布熵的指数 | 对“谱平坦度”敏感，介于 $\mathrm{rank}_{95}$ 与满秩之间 |
| $\mathrm{energy@5\%}h$ | 固定前 $5\%$ 维的累计能量 | 不依赖阈值 95%，可与 $\mathrm{rank}_{95}$ **反向**（见 Gemma-4 反例） |

**主组实证**（$n=26$）：$\mathrm{rank}_{95}(G_\Delta)/h \approx 0.77 > \mathrm{rank}_{95}(G_{\Delta A})/h \approx 0.43$，同时 $\mathrm{energy@5\%}h(G_{\Delta A}) > \mathrm{energy@5\%}h(G_\Delta)$。这与 §3.1 的代数秩上界**不矛盾**：$E_\Delta$ 可以**满秩但能量分散**，$A-I$ 可以**低 $\mathrm{rank}_{95}$ 且低维高能**。

---

## 4. 为何实证上 $E_\Delta$ 的 $\mathrm{rank}_{95}/h$ 更高

以下在 $R^2\gtrsim 0.99$（主组）下讨论；各条可并存。

### 4.1 左乘 $X_c$：对 $\Delta A$ 奇异向量的混合（主因）

忽略 $b,R$ 时 $E_\Delta \approx X_c\Delta A$。设 $X_c^\top X_c = U_X D_X U_X^\top$，$\Delta A = U_{\Delta} D_{\Delta} V_{\Delta}^\top$（SVD）。则

$$
G_\Delta \approx V_{\Delta} D_{\Delta}\, \underbrace{(U_{\Delta}^\top U_X) D_X (U_X^\top U_{\Delta})}_{=:K}\, D_{\Delta} V_{\Delta}^\top.
$$

一般 $K\neq I$：$X_c$ 的列协方差**不会**与 $\Delta A$ 的主方向对齐。效果：

1. **旋转**：$G_{\Delta A}$ 的特征向量是 $\Delta A$ 的右奇异方向；$G_\Delta$ 的特征向量是 $V_{\Delta}$ 经 $K$ 混合后的方向，主能量可摊到更多特征值 → $\mathrm{rank}_{95}(G_\Delta)$ **上升**。
2. **缩放**：若 $D_X$ 特征值跨度大，$K$ 对 $D_\Delta$ 做加权，使少数 $\sigma_i(\Delta A)$ 在 $G_\Delta$ 中相对变弱 → 达到 95% 累计能量需要更多方向。

**与 $G_{\Delta A}$ 的对比**：$G_{\Delta A}=(\Delta A)^\top(\Delta A)$ **不经过** $X_c$；它是“参数空间偏移”的纯二次型，与 token 几何无关。故 **$G_\Delta$ 与 $G_{\Delta A}$ 谱不等价是结构性的**，非拟合误差 artifact。

### 4.2 $n\times d$ 行对象 vs $d\times d$ 参数对象（口径澄清）

| | $E_\Delta=Y-X$ | $\Delta A=A-I$ |
|--|----------------|----------------|
| 形状 | $n\times d$ | $d\times d$ |
| 用于谱的 Gram | $G_\Delta\in\mathbb{R}^{d\times d}$ | $G_{\Delta A}\in\mathbb{R}^{d\times d}$ |
| 对 $n$ 的依赖 | $G_\Delta=\sum_{i=1}^n (y_i-x_i)(y_i-x_i)^\top$；**总能量归一化后与 $n$ 缩放无关** | 与 $n$ 无关 |

二者报告指标均在 **$d$ 维特征空间**上比较；差异来自 **Gram 定义不同**，而非“一个按词表维归一、一个按 hidden 归一”。

### 4.3 平移 $b$ 与残差 $R$ 对谱的影响

**平移**：$\mathbf{1}b^\top$ 在 $G_\Delta$ 中贡献 $nb b^\top$（秩 1）。若 $\|b\|$ 与 $\|X_c\Delta A\|_F$ 可比：

- 可能在单一特征方向注入大量能量 → $\mathrm{energy@5\%}h$ **升高**但 $\mathrm{rank}_{95}$ **降低**（一维即达 95%）；
- 或与 $X_c\Delta A$ 项**不正交**时产生交叉项，使能量更分散 → $\mathrm{rank}_{95}$ **升高**。

主组 $\|b\|$ 相对较小（$E\_R^2$ 高），平移项常次要；**不能**从符号上单独断定符号，需看 $b$ 与 $X_c\Delta A$ 主方向的夹角（**模型依赖**，启发式）。

**残差 $R$**：$G_\Delta = G_{X_c\Delta A} + G_{\mathbf{1}b^\top} + G_R + \text{cross}$。$R^2$ 高时 $G_R$ 小，但 $R$ 可在**与主成分正交**的方向上有噪声 → 抬高 $\mathrm{rank}_{95}(G_\Delta)$ 而几乎不改变 $\mathrm{rank}_{95}(G_{\Delta A})$。

### 4.4 小扰动 $A\approx I+\varepsilon M$（一阶，启发式）

设 $A=I+\varepsilon M$，$\varepsilon\ll 1$。则 $\Delta A=\varepsilon M$，

$$
Y-X \approx \varepsilon X_c M + \mathbf{1}(\mu_Y-\mu_X)^\top + R.
$$

- $G_{\Delta A}=\varepsilon^2 M^\top M$：谱随 $\varepsilon^2$ 整体缩放，$\mathrm{rank}_{95}/h$、$\mathrm{eff\_rank}/h$ **与 $\varepsilon$ 无关**（能量份额不变）。
- $G_\Delta \approx \varepsilon^2 (X_c M)^\top(X_c M) + n(\mu_Y-\mu_X)(\cdots)^\top$：第一项是 $M^\top (X_c^\top X_c) M$ 的 congruence；第二项与 $\varepsilon$ **解耦**。

故：**$E_\Delta$ 的“总变化强度”随 IT 增大，但 $A-I$ 的谱形状可由 $M$ 刻画**；族内 $\|\Delta A\|_F$ 与 $\|Y-X\|_F$ 相关时，出现 $\mathrm{eff\_rank}$ 的线性律（§5）并不令人意外，但斜率 $\approx 0.5$ **不能**从 $\varepsilon$ 一阶单独推出。

---

## 5. 可检验推论与族内 $\mathrm{eff\_rank}/h$ 线性律

### 5.1 由分解式得到的偏序（$R^2$ 高时）

| 推论 | 内容 | 可检验性 |
|------|------|----------|
| C1 | $\mathrm{rank}_{95}(G_\Delta) \ge \mathrm{rank}_{95}(G_{\Delta A})$ **不保证**；代数秩上界允许 $\mathrm{rank}_{95}(G_\Delta) > \mathrm{rank}_{95}(G_{\Delta A})$ | 主组 26/26 对 $G_\Delta$ 的 $\mathrm{rank}_{95}$ 更高 |
| C2 | $\mathrm{energy@5\%}h(G_{\Delta A}) > \mathrm{energy@5\%}h(G_\Delta)$ 与 C1 **可同时成立** | 主组中位 0.628 vs 0.397 |
| C3 | 增大 $\|X_c^\top X_c\|$ 条件数或 $b$ 与 $X_cM$ 不对齐 → $\mathrm{rank}_{95}(G_\Delta)$ 上升 | 需 per-pair 算 $\kappa(X_c)$、$\cos(b, \text{top vec})$ |
| C4 | $R^2$ 下降 → $\|R\|$ 项破坏 C1–C2 | 异常组 rank95 双高但 energy@5%h 双低 |

### 5.2 族内 $\mathrm{eff\_rank}/h$ 回归斜率 $\approx 0.5$（启发式）

主组回归（[`laws_old.md`](laws_old.md) §4）：

$$
\frac{\mathrm{eff\_rank}(G_{\Delta A})}{h} \approx -0.012 + 0.515\,\frac{\mathrm{eff\_rank}(G_\Delta)}{h},\quad R^2\approx 0.55;
$$

族内 Qwen2.5 / Qwen3.5 / Llama 的 $R^2>0.9$。

**分解式给出的机制（非严格推导）**：

1. 同族模型 $X_c$ 的 $G_X=X_c^\top X_c$ 谱形相近 → $K$ 在族内近似固定，$G_\Delta$ 与 $G_{\Delta A}$ 的特征值份额呈**单调**但非恒等映射。
2. $\mathrm{eff\_rank}=\exp(H(p))$ 对 $p$ 的卷积型变换在“$p_\Delta$ 与 $p_{\Delta A}$ 均较集中”时，数值上常出现 $\mathrm{eff\_rank}(G_\Delta) \sim 2\,\mathrm{eff\_rank}(G_{\Delta A})$ 量级（$\mathrm{eff\_rank}/h$ 均值 0.418 vs 0.203，比 $\approx 2.06$）。
3. **斜率 0.5 是族内经验律**，不能从 $G_\Delta \approx (A-I)^\top G_X (A-I)$ 在一般 $G_X$ 下解析推出；跨亚族混合（16 Qwen 合并 $R^2=0.43$）会稀释，符合“$G_X$ 非共享”。

**可追加检验**：在同一 pair 上算 $\mathrm{eff\_rank}\big((A-I)^\top G_X (A-I)\big)$ 与 $\mathrm{eff\_rank}(G_\Delta)$ 的相关；若 $r\gg$ 对 raw $G_{\Delta A}$ 的相关，则支持“左乘 $X_c$”为主因。

---

## 6. 局限与失效情形

| 情形 | 后果 |
|------|------|
| $E\_R^2$ 低（Gemma-3-1B $\approx 0.375$，Gemma-4 $\approx 0.67\text{–}0.78$） | §2 中 $R$ 不可忽略；$A$ 仍最小化 MSE 但**不再**代表 $Y-X$ 的主要方向 → $G_{\Delta A}$ 与 $G_\Delta$ 脱钩 |
| Gemma-4：$\mathrm{rank}_{95}(G_\Delta)/h$ 高、$\mathrm{energy@5\%}h(G_{\Delta A})$ 仍高 | 说明单看 $\mathrm{rank}_{95}$ 不足；与 §3.2 一致 |
| $X$ 列秩 $<d$ | §3.1 上界变松；实际 embedding 罕见 |
| 跨模型 / 跨 hidden | $G_X$、$\Delta A$ 机制不可比；勿外推 §5 回归 |

---

## 7. 小结（写作用）

1. **恒等式** $Y-X = X(A-I)+\mathbf{1}b^\top+R$ 始终成立；$R^2\to 1$ 只保证 $R$ 小，**不**保证 $E_\Delta$ 与 $A-I$ 同谱。
2. **严格秩**：$\mathrm{rank}(E_\Delta)\lesssim \mathrm{rank}(A-I)+1+\mathrm{rank}(R)$；与 **$\mathrm{rank}_{95}$**（能量集中度）是不同问题。
3. **实证 $\mathrm{rank}_{95}(E_\Delta)>\mathrm{rank}_{95}(A-I)$** 与 **$A-I$ 的 $\mathrm{energy@5\%}h$ 更高** 可由 $G_\Delta\approx (A-I)^\top G_X(A-I)$ 的混合 + 平移/残差项定性解释；主组高 $R^2$ 下这是**预期结构**，不是矛盾。
4. **族内 $\mathrm{eff\_rank}$ 斜率 $\approx 0.5$** 宜表述为“同族共享 $G_X$ 时的经验耦合”，需标注启发式；异常组应剔除。

---

## 8. rank95 / energy 集中度：可严格化的命题

本节在 **$R\approx 0$、忽略平移 $b$** 下，主项取
$G_\Delta^{(0)}=(A-I)^\top G_X(A-I)$，$G_{A-I}=(A-I)^\top(A-I)$，$G_X=X_c^\top X_c\succ 0$。
记 $G$ 的特征值 $\lambda_1\ge\cdots\ge\lambda_d$，能量份额 $p_i=\lambda_i/\sum_j\lambda_j$，
$\mathrm{rank}_{95}(G)=\min\{k:\sum_{i=1}^k p_i\ge 0.95\}$，
$\mathrm{energy@}m(G)=\sum_{i=1}^m p_i$（Task6 中 $m=\lceil 0.05d\rceil$ 即 `energy@5%h`）。

### 8.1 基线：各向同性 $G_X$ 时谱完全相同

**命题 1.** 若 $G_X=cI$（$c>0$），则 $G_\Delta^{(0)}=c\,G_{A-I}$，故
$p^{(\Delta)}=p^{(A-I)}$，从而
$\mathrm{rank}_{95}(G_\Delta^{(0)})=\mathrm{rank}_{95}(G_{A-I})$，
$\mathrm{energy@}m(G_\Delta^{(0)})=\mathrm{energy@}m(G_{A-I})$ 对任意 $m$ 成立。

*证.* 直接代入。$\square$

**含义：** 实证里 $G_\Delta$ 比 $G_{A-I}$「更散」，**必须**来自 $G_X$ 的各向异性（$\kappa(G_X)>1$），而非仿射拟合误差。

---

### 8.2 可交换（commuting）情形：充要结构的解析模型

设 $A-I$ 与 $G_X$ **可交换**，存在正交基 $V$ 使
$A-I=V\Sigma V^\top$，$G_X=V\Gamma V^\top$，
$\Sigma=\mathrm{diag}(\sigma_1,\ldots,\sigma_d)$，$\Gamma=\mathrm{diag}(\gamma_1,\ldots,\gamma_d)$，$\gamma_i>0$。
则
$G_{A-I}=V\,\mathrm{diag}(\sigma_i^2)\,V^\top$，
$G_\Delta^{(0)}=V\,\mathrm{diag}(\sigma_i^2\gamma_i)\,V^\top$。

定义未归一化权重 $w_i=\sigma_i^2\ge 0$，$u_i=w_i\gamma_i$。则
$p_i^{(A-I)}=w_i/W$，$p_i^{(\Delta)}=u_i/U$，其中 $W=\sum w_i$，$U=\sum u_i$。

**命题 2（头能量稀释 $\Rightarrow$ rank95 上升，充分条件）.**
设 $w_1>0$，分布 $p^{(A-I)}$ 非退化，且 $\gamma_i$ **严格递增**。
若存在 $k_\star<d$ 使 $\sum_{i=1}^{k_\star} p_i^{(A-I)}\ge 0.95$（即 $\mathrm{rank}_{95}(G_{A-I})\le k_\star$），
且 $\gamma_{k_\star+1}/\gamma_1 > W/U$（尾向 embedding 方差相对放大足够），则
$\sum_{i=1}^{k_\star} p_i^{(\Delta)} < 0.95$，故
$\mathrm{rank}_{95}(G_\Delta^{(0)}) > \mathrm{rank}_{95}(G_{A-I})$。

*证.* 
$\sum_{i=1}^{k_\star} p_i^{(\Delta)}
=\frac{\sum_{i\le k_\star} w_i\gamma_i}{U}
\le \frac{\gamma_{k_\star}\sum_{i\le k_\star} w_i}{U}
=\frac{\gamma_{k_\star}}{\gamma_1}\cdot\frac{\gamma_1 W}{U}\cdot\sum_{i\le k_\star}p_i^{(A-I)}$。
当 $\gamma_{k_\star}/\gamma_1 > W/U$ 且 $\sum_{i\le k_\star}p_i^{(A-I)}\ge 0.95$ 时，
若 $\gamma_1 W/U < \gamma_{k_\star}/\gamma_1$ 的适当形式… 

更直接的充分条件：因 $\gamma_i$ 递增，
$\sum_{i=1}^{k_\star} w_i\gamma_i \le \gamma_{k_\star}\sum_{i=1}^{k_\star} w_i$，
而 $U=\sum w_i\gamma_i \ge \gamma_1 W$。
故
$\sum_{i=1}^{k_\star} p_i^{(\Delta)}
\le \frac{\gamma_{k_\star}}{\gamma_1}\cdot\frac{\sum_{i\le k_\star}w_i}{W}
=\frac{\gamma_{k_\star}}{\gamma_1}\cdot C_{A-I}(k_\star)$，
其中 $C_{A-I}(k)$ 为 $G_{A-I}$ 的累计能量。
又 $U/W = \sum w_i\gamma_i / \sum w_i \ge \gamma_1$（$\gamma$ 递增），
故 $\sum_{i\le k_\star}p_i^{(\Delta)} \le \frac{\gamma_{k_\star}}{\gamma_1}\cdot\frac{W}{U}\cdot C_{A-I}(k_\star)$。
当 $\gamma_{k_\star}/\gamma_1 \cdot (W/U) < 1$ 且 $C_{A-I}(k_\star)\ge 0.95$ 不足以使乘积 $\ge 0.95$，
即 $(W/U)\cdot C_{A-I}(k_\star) < 0.95\cdot \gamma_1/\gamma_{k_\star}$ 时 rank95 上升。
因 $\gamma_{k_\star}/\gamma_1>1$，右端 $<0.95$，故存在参数区间使 rank95 严格上升。$\square$

*注.* 条件可直观读作：**$A-I$ 的能量集中在与 $G_X$ 较小特征值对齐的方向**（头 $\sigma_i$ 大但 $\gamma_i$ 小），
经 $w_i\mapsto w_i\gamma_i$ 重权后，达到 95% 需更多方向——与 §4.1「左乘 $G_X$ 摊散能量」一致。

**命题 3（固定前缀 energy@$m$ 下降，充分条件）.**
在命题 2 同设定下，令 $m<d$。若 $\gamma_1\le\cdots\le\gamma_m<\gamma_{m+1}$ 且
$\sum_{i=1}^m p_i^{(A-I)}$ 较大（$A-I$ 头集中），则
$\sum_{i=1}^m p_i^{(\Delta)}
=\frac{\sum_{i\le m}w_i\gamma_i}{U}
\le \frac{\gamma_m}{\gamma_1}\cdot\frac{W}{U}\cdot\sum_{i=1}^m p_i^{(A-I)}$。
当 $\gamma_m W/(\gamma_1 U) < 1$ 时，
$\mathrm{energy@}m(G_\Delta^{(0)}) < \mathrm{energy@}m(G_{A-I})$。

*证.* 同命题 2 的加权估计。$\square$

**与主组实证对照：** Task6 主组（$n=26$）median `energy@5%h`：
$G_{A-I}$ **0.628** vs $G_\Delta$ **0.397**——与命题 3 方向一致（$m=\lceil0.05d\rceil$）。

---

### 8.3 熵 / effective rank 的严格单调性（可交换 + 严格递增 $\gamma$）

**命题 4.** 在 §8.2 设定下，若 $w_i>0$ 至少两项非零且 $\gamma_i$ **严格递增**，则 Shannon 熵
$H(p^{(\Delta)}) > H(p^{(A-I)})$，从而 $\mathrm{eff\_rank}(G_\Delta^{(0)})>\mathrm{eff\_rank}(G_{A-I})$。

*证.* 映射 $T_\gamma: (w_i)\mapsto (w_i\gamma_i)$ 在固定 $\gamma$ 递增时对正权重向量是 **严格 Schur-凹** 的重标定
（尾索引权重相对放大）。$p^{(A-I)}$ 非退化时，$p^{(\Delta)}=T_\gamma(w)/U$ 严格 majorization-平化
（除非 $\gamma_1=\cdots=\gamma_d$，退化为命题 1）。
熵在严格平化下严格增大。$\square$

*注.* 这给出 **eff_rank 斜率 $\approx 0.5$** 的解析空间：在共享 $G_X$（固定 $\gamma$）的族内，
$\Delta A$ 扰动强度 $\|w\|$ 变化时，$H(p^{(\Delta)})$ 与 $H(p^{(A-I)})$ 单调耦合；
斜率数值依赖 $\gamma$ 谱与 $w$ 谱，**一般不能**从命题 4 推出常数 $1/2$，需族内拟合（§5.2）。

---

### 8.4 非可交换：扰动界（把「严格」降一级）

一般地 $G_\Delta^{(0)}=(A-I)^\top G_X(A-I)$ 与 $G_{A-I}$ 不可交换。
设 $G_X=G_X^{(0)}+E$，$\Delta A=\Delta A^{(0)}+F$，标准 Davis–Kahan / Weyl 给出：
$G_\Delta^{(0)}$ 的特征值相对可交换基准 $\{\sigma_i^2\gamma_i\}$ 的偏移为
$O(\|[\Delta A,G_X]\|/(\gap))$（$\gap$ 为 $G_{A-I}$ 特征值间隙）。

**推论 8.4（稳定性）.** 若 $R^2$ 高且 $\|[\Delta A,G_X]\|$ 相对 $\|G_{A-I}\|_F$ 小
（主组 $A\approx I$、$G_X$ 谱隙适中），则 rank95 / energy@$m$ 的**序关系**与 §8.2 可交换模型**同号**，
偏差由 $\|[\Delta A,G_X]\|$ 控制。

*可检验.* 对每个 BI pair 估计
$\kappa(G_X)=\lambda_{\max}(G_X)/\lambda_{\min}(G_X)$ 与
$\Delta_{\mathrm{rank95}}=\mathrm{rank}_{95}(G_\Delta)-\mathrm{rank}_{95}(G_{A-I})$ 的 Spearman 相关；
若 $r>0$，支持「$G_X$ 越各向异性，rank95 差距越大」。

---

### 8.5 仍不能无条件证明的（写作边界）

| 陈述 | 状态 |
|------|------|
| 主组 **26/26** 对 $\mathrm{rank}_{95}(G_\Delta)>\mathrm{rank}_{95}(G_{A-I})$ | **经验**（Task6 CSV）；非定理 |
| $\mathrm{energy@5\%}h(G_{A-I})>\mathrm{energy@5\%}h(G_\Delta)$ 恒成立 | **条件命题 3**；Gemma 异常可反例 |
| eff_rank 斜率 $=0.5$ | **族内经验** + 命题 4 解释方向；非普适常数 |

**推荐论文写法：**
> Under high $R^2$, $G_\Delta\approx (A-I)^\top G_X(A-I)$. When $G_X$ is anisotropic
> ($\kappa(G_X)>1$) and $\Delta A$ concentrates on directions misaligned with top $G_X$ modes,
> Propositions 2–3 predict **more dispersed** $G_\Delta$ spectrum and **more concentrated** $G_{A-I}$ spectrum
> in rank95 / energy@$m$ sense—consistent with $n=26$ main-group pairs.

---

## 参考文献（本仓库）

- `ijcai_clean/src/ijcai_clean/experiments/full_vocab_affine.py` — `gram_of_row_delta`, `svd_energy_from_gram`, `full_affine_stream`
- [`laws_old.md`](laws_old.md) §4 — 主组/异常组数值
- [`task6_full_vocab_svd.md`](task6_full_vocab_svd.md) — Task6 汇总表
