# 有效 Affine 子空间验证（raw ΔW = Y−X）

**Affine 形式**：$\Delta W = Y-X$；**部分**可落在 $X(A-I)+\mathbf{1}_n b^\top=Z\Theta$（$Z=[X,\mathbf{1}_n]$）张成的子空间内，主指标 $R^2_{\mathrm{aff},K}$。详见 [`notes/02_affine_effective_update_insight.md`](../notes/02_affine_effective_update_insight.md)。

口径：`Z = [X, 1_n]`，`ΔW = Y − X`（**未中心化**）。  
脚本：`__tep/scripts/evaluate_affine_effective_subspace.py`（cuda:7）。  
数据：`affine_effective_subspace.csv`（43 行 = 17 tied 仅 E + **13 untied×2**；**30 对** non-excluded）。  
数据目录：[`README.md`](README.md)。

---

## 指标分层

| 层次 | 指标 | 问什么 |
|------|------|--------|
| **谱集中度** | r₁, r₅, r₁₀ | ΔW 前 L 个奇异方向占总能量的比例 |
| **rank-1 对齐** | a₁(K), a₁/(K/n) | 第一主方向落在 Q_K 的比例 / 相对随机 |
| **rank-L 对齐** | A_{L,K}, A_{L,K}/(K/n) | 前 L 个主方向**加权**落在 Q_K 的比例 |
| **L 阶解释贡献** | R²_aff,K^(L) = r_L · A_{L,K} | 前 L 个方向对**总 ΔW** 的 affine 解释量 |
| **全谱解释** | R²_aff,K | 全部 ΔW 落在 Q_K 的比例 |
| **bias 型** | a_bias, B_L | rank-1 / 前 L 方向里有多少是 1_n 型 |

核心关系：**R²_aff,K^(L) = r_L · A_{L,K}**

---

## untied 13 对：中位数（η=95%）

### 谱集中度 r_L

| 侧 | r₁ | r₅ | r₁₀ |
|:---:|:---:|:---:|:---:|
| **E** | 0.007 | 0.019 | 0.029 |
| **U** | 0.117 | **0.222** | **0.261** |

U 侧前 5 方向承载 ~22% 总能量——**不是全局极低秩，但符合「探究主成分高效表达」的目标**；在此 leading 子空间内 A_{L,K} 高、全谱 R²_aff,K ~33% 已是 substantial 占比。

### rank-L 对齐 A_{L,K}（及倍率）

| 侧 | a₁(K) | A₅,K | A₁₀,K | a₁/(K/n) | A₅/(K/n) | A₁₀/(K/n) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E** | 0.047 | 0.048 | 0.048 | 1.6× | 1.6× | 1.6× |
| **U** | 0.889 | **0.829** | **0.772** | 31.7× | **30.3×** | **29.5×** |

**13/13**：U > E（a₁、A₅、A₁₀、R²_aff,K、R²_aff,K^(5)、R²_aff,K^(10) 全部成立）。

rank-1 → rank-5 → rank-10：**倍率稳定在 ~30×**，A_{L,K} 绝对值仍高（77–89%）→ **不是 rank-1 偶然对齐，前 10 个主方向整体都显著偏向 affine 有效子空间**。

### 解释能量分解

| 侧 | R²_aff,K^(5) | R²_aff,K^(10) | R²_aff,K（全谱） |
|:---:|:---:|:---:|:---:|
| **E** | 0.001 | 0.001 | 0.036 |
| **U** | **0.133** | **0.161** | **0.330** |

验算：r₅·A₅,K ≈ 0.22 × 0.83 ≈ **0.13** = R²_aff,K^(5) ✓

- 前 5 方向贡献 ~13%、前 10 贡献 ~16%——受 r_L 限制
- 全谱 R²_aff,K ~33% **> R²_aff,K^(10)** → u₆ 及更高阶方向也有可观对齐

### bias 型占比

| 侧 | a_bias | B₅ | B₁₀ |
|:---:|:---:|:---:|:---:|
| **E** | 0.006 | 0.006 | 0.004 |
| **U** | 0.477 | **0.340** | **0.282** |

U 前 5 方向里 ~34% 能量是 bias 型；a_bias 高但 A_{L,K} 仍高 → **affine 解释混有 bias + X(A−I) 型**。

---

## tied 17 对（仅 E）

| 指标 | 中位 |
|------|------|
| r₅ / r₁₀ | 0.22 / 0.25 |
| A₅,K / A₅/(K/n) | 0.82 / **85×** |
| R²_aff,K^(5) / R²_aff,K | 0.16 / 0.32 |

与 untied U 同档，不像 untied E。

---

## 三层论证（η=95%）

```
rank-1   a₁ ≈ 89%,  31.7× 随机   → 最强方向对齐
rank-5   A₅ ≈ 83%,  30.3× 随机   → 前 5 主方向整体对齐（结论更稳）
rank-10  A₁₀ ≈ 77%, 29.5× 随机  → 前 10 仍显著对齐
全谱     R²_aff,K ≈ 33%          → 整体有效 affine 解释占比
```

**判断模板应用**：

| 条件 | E | U |
|------|---|---|
| r_L 高（全局极低秩）？ | ✗（r₅≈2%） | △（r₅≈22%；**主成分级，非全局 rank-5**） |
| A_{L,K} ≫ K/n？ | ✗（~1.6×） | **✓（~30×，L=1/5/10 均成立）** |
| R²_aff,K substantial？ | ✗（~4%） | **✓（~33%，对主成分仿射表达已满意）** |

---

## 与 centered 分解的关系（线 A）

| | 本分析（raw ΔW, Z=[X,1]） | centered 线 A（D=Y_c−X_c, P=X_c(A−I)） |
|---|---|---|
| E 侧 | R²_aff,K ~4%，A_{L,K} ~5%，倍率 ~1.6× | P/D ~5%，R 主导 |
| U 侧 | R²_aff,K ~33%，R²_aff,full ~36%，A_{L,K} ~77–83%，倍率 ~30× | P/D ~32%（BI-clean 13 个 untied pair 中位），P 谱集中 |

**勿只读一侧**：U 侧 affine 在 raw ΔW 与 centered D 上**同向 substantial**；mean-shift 占 raw Δ 仅 ~2–8%，去中心化不改变 E/U 分裂。详见 [`README.md`](README.md) 线 A「Raw ΔW 对照」。

---

## 最短结论

> **rank-1 只是第一证据；rank-5/10 表明 U 侧的前 10 个主变化方向整体（非偶然）显著落在 X 的有效 affine 子空间里（A_{L,K} ~77–83%，倍率 ~30× 稳定）。**  
> 受 r_L 中等（r₅~22%）限制，前 L 方向对总 ΔW 的直接解释量为 13–16%；全谱 R²_aff,K ~33%——**不要求全局 rank-5 重构，但对「affine 高效表达主变化成分」已是 substantial 占比**。  
> E 侧 r_L 极低、A_{L,K}≈随机，无论 rank-1 还是 rank-L 均不支持 affine 解释。  
> U 的 affine 解释中约 1/3 来自 bias 型（B₅~0.34），其余为 X(A−I) 型结构。

完整叙事（理论 → 实测 → 主结论）：[`notes/02_affine_effective_update_insight.md`](../notes/02_affine_effective_update_insight.md)  
Aff/LoRA 同解释量验证：[`AFFINE_LORA_BUDGET.md`](AFFINE_LORA_BUDGET.md)
