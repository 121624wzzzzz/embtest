# Untied：静态 U vs E 谱——mean 更强，且仅 rank1 更集中

> **对象**：存盘 checkpoint 权重矩阵（非 BI 变化量 ΔW）。  
> **范围**：**48 个可比 untied 模型**（全库 50 untied，排除 DeepSeek-V4-Flash / DeepSeek-V4-Pro）。  
> **数据**：[`data/static/layer3_spectral.csv`](../data/static/layer3_spectral.csv)。

---

## 主结论

在可比 untied 模型上，**U（lm_head）相对 E（embed）的 raw 谱「更集中」几乎完全体现在 rank1（σ₁² 能量占比）**；**不是**「前几个主成分整体更集中」。

机制链：

1. **U 的 mean 相对行尺度远大于 E**（‖μ‖ / mean(row norm) 约 **2.3×**，44/48）；
2. U 的 **leading 奇异方向与 μ 几乎重合**（|cos(v₁, μ̂)| 两侧都很高，U 略高）；
3. 二者叠加 → **raw rank1：U 5.2× 于 E，48/48 严格成立**；
4. 拆增量谱后：**rank2–5、rank6–10 的增量能量 48/48 均为 U < E** → 更高阶主方向 U 反而更散。

---

## 1. 三个核心指标（48 untied，中位数）

| 指标 | E | U | U/E | 成立 |
|------|:---:|:---:|:---:|:---:|
| **‖μ‖ / mean(row norm)** | 0.09 | **0.21** | **2.3×** | **44/48** |
| **rank1（raw，σ₁² 占比）** | 1.0% | **5.3%** | **5.2×** | **48/48** |
| **\|cos(v₁, μ̂)\|** | 0.97 | 0.996 | ~1.0× | 44/48（U 略高） |

**读法**：

- mean 相对行尺度：**U 明显更大**（主因之一）；
- rank1 raw：**U 远高于 E**，排除 DSV4 后 **100%（48/48）**；
- v₁ 与 μ 对齐：E/U **两侧都已很高**（~0.97–1.0），U 只是略高 → rank1 差距（5×）**不能**只归因于对齐差异，**mean 强度（2.3×）是关键放大项**。

---

## 2. 不是「前几项更集中」，而是「只有第一项更集中」

对 48 模型，将 Frobenius 能量按奇异值分解，比较 **增量段**（非累计）：

| 能量段 | U > E | E 中位 | U 中位 | U/E |
|--------|:-----:|--------|--------|:---:|
| **rank1（σ₁²）** | **48/48** | 1.0% | 5.3% | **5.2×** |
| rank2–5（增量） | **0/48** | 1.5% | 0.9% | 0.56× |
| rank6–10（增量） | **0/48** | 1.2% | 0.6% | 0.50× |
| rank5（累计） | 38/48 | 3.1% | 6.0% | 2.0× |
| rank10（累计） | 37/48 | 4.4% | 6.6% | 1.5× |

**48/48 模型同时满足**：U rank1 > E，且 U 的 rank2–5 增量 ≤ E。

rank1 占 rank5 累计能量的比例：E 中位 **41%**，U 中位 **82%** → U 的前 5 个方向里，**绝大部分能量已在 rank1**；E 则相对均匀。

**因此**：rank5 / rank10 累计值虽常 U > E，**完全由 rank1 拉高**；不能说 U 的 rank2–10 主方向也更集中。

---

## 3. 去中心（centered）后：mean 贡献被剥离，只剩弱 residual 信号

| 指标 | U > E | E 中位 | U 中位 | U/E |
|--------|:-----:|--------|--------|:---:|
| rank1 → rank1(c) 降幅 | — | 0.53% | **4.04%** | U 掉得更多 |
| **rank1(c)** | 43/48 | 0.62% | 1.11% | 1.8× |
| rank2–5(c) 增量 | **1/48** | 1.3% | 0.9% | 0.70× |

- raw rank1 优势 **largely 来自 mean**（U 去中心后 rank1 掉 ~4%，E 仅 ~0.5%）；
- centered 后 U rank1(c) 仍略高（43/48），但差距缩至 **1.8×**，且 rank2–5(c) **不跟** → 不宜与 raw rank1 同等强度表述。

---

## 4. 排除 DSV4 的说明

全库 50 untied 中，仅 **DeepSeek-V4-Flash / Pro** 出现 raw rank1：E > U（E 承担更多 HC 初始化偏置，裸 U 非传统 lm_head）。  
主线 untied 规律：**48/48，U rank1 > E rank1**。详见 [`docs/SPECIAL_CASES.md`](../docs/SPECIAL_CASES.md) §2。

---

## 5. 建议表述（可直接引用）

**中文**：

> 在 48 个可比 untied 模型的静态 checkpoint 谱上，U 相对 E 的 leading 集中 **几乎完全是 rank1 现象**：U 的 ‖μ‖/mean(row norm) 约为 E 的 2.3×（44/48），raw rank1 能量占比约为 E 的 5.2×（48/48），且 v₁ 与 μ 在两侧均已高度对齐（U 略高）。增量谱显示 rank2–10 能量 **48/48 为 U < E**，故不是「前几个主成分整体更集中」，而是 **「U 的 mean 轴 / rank1 显著更强」**。

**English**：

> *Among 48 comparable untied models (excluding DeepSeek-V4), the U-side static spectrum is more concentrated only in the leading component: μ/row-norm ≈2.3× (44/48), raw rank1 energy fraction ≈5.2× (48/48), with v₁ ≈ μ on both sides (slightly stronger on U). Incremental ranks 2–10 are less concentrated on U (0/48). This is a rank-1 / mean-axis effect, not broad low-rank concentration across the first several components.*
