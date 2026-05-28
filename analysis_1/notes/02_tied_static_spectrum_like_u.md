# Tied：静态谱更像 untied U（排除 Gemma-4、Gemma-3-1B）

> **对象**：存盘 checkpoint 权重矩阵（非 BI ΔW）。  
> **范围**：**34 个 tied 模型**（全库 44 tied，排除 Gemma-4 全系与 Gemma-3-1B）。  
> **数据**：[`data/static/layer3_spectral.csv`](../data/static/layer3_spectral.csv)（tied 仅 E 行，E=U 共享矩阵）。

---

## 主结论

排除 Gemma-4 与 Gemma-3-1B 后，**tied 共享矩阵的静态谱整体更像 untied U，不像 untied E**；且多为 **rank1 / mean 轴主导**（与 [`01_untied_u_vs_e_static_spectrum.md`](01_untied_u_vs_e_static_spectrum.md) 同型）。

---

## 1. 中位数：tied 更接近 untied U，且常超过 U 中位

| 指标 | untied E | untied U | tied* | 更接近 |
|------|:--------:|:--------:|:-----:|:------:|
| **‖μ‖ / mean(row norm)** | 0.09 | 0.21 | **0.32** | **U**（高于 U 中位） |
| **rank1（raw）** | 1.0% | 5.3% | **11.4%** | **U**（高于 U 中位） |
| rank5（raw） | 3.1% | 6.0% | **12.7%** | U |
| rank1（centered） | 0.62% | 1.11% | **2.48%** | U |
| **\|cos(v₁, μ̂)\|** | 0.97 | 0.996 | **0.999** | U |

- **0/34** tied 的 rank1 或 μ/row 低于 untied E 中位；
- **26/34** rank1 **高于** untied U 中位（8 个落在 E–U 区间）；
- rank1 数值更接近 untied U 中位：**28/34**；
- rank5 / rank10 / rank1(c) / PR/d：**29–34/34** 更接近 U。

*untied 中位数口径：48 可比 untied（排除 DSV4），与 note 01 一致。*

---

## 2. 同样是 rank1 / mean 轴现象

| 分组 | rank1 / rank5（中位） | rank2–5 增量（中位） |
|------|:---------------------:|:--------------------:|
| untied E | 0.41 | 1.5% |
| untied U | 0.82 | 0.9% |
| **tied*** | **0.90** | 1.4% |

tied 的 rank1 占 rank5 比例 **0.90**，甚至比 untied U（0.82）更极端 → **不是前几项全面更集中**，而是 **mean 轴 / rank1 主导**。

---

## 3. 与 untied 分裂的合读

| 矩阵 | 静态谱角色 |
|------|-----------|
| untied **E**（embed） | μ 弱、rank1 低、谱 diffuse |
| untied **U**（lm_head） | μ 强、rank1 高、v₁ ≈ μ |
| **tied**（E = U 共享） | **谱落在 U 档**；小模型常超过 untied U 中位 |

这与 [`bi_analysis`](../bi_analysis/) 中「tied E ≈ untied U（affine-friendly）」同向：**共享权重时，性质像 lm_head / U 侧，不像 embed / E 侧**。

**边界**：综合 9 个谱指标 L1 距离，18/34 整体更像 U、16/34 更像 E；偏 E 侧的多为 Gemma-3（4B/12B/27B）、部分 Qwen3 小模型等（rank1 没那么高），但 **仍不像 untied E 那种 rank1 ~1% 的 diffuse 谱**。

---

## 4. 建议表述（可直接引用）

**中文**：

> 排除 Gemma-4 与 Gemma-3-1B 后，34 个 tied 模型的共享矩阵谱整体更像 untied U（rank1 中位 11.4% vs U 5.3% / E 1.0%），且 rank1/rank5 ≈ 0.90，同为 rank1 / mean 轴主导，而非 embed 侧那种 diffuse 谱。

**English**：

> *For 34 tied models (excluding Gemma-4 and Gemma-3-1B), the shared E=U matrix spectra align with untied U-like profiles (median rank1 11.4% vs U 5.3% / E 1.0%), not untied E.*
