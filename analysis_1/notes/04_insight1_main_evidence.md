# Insight 1 主证据：tied 共享矩阵 ≈ untied U（不依赖 Gemma-4 BI E_euc）

> **Insight 1（论文）**：tied 模型的共享 E=U 矩阵，在跨模型几何上更接近 **untied 的 U（lm_head）**，而非 untied 的 **E（embed）**。  
> **本文档**：归纳 **主证据线**——二者均 **不经过 Task1 tied BI 的 E_euc**，故 **不受 Gemma-4 gauge 异常（BI E_euc 崩到 ~0.4–0.7）影响**。

---

## 主证据（两条，独立于 Gemma-4 BI）

| # | 证据 | 数据 | 结论 | 与 Gemma-4 BI 关系 |
|---|------|------|------|-------------------|
| **A** | **Task2** tied×untied GCorr | [`data/task2_model_series/tied_x_untied_pairs.csv`](../data/task2_model_series/tied_x_untied_pairs.csv) | **37/37** 对 U 侧 > E 侧 | 跨 **不同模型** 的静态权重 GCorr；不用 Base–Instruct BI |
| **B** | **静态谱** tied vs untied E/U | [`notes/02_tied_static_spectrum_like_u.md`](02_tied_static_spectrum_like_u.md) | 34 tied（excl Gemma-4/3-1B）谱 **像 untied U** | checkpoint **权重 SVD**；与 Task1 BI 无关 |

**写法要点**：Insight 1 应 **优先引用 A + B**；Task1 BI（[`03_gcorr_bi_confidence.md`](03_gcorr_bi_confidence.md)）是 **补充语境**（untied embed 跨 checkpoint 几乎不变、tied 方向仍高等），**不是** Insight 1 的必要前提。

---

## A. Task2：37 对 tied×untied

**设定**：Task2 为同系列 Instruct 模型间的 token 几何相关（GCorr）；筛选 **一侧 tied、一侧 untied** 的 37 对。

| 指标 | U 侧 mean | E 侧 mean | U > E |
|------|-----------|-----------|-------|
| **cos** | **0.8114** | 0.1395 | **37/37** |
| **euc** | **0.8169** | 0.5553 | **37/37** |

- tied 共享矩阵与 untied **U** 的 GCorr 约为与 untied **E** 的 **5–6×（cos）** / **~1.5×（euc）**。
- 该对比在 **任意 tied 模型**（含 Gemma 系）与 **任意 untied Instruct** 之间做，**不读 Task1 Base–Instruct 的 E_euc**。
- Gemma-4 若出现在 Task2 对中，影响的是 **跨模型静态 GCorr**，与 Gemma-4 在 Task1 BI 里 E_euc≈0.44–0.68 的 gauge 问题是 **不同实验**。

---

## B. 静态谱：tied 更像 untied U

见 [`02_tied_static_spectrum_like_u.md`](02_tied_static_spectrum_like_u.md)（34 tied，排除 Gemma-4 全系、Gemma-3-1B）。

| 指标 | untied E | untied U | tied* |
|------|:--------:|:--------:|:-----:|
| rank1（raw） | 1.0% | 5.3% | **11.4%** |
| ‖μ‖/mean(row norm) | 0.09 | 0.21 | **0.32** |
| \|cos(v₁, μ̂)\| | 0.97 | 0.996 | **0.999** |

- **0/34** tied 的 rank1 或 μ/row 低于 untied E 中位；**26/34** rank1 高于 untied U 中位。
- 数据源为 [`data/static/layer3_spectral.csv`](../data/static/layer3_spectral.csv) 的 **economy SVD**，与 BI ΔW、Task1 E_euc **无关**。
- 此处排除 Gemma-4/3-1B 是因为 **静态谱 / BI 主分析** 的一贯约定（见 [`BI-excluded`](../../docs/ANALYSIS_SCOPES_AND_SPECIAL_CASES.md#42-bi-excluded-5-对)），**不是因为** 排除后 Insight 1 才成立——Task2 37/37 **已含** Gemma tied 与 untied 的交叉对。

---

## 与 Task1 BI 的分工（为何不依赖 Gemma-4 E_euc）

| 实验 | 问什么 | Insight 1 角色 |
|------|--------|----------------|
| Task2 tied×untied | tied **共享矩阵** vs untied **E/U 谁更像** | **主证据 A** |
| 静态谱 | tied 存盘矩阵 **谱形** 像 E 还是 U | **主证据 B** |
| Task1 BI | 同模型 Base–Instruct **E 变多少** | **非必要**；tied E_euc 受 Gemma gauge / 尺度漂移干扰（见 note 03） |

Gemma-4 在 Task1 BI 上 **E_euc 极低**（~0.44–0.68），会 **削弱** 「用 Task1 tied E_euc 单独论证 Insight 1」的写法，但 **不改变** Task2 37/37 与静态谱结论——故论文中 Insight 1 **应写 A+B，勿把 Gemma-4 BI E_euc 当反例或前提**。

---

## 建议表述

**中文**

> Insight 1 的主证据来自两方面，均不依赖 Gemma-4 在 Task1 BI 上的 E_euc：（1）Task2 中 37 对 tied×untied 跨模型 GCorr，**37/37** 满足 U 侧高于 E 侧（U_cos 均值 0.81 vs E_cos 0.14）；（2）静态 checkpoint 谱上，排除 Gemma-4/3-1B 后的 34 个 tied 共享矩阵 **整体更像 untied U 而非 untied E**（rank1、mean 轴等，见 analysis_1 note 02）。Task1 BI 仅作补充：untied E_euc 几乎为 1，tied 方向 GCorr（E_cos）仍 ~0.99。

**English**

> *The primary evidence for Insight 1 does not rely on Gemma-4 Task1 BI E_euc: (i) Task2 tied×untied GCorr (37/37 pairs, U-side > E-side; mean U_cos 0.81 vs E_cos 0.14); (ii) static spectra of 34 tied models (excluding Gemma-4 / Gemma-3-1B) align with untied U-like profiles (analysis_1 note 02). Task1 BI provides supplementary context only.*
