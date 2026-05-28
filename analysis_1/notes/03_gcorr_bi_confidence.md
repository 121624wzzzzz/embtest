# Task1 BI：E_cos / E_euc 与 1 的距离（置信区间）

> **对象**：Base–Instruct 同模型对的 input embedding GCorr（Task1）。  
> **范围**：与静态谱 note 02 对齐——**untied 13 对**；**tied 17 对**（排除 Gemma-4 全系、Gemma-3-1B）。  
> **数据**：[`data/task1_base_instruct/summary.csv`](../data/task1_base_instruct/summary.csv)（逐对 bootstrap CI）；[`data/task1_base_instruct/bootstrap_results.csv`](../data/task1_base_instruct/bootstrap_results.csv)（100 次重采样组均值 CI）。  
> **模型子集**：[`configs/analysis_subsets.yaml`](../configs/analysis_subsets.yaml)（`task1_untied_13`、`task1_tied_17_excl_gemma4_3_1b`）。  
> **指标**：`E_cos`（方向）、`E_euc`（欧氏距离 GCorr）。缺口记 **δ = 1 − mean**；**log₁₀(δ)** 表示与 1 的数量级距离。

---

## 主结论（一句话）

在现有模型尺度上：**untied 的 `E_euc` 几乎全为 1**（13/13 ≥ 0.999，δ ~ 10⁻⁵）；**tied（排除 Gemma 异常）的 `E_cos` 同样极高**（均值 0.994，δ ~ 10⁻³，与 untied `E_cos` 同量级），而 **`E_euc` 才有明显 spread**（range ~ 0.11，δ ~ 10⁻²，比 untied 大 **~2 个数量级**）——缺口主要是尺度漂移，不是方向失配。

---

## 1. 对照表：一个「几乎全 1」，一个有 range

| 分组 | n | 指标 | mean | **range** [min, max] | δ = 1−mean | log₁₀(δ) | ≥0.999 对数 |
|------|---|------|------|----------------------|------------|----------|-------------|
| **untied** | 13 | **E_euc** | **0.99996** | **[0.99982, 1.00000]**（Δ=**1.8×10⁻⁴**） | **3.6×10⁻⁵** | **−4.4** | **13/13** |
| untied | 13 | E_cos | 0.9974 | [0.9866, 1.0000]（Δ=1.3×10⁻²） | 2.6×10⁻³ | −2.6 | 6/13 |
| tied* | 17 | **E_cos** | **0.9941** | **[0.9792, 0.9997]**（Δ=**2.1×10⁻²**） | **5.9×10⁻³** | **−2.2** | 6/17 |
| tied* | 17 | E_euc | 0.9828 | [0.8940, 0.9999]（Δ=**1.1×10⁻¹**） | 1.7×10⁻² | −1.8 | 3/17 |

\*tied：排除 Gemma-4 全系与 Gemma-3-1B（gauge / BI 主分析排除，见 [`docs/SPECIAL_CASES.md`](../docs/SPECIAL_CASES.md)）。

**读法**：

- **「几乎全 1」**：untied **`E_euc`**——range 仅 **~2×10⁻⁴**，11/13 对 ≥ 0.99995，组 bootstrap CI **[0.999963, 0.999965]**。
- **「另外一个有 range」**：tied **`E_euc`**——最差对 Gemma-3-4B 仅 0.894，全组 range **0.106**；但同组 **`E_cos` range 仅 0.021**，且 11/17 对 ≥ 0.99。
- **数量级**：untied `E_euc` 的 δ（**~10⁻⁵**）比 tied `E_euc` 的 δ（**~10⁻²**）小 **~480×**（**~2.6 个 log 单位**）；tied `E_cos` 的 δ（**~10⁻³**）与 untied `E_cos`（**~10⁻³**）**同量级**，仅约 **2.3×**。

---

## 2. Bootstrap 组均值 CI（100 次，逐对 mean 再组平均）

| 分组 | E_cos mean [CI95] | E_euc mean [CI95] |
|------|-------------------|-------------------|
| untied (13) | 0.9974 [0.9973, 0.9976] | **1.0000 [1.0000, 1.0000]** |
| tied* (17) | **0.9941 [0.9910, 0.9946]** | 0.9828 [0.9812, 0.9841] |

**untied − tied* 差值**（bootstrap CI，均不含 0，但绝对值小）：

| 指标 | Δ mean | Δ 的 CI95 | δ 之比 (untied / tied) |
|------|--------|-----------|-------------------------|
| E_cos | +0.0034 | [+0.0028, +0.0064] | 2.6×10⁻³ / 5.9×10⁻³ ≈ **0.44×**（tied 缺口略大） |
| E_euc | +0.0172 | [+0.0159, +0.0187] | 3.6×10⁻⁵ / 1.7×10⁻² ≈ **480×**（untied 更贴 1） |

---

## 3. tied* 内部：E_cos 高、E_euc 低的典型对

12/17 对 **E_cos > E_euc**（Insight 3：方向一致、尺度漂移）。

| 模型 | E_cos | E_euc | 1−E_cos | 1−E_euc | 备注 |
|------|-------|-------|---------|---------|------|
| Gemma-3-4B | 0.979 | 0.894 | 2.1×10⁻² | **1.1×10⁻¹** | E_euc 离 1 最近（−1.0 log） |
| Gemma-3-12B | 0.988 | 0.946 | 1.2×10⁻² | 5.4×10⁻² | |
| Llama-3.2-1B | 0.982 | 0.970 | 1.9×10⁻² | 3.0×10⁻² | |
| Gemma-2-27B | 0.998 | 0.982 | 1.8×10⁻³ | 1.8×10⁻² | cos≈1，euc 仍低 ~10× |
| Qwen2.5-3B | 0.9997 | 0.9999 | 3.4×10⁻⁴ | 1.1×10⁻⁴ | 双高 |

Gemma-3 小中模型拉低 tied `E_euc` 下界；Qwen / Gemma-2 小模型双指标均 ≥ 0.996。

---

## 4. 补充：tied 非 Gemma 全系（11 对）

若进一步排除全部 Gemma tied（含 Gemma-2/3），**两指标都更接近 1**，与 untied 差距缩至 **~10⁻³**：

| 指标 | mean | range | δ | log₁₀(δ) | ≥0.999 |
|------|------|-------|---|----------|--------|
| E_cos | 0.9950 | [0.9815, 0.9997] | 5.0×10⁻³ | −2.3 | 5/11 |
| E_euc | 0.9943 | [0.9700, 0.9999] | 5.7×10⁻³ | −2.2 | 2/11 |

相对 untied `E_euc`（δ ~ 10⁻⁵），非 Gemma tied 的 `E_euc` 仍差 **~2 log 单位**，但已无 0.89 档 outlier。

---

## 5. 建议表述（中英）

**中文**

> 在 Task1 的 13 对 untied Base–Instruct 上，input embedding 的 **Euclidean GCorr（E_euc）几乎全为 1**（13/13 ≥ 0.999，均值 0.99996，与 1 的缺口 δ ≈ **3.6×10⁻⁵**，即 **10⁻⁴·⁴ 量级**；逐对 range 仅 **1.8×10⁻⁴**）。  
> 17 对 tied（排除 Gemma-4/3-1B）中，**cosine GCorr（E_cos）同样极高**（均值 0.994，δ ≈ **5.9×10⁻³**，与 untied E_cos 同 **10⁻³ 量级**；range **0.021**），而 **E_euc 才有明显 spread**（range **0.106**，δ ≈ **1.7×10⁻²**，比 untied E_euc 离 1 远 **~480× / ~2.6 log**）。这与「方向高度一致、欧氏结构受尺度漂移影响」（Insight 3）一致；不宜仅用 tied E_euc 概括 BI 对齐质量，应并列 **E_cos ≈ 0.99+**。

**English**

> *On 13 untied Base–Instruct pairs, input-embedding **E_euc is essentially unity** (13/13 ≥ 0.999; mean 0.99996; gap δ ≈ 3.6×10⁻⁵, i.e. ~10⁻⁴·⁴; pair range 1.8×10⁻⁴). Among 17 tied pairs (excluding Gemma-4 / Gemma-3-1B), **E_cos remains near unity** (mean 0.994; δ ≈ 5.9×10⁻³, same 10⁻³ order as untied E_cos; range 0.021), while **E_euc shows the spread** (range 0.106; δ ≈ 1.7×10⁻², ~480× / ~2.6 log farther from 1 than untied E_euc)—consistent with directional alignment with scale drift (Insight 3).*

---

## 6. 与静态谱 / Task2 的合读（Insight 1 主证据）

Insight 1 的 **主证据** 见专文 [`04_insight1_main_evidence.md`](04_insight1_main_evidence.md)——**Task2 37/37** 与 **静态谱（note 02）均不依赖 Gemma-4 BI 的 E_euc**。本 note（Task1 BI）仅为补充。

| 证据线 | 结论 | Insight 1 权重 |
|--------|------|----------------|
| Task2 tied×untied | **37/37** U 侧 GCorr > E 侧 | **主证据** |
| 静态谱（note 02） | tied 共享矩阵谱像 untied **U** | **主证据** |
| Task1 BI（本 note） | untied **E_euc ≈ 1**；tied **E_cos ≈ 0.99**，E_euc 有 range | 补充 |
