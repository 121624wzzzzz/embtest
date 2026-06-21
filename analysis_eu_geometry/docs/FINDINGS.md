# E/U checkpoint 几何审计：核心发现

本目录对 `configs/models.yaml` 中 **94 个模型** 的存盘 embedding（E）与 lm_head（U）做 checkpoint 几何审计（行范数、μ-ratio、谱分析）。数据来自 `extracts/*.safetensors`，与 `cross_model_geometry` Task1–6 主线结果独立。

## 1. 问题背景

主线 Task1（Base–Instruct GCorr）中，**Gemma-4** 出现反常：Cosine GCorr 仍较高，但 raw Euclidean GCorr 明显偏低。这促使对 checkpoint 存盘 embedding 的**行范数标尺（row-norm gauge）**做全库排查。

## 2. 全库结论：Gemma-4 Base 是唯一 near-unit-sphere 异常

对 94 模型逐行统计 E/U 行范数的 mean、var、min、max 后：

- **Gemma-4 四个 Base**（E2B / E4B / 26B-A4B / 31B）的 E 行范数 mean ≈ **1.000000**，var 在 **10⁻⁹～10⁻⁷** 量级，range 极窄（约 ±0.001）。
- **Gemma-4 Instruct** 不再保持 unit sphere：mean 上移到约 **1.12–1.52**，var 与 range 明显变大。
- **其余 90 个模型**（Qwen、Llama、DeepSeek、GLM、MiniMax、Gemma-2/3 等）均无同类「Base 全 near-unit + Instruct 径向漂移」组合。

因此：**Gemma-4 Base 存盘 embedding 的 near-unit 行范数是全库 outlier**，不像普通初始化噪声，更像是预训练/存盘阶段的 deliberate 球面参数化。

## 3. 与 forward `sqrt(hidden_size)` 的关系

Gemma-4 的 HuggingFace 实现中，text embedding 在 forward 时会再乘 `embed_scale = sqrt(hidden_size)`。这是 **runtime activation 尺度**，与 checkpoint 里行范数 ≈ 1 **是两件事**：

- checkpoint：**方向 + 近单位半径** 存在权重里；
- forward：**固定标量** 把 near-unit 行向量放大到 ~√d 的 activation 尺度。

不能用手动乘 √d 来解释「为何存盘行范数恰好 ≈ 1」；后者更可能来自训练/存盘约束，而非 attention 里的 `1/√d_k`。

## 4. 对 Task1 / GCorr 分析的含义

若直接比较 Gemma-4 Base 与 Instruct 的 **raw checkpoint embedding**：

- **Cosine / row-unit Euclidean**：更接近「方向结构是否对齐」；
- **raw Euclidean**：同时惩罚方向差异与 Base near-unit vs Instruct 更大半径的**径向标尺差**，会得到误导性低相似度。

**建议**：对 Gemma-4 Base–Instruct，主结论优先报告 cosine 或 row-normalized Euclidean；raw Euclidean 若保留，需标注受 checkpoint row-norm gauge 影响。

## 5. 静态谱分析（untied 模型：E vs U）

除行范数外，本目录第 3 层（`spectral/`）对全部 94 模型的存盘 E/U 权重矩阵做 **GPU Gram economy SVD**，得到全部 d 个奇异值的能量分布（即"功率谱"），输出 effective_rank、PR/d、rank1/5/10 能量占比、centered 谱等指标。原始数据在 `results/layers/layer3_spectral.csv`，合并总表在 `results/all_models_eu_features.csv`。

本节聚焦 **50 个 untied 模型（E ≠ U）** 的 E 与 U 静态谱对比。其中 **DeepSeek-V4-Flash / DeepSeek-V4-Pro** 因 hc_head 架构（裸 `head.weight` ≠ 完整 lm_head）作为架构例外排除，剩余 **48 个可比模型**。例外说明见 [`dsv4_hc_head`](../../docs/分析口径与特殊案例.md#63-dsv4_hc_head)。

### 5.1 核心指标一览（48 untied，中位数）

| 指标 | E | U | U/E | U > E 成立 |
|------|:--:|:--:|:---:|:---:|
| **rank1 能量占比**（σ₁² / Σσᵢ²） | 1.0% | **5.3%** | **5.2×** | **48/48** |
| **‖μ‖ / mean(row norm)** | 0.09 | **0.21** | **2.3×** | **44/48** |
| **\|cos(v₁, μ̂)\|** | 0.97 | 0.996 | ~1.0× | 44/48（U 略高） |
| rank5 能量占比（累计） | 3.1% | 6.0% | 2.0× | 38/48 |
| rank10 能量占比（累计） | 4.4% | 6.6% | 1.5× | 37/48 |
| rank1(c) 去中心后 | 0.6% | 1.1% | 1.8× | 43/48 |
| **effective_rank** | 2,840 | **3,649** | 1.3× | 35/48 |
| effective_rank(c) | 2,962 | 4,517 | 1.5× | — |
| **PR/d**（各向同性） | 0.31 | **0.10** | 0.32× | — |

### 5.2 rank1：U 严格高于 E（48/48）

排除 DSV4 后，**48/48 模型满足 U 的 raw rank1 能量占比 > E**。无一例外。中位差距 5.2×。

这一结论的成立不依赖任何阈值选择——它是逐模型成对比较的严格方向一致性。

### 5.3 机制：mean 强度 × v₁ 对齐 → rank1

rank1 由两项因子驱动：

1. **mean 相对强度**（‖μ‖ / mean row norm）：U 约为 E 的 **2.3×**（44/48）
2. **v₁ 与 μ 的对齐度**（|cos(v₁, μ̂)|）：两侧均很高（E 0.97，U 0.996）

rank1 差距（5.2×）的**主要驱动是 mean 强度（2.3×）**，v₁-μ 对齐度对 U/E 差异的贡献有限——因为两边都已接近 1。

**值得注意的分布差异**：U 侧 cos(v₁, μ̂) 始终高位（min 0.79），说明 lm_head 的 leading 方向**强制与 mean 对齐**。E 侧的分布则更宽（min 0.10），部分模型（Qwen2.5-14B 0.12、GLM-5 0.28、DeepSeek-R1-Distill-Qwen-14B 0.10）的 v₁ 与 μ 几乎正交——这些模型的 E rank1 被进一步压低，反而放大了 U/E 差距。

### 5.4 不是"前几个主成分都更集中"，只是 rank1

将 Frobenius 能量按奇异值分解，比较 rank2–5、rank6–10 的**增量**（非累计）能量：

| 能量段 | U > E | E 中位 | U 中位 | 含义 |
|--------|:-----:|--------|--------|------|
| rank1（σ₁²） | **48/48** | 1.0% | 5.3% | U 的 leading 方向绝对集中 |
| rank2–5 增量 | **0/48** | 1.5% | 0.9% | U 的后续方向反而更散 |
| rank6–10 增量 | **0/48** | 1.2% | 0.6% | 同上 |

**48/48 模型同时满足**：U rank1 > E，且 U 的 rank2–5 增量 ≤ E。因此 U 的 raw 谱"更集中"**仅体现在 rank1**，rank2–10 方向 U 反而更发散。不能说 U 整体更低秩。

### 5.5 effective_rank 与 PR/d：两个谱指标的对比

`spectral.py` 对每个矩阵的 SVD 计算两个独立的"有效维度"指标，两者从同一组 σᵢ² 出发但反映不同的分布特性。下面对比均针对 **48 可比 untied 模型**（排除 DSV4）。

#### 定义

设 `pᵢ = σᵢ² / Σⱼ σⱼ²`（归一化奇异值平方，∑pᵢ = 1，d = hidden_dim）：

| 指标 | 公式 | 等价形式 | 对 pᵢ 的敏感阶 |
|------|------|----------|:--:|
| **effective_rank** | exp(−Σ pᵢ log pᵢ) | exp(H)，H = Shannon 熵 | 全阶矩（log 权重） |
| **participation_ratio（PR）** | (Σ σᵢ²)² / Σ σᵢ⁴ | 1 / Σ pᵢ² | 二阶矩（平方权重） |
| **PR/d** | PR / d | 归一化各向同性 | — |

**直觉**：
- effective_rank → 如果分布均匀（pᵢ = 1/d），则 = d；如果全部能量集中在一个分量（p₁=1），则 = 1
- PR → 同样是 d（均匀）到 1（完全集中），但**对少数大分量更敏感**（因为平方放大了大值）

#### 统计值（48 untied，中位数）

| | E | U | U vs E | 说明 |
|--|:--:|:--:|:--:|------|
| effective_rank（raw） | 2,840 | **3,649** | **U > E**（1.3×） | U 判为更高维 |
| effective_rank（centered） | 2,962 | **4,517** | **U > E**（1.5×） | 去 rank1 后差距扩大 |
| PR（raw） | — | — | **U < E**（~0.27×） | 与 effective_rank 相反！ |
| PR/d（raw） | 0.31 | **0.10** | U 更各向异性 | 仅 ~10% 的有效维度利用率 |
| PR/d（centered） | — | — | **U > E**（~1.75×） | 去 rank1 后反转 |
| rank1 energy（raw） | 1.0% | 5.3% | U 5.2× E | rank1 是 U 的主导分量 |

#### 为什么 raw PR 判 U < E 而 effective_rank 判 U > E

两者的排序矛盾**不是 bug，而是 rank1 对两者的影响权重不同**：

**具体机制**：U 的 p₁ ≈ 5.3%，E 的 p₁ ≈ 1.0%。

- **PR = 1 / Σ pᵢ²**：U 的 p₁² = 0.0028 vs E 的 p₁² = 0.0001。U 的分母被单个大 p₁ 显著拉高，PR 急剧缩小。这个平方效应使得 U 的 PR 被 rank1"压垮"。
- **effective_rank = exp(−Σ pᵢ log pᵢ)**：U 的 −p₁ log p₁ ≈ 0.16 vs E 的 −p₁ log p₁ ≈ 0.046。差异被 log 压缩。同时 U 在 rank2–d 的尾部分布比 E 更均匀（因 rank1 后的增量能量 U < E），熵在尾部补齐，最终 effective_rank 反而 U > E。

**数值直觉**（假设 d=1000）：
- U：p₁=0.053，其余 999 个均匀 → PR≈283，eff_rank≈573
- E：p₁=0.010，其余 999 个均匀 → PR≈497，eff_rank≈833
- 但实际情况 U 的尾部确实比 E 更平（rank2–5 增量 0/48 U<E），所以 effective_rank 最终 U > E。

#### 去中心后的收敛

去掉 mean 向量（即剥离 rank1 的主要来源）后：

- centered PR/d 从 raw 的 U < E（0.32×）**反转为 U > E（~1.75×）**
- centered effective_rank 保持 U > E（差距从 1.3× 扩大到 1.5×）

**这意味着**：raw 谱中 U 的 rank1（主要来自 mean）掩盖了其尾部更均匀的事实。去掉 rank1 后，U 的剩余奇异值分布比 E 更各向同性——这与"U 的 rank2+ 更发散"（5.4 节）完全一致。

### 5.6 按模型族的典型值

| 系列 | E rank1 | U rank1 | 特征 |
|------|---------|---------|------|
| DeepSeek V2/V3/R1（12 模型） | ~1% | ~1–3% | 差距温和（~1.3×），U 略高 |
| Qwen2.5 大模型（14B–72B） | ~0.6% | ~2–7% | U 远超 E（~3–5×） |
| Qwen3 大模型（8B–32B） | ~2% | ~2–6% | 差距适中 |
| Qwen3.5/3.6 MoE | ~1–2% | ~3–12% | U 极高 rank1，差距最大 |
| Llama-3.1 70B | ~1% | ~3% | 差距温和 |
| Llama-3.1 8B | ~2% | ~8% | 差距显著 |
| Qwen2.5-14B | 0.6% | 2.1% | E v₁⊥μ（cos=0.12），U 正常 |

### 5.7 DeepSeek-V4：唯一的架构例外

50 untied 中仅 DeepSeek-V4-Flash / Pro 出现 **E rank1 > U rank1**（E 5.5%/3.5% vs U 2.8%/2.7%）。这是因为 V4 的输出路径为 `HC states → hc_head → RMSNorm → head.weight → logits`，裸 `head.weight` 不是完整 lm_head，E 承担了更多 HC 偏置角色。详见 [`dsv4_hc_head`](../../docs/分析口径与特殊案例.md#63-dsv4_hc_head)。

## 6. BI 行范数漂移（Δ mean）与 tied/untied

对 35 对 BI，剔除 Gemma-4（4 对，行范数 gauge 特殊）与 Gemma-3-1B（1 对，后训练改写过大）后，剩余 30 对。排除原因见 [`BI-excluded`](../../docs/分析口径与特殊案例.md#42-bi-excluded-5-对)。Δ% = `(Instruct E/U mean − Base E/U mean) / Base mean × 100`。

| 分组 | 对数 | \|ΔE\| 中位 | \|ΔU\| 中位 | 说明 |
|------|------|------------|------------|------|
| **untied**（E≠U） | 13 | **0.31%** | **0.87%** | E、U **分别统计**；U 通常漂得更多 |
| **tied**（E=U） | 17 | 1.02% | (=ΔE) | 只一条标尺；若只排除 Gemma-4、包含 Gemma-3-1B，则 n=18、\|ΔE\| 中位为 **1.25%**（Gemma-3-1B 单对为 −9.48%） |
| 全 30 对（排除 5 异常） | 30 | 0.61% | — | 常态 ±1% 以内；Gemma-4 为 +12%～+52%，Gemma-3-1B 为 −9.5% |

**untied 要点**（详见 `BI_ROW_NORMS_SUMMARY.md` 的 Untied BI 节）：

- Qwen3 8B+ / Qwen2.5 7B+：\|ΔE\| ~0.1%–1.2%，\|ΔU\| ~0.3%–2.7%
- Llama-3.1：\|ΔE\| ~0.4%，\|ΔU\| ~1.4%
- DeepSeek V3/V3.1：≈ 0%

配对数值列：`results/bi_pair_delta.csv` 的 `delta_E_pct` / `delta_U_pct`。

## 7. 数据与复现

| 文件 | 说明 |
|------|------|
| **全库特征总表** | `results/all_models_eu_features.csv`（188 行，三层合一） |
| BI 配对 Δ | `results/bi_pair_delta.csv` |
| 层 1 / 2 / 3 | `results/layers/layer1_row_norms.csv`、`layer2_mu_ratio.csv`、`layer3_spectral.csv` |
| BI / 其他中间表 | `results/bi_row_norms.csv`、`other_models_row_norms.csv` |
| 静态×仿射交叉 | `results/rank1_vs_affine_untied_gpu.csv`（13 对 untied BI 的静态谱指标与仿射分解指标宽表，用于检查静态 rank1 对仿射可解释性的预测力） |
| rank1(c) vs rank2 | `results/rank2_vs_rank1c_untied_gpu.csv`（101 行，验证去中心后 rank1(c) ≈ 原 raw 谱的 rank2，即 centering 剥离的是 mean 主导的 rank1） |

```bash
conda activate wzall
python analysis_eu_geometry/scripts/audit_eu_geometry.py
```

只合并已有 CSV、不重跑统计：

```bash
python analysis_eu_geometry/scripts/audit_eu_geometry.py --merge-only
```

只跑 μ-ratio：

```bash
python analysis_eu_geometry/scripts/audit_eu_geometry.py --mu-ratio-only --workers 8
```

只跑 GPU 谱分析：

```bash
CUDA_VISIBLE_DEVICES=0 python analysis_eu_geometry/scripts/audit_eu_geometry.py --svd-only --device cuda:0
```

## 8. Gemma-4 速查（E，mean）

| 模型 | mean (var 量级) |
|------|-----------------|
| Gemma-4-E2B Base | 0.999998 (10⁻⁹) |
| Gemma-4-E2B-Instruct | 1.193636 (10⁻²) |
| Gemma-4-E4B Base | 0.999999 (10⁻⁹) |
| Gemma-4-26B-A4B Base | 0.999902 (10⁻⁹) |
| Gemma-4-31B Base | 0.999800 (10⁻⁷) |

完整表格见 `results/BI_ROW_NORMS_SUMMARY.md` 的 Gemma4 节。
