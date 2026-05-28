# E/U checkpoint 几何审计：核心发现

本目录对 `configs/models.yaml` 中 **94 个模型** 的存盘 embedding（E）与 lm_head（U）做 checkpoint 几何审计（行范数、μ-ratio、谱分析）。数据来自 `extracts/*.safetensors`，与 `ijcai_clean` Task1–6 主线结果独立。

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

## 5. 谱分析与特殊案例

除行范数外，本目录还补充了 `mu_ratio/` 与 `spectral/`：

- `mu_ratio/`：统计 `||μ|| / mean(row norm)`，衡量行向量公共 mean 方向强度；
- `spectral/`：用 GPU Gram 做完整 economy SVD，输出 PR/d、rank1、centered rank1 等各向同性/异性指标。

静态 U vs E 谱专题（untied rank1 / tied≈U）见 [`../analysis_1/`](../analysis_1/)。

两个特殊排错案例汇总在 [`SPECIAL_CASES.md`](SPECIAL_CASES.md)：

- **Gemma-4**：Base 存盘 E 是 near-unit-sphere gauge；Instruct 径向漂移导致 raw Euclidean GCorr 误导。
- **DeepSeek-V4**：输出路径为 `HC states -> hc_head -> RMSNorm -> head.weight -> logits`，裸 `head.weight` 不等价于传统完整 lm_head；当前无法分析完整有效输出头，因此 DSV4 的裸 U 谱结果对本目录的 embedding 存盘几何主线解释意义较低，仅作为架构例外记录。

## 6. BI 行范数漂移（Δ mean）与 tied/untied

对 35 对 BI，Δ% = `(Instruct E/U mean − Base E/U mean) / Base mean × 100`。

| 分组 | 对数 | \|ΔE\| 中位 | \|ΔU\| 中位 | 说明 |
|------|------|------------|------------|------|
| **untied**（E≠U） | 13 | **0.31%** | **0.87%** | E、U **分别统计**；U 通常漂得更多 |
| **tied**（E=U） | 22 | 1.25% | (=ΔE) | 只一条标尺；含 Gemma-3-1B、Llama-3.2 等 outlier |
| 排除 Gemma-4 全库 | 31 | 0.65% | — | 常态 ±1% 以内；Gemma-4 为 +12%～+52% |

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
