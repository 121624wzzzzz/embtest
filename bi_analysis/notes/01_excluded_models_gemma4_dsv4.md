# 不宜纳入 BI 主分析的模型：Gemma-4 与 DeepSeek-V4

> 结论备忘。原始矩阵、脚本与长表仍留在 `analysis_eu_geometry/`、`ijcai_clean/`、`__tep/`。

---

## 总览

| 系列 | 是否 BI pair | 主组处理 | 核心原因 |
|------|-------------|----------|----------|
| **Gemma-4**（4 对） | 是 | **排除**（与 Gemma-3-1B 一并，主组 n=26） | checkpoint 行范数 gauge + 架构（PLE/MoE）使 raw E/U 与仿射 R² 不可与常规模型同口径比较 |
| **DeepSeek-V4**（Flash / Pro） | **否**（无 Base–Instruct） | 不参与 BI 主叙事；静态 U 仅作架构例外记录 | 输出头为 `hc_head → RMSNorm → head.weight`，裸 `head.weight` ≠ 传统 lm_head |

---

## 一、Gemma-4

### 1.1 为何单独成类

1. **Row-norm gauge（存盘 E）**  
   - **Base**：E 行范数 mean ≈ **1.000000**，var **10⁻⁹～10⁻⁷**，全库唯一 near-unit-sphere outlier。  
   - **Instruct**：mean **1.12～1.52**，var **~10⁻²**，SFT 后径向标尺明显漂移。  
   - 后果：Task1 上 **cosine GCorr 仍高，raw Euclidean GCorr 明显偏低**——raw 距离把「半径差」和「方向差」混在一起。

2. **架构**  
   - Gemma 4 含 **Per-Layer Embeddings（PLE）** 等机制（`embed_tokens_per_layer`、逐层 lookup、context projection 等）。  
   - 只分析常规 `embed_tokens` 与 tied `lm_head`，**不能覆盖**模型实际使用的嵌入路径。  
   - 子型号差异：E2B/E4B（PLE 为主）、26B-A4B（+ MoE）、31B（dense 但仍属 Gemma 4 新架构族）。

3. **Full-vocab 仿射 R² 显著低于主流**  
   主流 Qwen/Llama/DeepSeek V3 的 `R²_E` 多在 **0.99+**；Gemma-4 四对明显偏低：

| Pair | GCorr E (eucl) | **E_R2** | E_Δ rank95/h | A−I rank95/h |
|------|----------------|----------|--------------|--------------|
| Gemma-4-E2B | 0.676 | **0.668** | 0.893 | 0.542 |
| Gemma-4-E4B | 0.531 | **0.725** | 0.915 | 0.547 |
| Gemma-4-26B-A4B | 0.441 | **0.707** | 0.908 | 0.699 |
| Gemma-4-31B | 0.571 | **0.776** | 0.903 | 0.383 |

（来源：`ijcai_clean/results/_analysis/anomaly_group_base_instruct.csv`）

4. **BI 行范数漂移远超常态**  
   常态 tied BI：\|Δ mean%\| 多在 **±3%** 以内。Gemma-4：

| Pair | ΔE (=ΔU) |
|------|----------|
| Gemma-4-E2B | **+19.4%** |
| Gemma-4-E4B | **+20.6%** |
| Gemma-4-26B-A4B | **+51.6%** |
| Gemma-4-31B | **+11.8%** |

（来源：`analysis_eu_geometry/results/BI_ROW_NORMS_SUMMARY.md`）

### 1.2 行范数速查（E mean）

| 模型 | mean | var 量级 |
|------|------|----------|
| Gemma-4-E2B Base | 1.000000 | 10⁻⁹ |
| Gemma-4-E2B-Instruct | 1.194 | 10⁻² |
| Gemma-4-E4B Base | 1.000000 | 10⁻⁹ |
| Gemma-4-E4B-Instruct | 1.206 | 10⁻² |
| Gemma-4-26B-A4B Base | 0.999902 | 10⁻⁹ |
| Gemma-4-26B-A4B-Instruct | 1.515 | 10⁻² |
| Gemma-4-31B Base | 0.999800 | 10⁻⁷ |
| Gemma-4-31B-Instruct | 1.118 | 10⁻² |

### 1.3 静态谱（单 checkpoint，非 BI 分解）

Gemma-4 Instruct 相对 Base：**rank1(c)** 在 instruct 侧可远高于 base（径向+方向混合）；Base 的 **rank1 ≈ rank1(c)**（因已 near-unit-sphere）。

示例（`ALL_MODELS_EU_FEATURES_SUMMARY.md`）：

| 模型 | M | rank1 | rank1(c) |
|------|---|-------|----------|
| Gemma-4-26B-A4B Base | E | 0.070 | 0.026 |
| Gemma-4-26B-A4B-Instruct | E | 0.090 | **0.011** |
| Gemma-4-E2B-Instruct | E | 0.154 | **0.023** |

### 1.4 排错结论（怎么用 / 不用）

- **不进入** BI 仿射主组（`__tep` 中 `is_anomaly`: `model_a.startswith("Gemma-4-")`）。  
- 若必须引用 Gemma-4 BI：  
  - 距离用 **cosine** 或 **row-normalized Euclidean**；  
  - raw Euclidean / raw 行范数对比需标注 **gauge 影响**；  
  - 不要与 Qwen/Llama 的 P/D、R²、GCorr 直接同一叙事。  
- HF forward 里的 `embed_scale = sqrt(hidden_size)` 是 **runtime activation scale**，**不解释** checkpoint E 为何 ≈1。

---

## 二、DeepSeek-V4（Flash / Pro）

### 2.1 与 BI 主线的关系

- **没有** Base–Instruct 配对（不在 `configs/base_instruct_pairs.yaml` 的 35 对内）。  
- 出现在 `analysis_eu_geometry` 的 **other_models** 静态 audit，**不参与** Task6 / `__tep` 的 BI 仿射 sweep。

### 2.2 为何裸 U 不宜当传统 lm_head 读

输出路径（简化）：

```text
传统:  logits = U · RMSNorm(h)

V4:   H = [h1,h2,h3,h4]
      y = Σ alpha_i(H) · hi      # hc_head，权重依赖输入
      z = RMSNorm(y)
      logits = head.weight · z
```

- 裸 `head.weight` 只是 **完整输出映射的一层**；`hc_head` 是输入相关的动态混合，**不能**吸收成固定矩阵。  
- Forward：`embed.weight → HC states`；U 面对的是 **合并+归一化后** 的 `z`，职责与传统 untied lm_head 不同。

### 2.3 静态 E/U 谱：与 V3 相反的 mean-shift 模式

常规模型（untied）：往往 **U 未中心化 rank1 > E**。  
V4 例外：**E 的 μ/mean(row) 更大**，中心化后 E/U 接近：

| 模型 | M | μ/mean | rank1 | rank1(c) |
|------|---|--------|-------|----------|
| DeepSeek-V4-Flash | E | 0.237 | 0.055 | 0.021 |
| DeepSeek-V4-Flash | U | 0.110 | 0.028 | 0.019 |
| DeepSeek-V4-Pro | E | 0.180 | 0.035 | 0.019 |
| DeepSeek-V4-Pro | U | 0.108 | 0.027 | 0.019 |

解读：更像 **E 承担 HC 初始化公共轴**，而非 U 谱「更各向同性」。**不能**用 V4 裸 U 去支撑「untied 普遍 U rank1 > E」类结论。

Task5 子采样仿射（无 BI pair，仅作参考）：Flash / Pro 的 subsampled `R²_E` ~ **0.42 / 0.49**，亦偏离 V3 BI 的近 1  regime。

### 2.4 排错结论

- V4 **不纳入** BI 主分析（本无 pair）。  
- 静态结果 **仅作架构例外档案**；分析真实 output geometry 需 **hc_head + RMSNorm + head.weight** 或局部 `∂logits/∂H`，当前仓库**未**对有效输出头做全词表级分解。  
- 讨论 untied E/U 规律时，应 **剔除 V4**，或与 V3/R1 分开表述。

---

## 三、主分析组边界（便于交叉引用）

```text
35 对 Base–Instruct（configs/base_instruct_pairs.yaml）
  └─ 排除 Gemma-3-1B、Gemma-4 四对 → 主组 n = 26
       用于 __tep/affine INSIGHTS、W-rank、P/D 等

DeepSeek-V4：不在 35 对内 → 仅静态 audit / 架构说明
DeepSeek-V3 / V3.1：有 BI pair，R² 与行范数行为正常，留在主组
```

---

## 四、一句话

- **Gemma-4**：有 BI pair，但 **gauge + PLE/新架构 + 低 R² + 大行范数漂移** → **排除出 BI 主结论**。  
- **DeepSeek-V4**：**无 BI pair**，且 **裸 U 非完整输出头** → **不参与 BI 叙事**，静态谱只作例外记录。
