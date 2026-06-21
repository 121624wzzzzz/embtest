# 异常专题：Gemma、低 R²、负 GCorr

## 异常分层（分析框架）

| 层级 | 模型/现象 | Task1 cos | Task6 E_R2 | 建议处理 |
|------|-----------|-----------|------------|----------|
| A | Gemma-3-1B | 0.777 | **0.375** | 排除；独立讨论 |
| B | Gemma-4 全系（4） | 0.79–0.95 | 0.67–0.78 | 排除；架构附录 |
| C | Gemma-3 4B/12B/27B | 0.98+ | 0.96–0.97 | **纳入 BI-clean** |
| D | Task3 负 euc（23 pair） | cos 常仍 >0.2 | 未算仿射 | 跨族 metric 问题 |
| E | Task4 DeepSeek×MiniMax 等 | euc<0 | — | 个案脚注 |

**口径**：前 5 个异常模型均排除；当前统计只使用 BI-clean 30 对。证据来自 `../tables/gcorr_task1_base_instruct_metrics.csv`（30 行）及上游 `cross_model_geometry/results/task1_base_instruct/summary.csv`、Task6 汇总。

---

## Gemma-3-1B：极端离群

### 数据事实

| 指标 | Gemma-3-1B | Gemma-3-4B（对照） | Qwen2.5-0.5B |
|------|------------|-------------------|--------------|
| `gcorr_E_cos_mean` | 0.777 | 0.990 | 0.994 |
| `gcorr_E_euc_mean` | 0.710 | 0.998 | 0.994 |
| `E_R2` | **0.375** | 0.958 | 0.990 |
| `E_rel_A_minus_I_over_I` | **0.911** | 0.053 | 0.057 |
| `E_identity_cosine` | **0.421** | 0.999 | 0.999 |
| `E_delta rank95/h` | 0.865 | 0.910 | — |

**来源**：`task1_base_instruct/summary.csv`、`task6/.../summary_pair_base_instruct_full_vocab.csv`。

### 与系列内大模型对比（解释假设）

`analysis.md` 记录：Gemma-3-4B/12B/27B 的 row cosine 仍 ≈0.98，而 1B 仅 ≈0.28（本地 row-level 审计，非 Task1 CSV）。

**工作假设**（非官方定论）：

1. **1B 为 text-only `gemma3_text` 分支**，4B+ 为 VLM conditional generation — 训练目标不同。  
2. **embedding 参数占比高**（~30% vs 4B ~17%）— post-training 更易改动 E。  
3. **强 distillation / IT** 导致小容量模型整体重写输入空间。

**已排除（分析.md + 数据）**：tokenizer id 大面积错位；特殊 token 主导（skip 128/1024 后 cosine 仍 ~0.28）。

---

## Gemma-4：系统性中等偏低

### 数据事实（4 pair 汇总）

| 模型 | E_R2 | gcorr_E_euc | gcorr_E_cos | A-I rank95/h |
|------|------|-------------|-------------|--------------|
| E2B | 0.668 | 0.676 | 0.847 | 0.542 |
| E4B | 0.725 | 0.531 | 0.788 | 0.547 |
| 26B-A4B | 0.707 | 0.441 | 0.812 | 0.699 |
| 31B | 0.776 | 0.571 | 0.951 | 0.383 |

**来源**：上游 Task1、Task6 汇总；这些排除案例不进入当前 30 对聚合。

### 架构假设：PLE（Per-Layer Embeddings）

**解释假设**（见 `analysis.md` 引用的 HF/Transformers 讨论）：

- Gemma 4 实际推理可能使用 `embed_tokens_per_layer`、逐层 lookup 与 context projection。  
- 实验仅抽取 **单层** `embed_tokens` 与 lm_head — **不足以** 代表完整输入嵌入机制。  
- E2B/E4B 与 26B-A4B（MoE）/31B（dense）应 **分型号** 叙述，不宜合并为单点。

### A 诊断上的矛盾模式

- 部分型号 `E_identity_cosine` 仍 >0.97（仿射 A 近 I），但 **E_R2<0.78** — 说明 **残差大但方向仍像缩放**。  
- `E_delta rank95/h` 可高达 **0.89–0.91**，与 Qwen 主组 (~0.82) 相当甚至更高 → **“高秩差分 + 低 R²”** 并存，支持“观测矩阵不是完整机制”的假设，而非单纯拟合失败。

---

## Gemma-2：对照组（正常）

| 指标 | 范围 |
|------|------|
| Task1 cos | 0.998–1.000 |
| Task6 E_R2 | 0.991–0.999 |

**结论（数据事实）**：Gemma **品牌** 并非异常来源；异常集中在 **Gemma-3-1B** 与 **Gemma-4** 新架构。

---

## Task3：负 `gcorr_E_euc`（23 pairs）

### 数据事实

- 列表：`results/_analysis/task3_negative_gcorr_pairs.csv`（**23** 对数据行）。  
- 最强负值：Gemma-2-9B-IT × Qwen3.5-9B-IT，`gcorr_E_euc=-0.323`，`gcorr_E_cos=0.288`。  
- 规模桶：多数在 `gt_4b_lt_27b` 与 `ge_27b`。

### 解释假设

1. **跨族距离标度不可比** → Pearson(euc 向量) 可为负。  
2. **cos 仍为正** → 方向结构仍有弱相关，与“完全无关”不同。  
3. **不等同于** Base-Instruct 失败，也不支持“IT 破坏几何”的全局命题。

### 与 Gemma-4 Base-Instruct 异常的关系

- **无直接重叠**：负 euc 主要在 **Instruct×Instruct 跨族**；Gemma-4 Base-Instruct 异常是 **同 id 对齐的 E↔E**。  
- 共同主题：涉及 Gemma 时 metric 解释需更谨慎。

---

## Task4 / MoE 脚注

- `DeepSeek-R1-0528` × `MiniMax-M2.7`：`gcorr_E_euc=-0.184`（Task4 `summary.csv`）。  
- 属 **跨族 MoE** 比较，与 Gemma Base-Instruct 机制不同，但说明 **euc GCorr 可出现负值** 需在全文中定义/限制解释范围。

---

## 当前证据文件

| 文件 | 用途 |
|------|------|
| `../tables/gcorr_task1_base_instruct_metrics.csv` | BI-clean 30 对 Task1 指标 |
| 上游 `cross_model_geometry/results/task1_base_instruct/summary.csv` | 35 对原始 Task1（含 5 个排除案例） |
| 上游 Task6 汇总 | 仿射 R² 与谱指标 |

---

## 写作建议（Pass 2）

1. **主文**：统一报告 BI-clean 30 对；不再设置 26 对主组。
2. **附录 A**：Gemma-3-1B 案例研究（表格 + row cosine 图）。  
3. **附录 B**：Gemma-4 与 PLE 局限（不强行纳入主回归）。  
4. **脚注**：Task3 负 euc 定义与占比 13%。
