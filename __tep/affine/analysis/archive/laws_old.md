# 规律深化：早期跨任务定律

> 本文件保留早期从 Task1/5/6 归纳出的跨任务规律，适合作为背景和边界材料。当前 affine 论文主线已经更新为 **E/U + tied/untied + update-scale + LoRA budget**，请优先阅读 [`../../INSIGHTS.md`](../../INSIGHTS.md) 与 [`../tasks/task6_pred_delta_probe.md`](../tasks/task6_pred_delta_probe.md)。  
> 数值经 `python3` 从 `ijcai_clean/results/**` CSV 复核（2026-05-19）；机器可读索引见 [`../../../data/pattern_checks.json`](../../../data/pattern_checks.json)。

**主分析组** = 31 对 Base-Instruct 中排除 `Gemma-3-1B` 与全部 `Gemma-4-*`（n=26）。**异常组** = 上述 5 对。

---

## 1. Base-Instruct 主规律 vs 异常组

### 1.1 现象
同 vocab、同 hidden 的 Base→Instruct 对，Qwen/Llama/Gemma-2/3(大模型) 上 GCorr 与 full-vocab 仿射 R² 双高；Gemma-3-1B 与 Gemma-4 全系双低。

### 1.2 证据

| 子集 | n | `gcorr_E_cos` 均值 | `E_R2` 均值 | `E_R2` 中位数 |
|------|---|-------------------|-------------|---------------|
| 全体 | 31 | 0.970 | 0.936 | 0.996 |
| **主组** | 26 | **0.995** | **0.991** | **0.997** |
| **异常** | 5 | 0.835 | 0.650 | 0.707 |

- 来源：Task1 `summary.csv`；Task6 `summary_pair_base_instruct_full_vocab.csv`。

### 1.3 边界条件
- 需 **id 对齐、同 hidden_dim**；跨模型/跨族不适用。
- Gemma-3-4B/12B/27B GCorr 0.98+、`E_R2` 0.96–0.97，属主组但仿射略低于 Qwen 同级——**非**极端异常。
- 主组内部的低一档与 `actual_tied=True` 明显同现：untied 9 对 `E_R2` 均值 **0.9984**、min **0.9956**；tied 17 对均值 **0.9878**、min **0.9579**。这更像 tied/untied 架构分层，而不只是系列差异。

### 1.4 反例
- `Gemma-3-1B`：cos **0.777**，`E_R2` **0.375**，`E_rel_A_minus_I/I` **0.911**。
- `Gemma-4-E2B`：cos 0.847，euc 0.676，`E_R2` **0.668**（cos 仍 >0.77 但 euc/R² 已崩）。

### 1.5 Tied/untied 分层（主组 n=26）

| actual_tied | n | `E_R2` 均值 | 中位数 | min | max |
|-------------|---|-------------|--------|-----|-----|
| False | 9 | **0.9984** | 0.9986 | 0.9956 | 0.9999 |
| True | 17 | 0.9878 | 0.9911 | 0.9579 | 0.9997 |

低 `E_R2` 排序前十全部为 tied；但 tied 不是充分条件，Gemma-2、Qwen2.5/3.5 的部分 tied pair 仍接近饱和。写作时宜表述为「主组内部低一档主要由 tied 架构解释/调节」，不要写成 tied 必然低 R²。

---

## 2. GCorr–R² 关系：双高/双低，而非连续预测

### 2.1 现象
31 对 BI 上，`gcorr_E_cos_mean` 与 `E_R2` Pearson **r = 0.932**；但这个数主要反映**主组双高、异常组双低**的分离。主组内二者都接近 1，动态范围很窄，Pearson 不宜写成仿射质量的连续预测证据。`gcorr_E_euc_mean` 与 `E_R2` 的全体相关为 **r = 0.808**，同样只作诊断参考。

### 2.2 证据
- 合并键：`model_a`（Task1 ⨝ Task6）。
- 主组内 cos 中位数 ≈0.998、R² 中位数 ≈0.997；异常组 cos 中位数 ≈0.812、R² ≈0.707。
- 主组内 cos 均值 **0.995**、`E_R2` 均值 **0.991**，已经有明显 ceiling effect；因此更稳的表述是「同向失效/同向饱和」，不是「cos 可预测 R²」。

### 2.3 边界条件
- 该诊断仅在 **BI 同 id** 上有意义；Task2–4 跨模型未算 full-vocab R²，不可外推。
- **Gemma-4-31B**：cos **0.951** 但 `E_R2` **0.776**——高 cos 不保证高 R²。

### 2.4 反例
- Gemma-4 四对：cos 0.79–0.95，euc 0.44–0.68，R² 0.67–0.78——**cos 与 R² 可分离**。

---

## 3. E vs U：BI 上同步，模型内部不同步

### 3.1 现象
Base-Instruct 上 E/U GCorr **几乎相同**（tied 对完全相同）；9 对 untied 大模型 E cos 均值仍 **0.998**。模型 **内部** E→U 仿射则普遍差。

### 3.2 证据

| 口径 | 指标 | 值 |
|------|------|-----|
| BI 31 对 | mean(E cos) − mean(U cos) | **−0.0003** |
| BI 31 对 | max \|E cos − U cos\| | **0.0043** |
| Intra E→U | `R2_EU < 0.5` | **47/92** |

- 来源：Task1 `summary.csv`；Task5 `summary_intra_EU.csv`。

### 3.3 边界条件
- 「E≈U 同步变化」仅指 **BI 扰动**；untied 架构下 E/U 本就不追求全局线性互映。

### 3.4 反例
- `Qwen2.5-7B` intra `R2_EU` **0.206**——BI 上 E/U GCorr 仍 ≈1.0。

---

## 4. A-I 比 E_Δ 更低秩、谱更集中（按族）

### 4.1 现象
主组 naive 差分 `E_Δ = E_instruct − E_base` 的 rank95/h 约为 **0.77h**，最优仿射 `A−I` 仅 **0.43h**；5% hidden 维预算下 A-I 能量中位数 **0.628** vs E_Δ **0.397**。

### 4.2 证据（主组 n=26，Task6）

| 族 | n | E_Δ rank95/h | A-I rank95/h | E_Δ e@5%h 中位 | A-I e@5%h 中位 | eff/h 线性 R² |
|----|---|-------------|-------------|---------------|---------------|--------------|
| Qwen2.5 | 7 | 0.624 | 0.310 | 0.407 | 0.505 | **0.915** |
| Qwen3 | 9 | 0.799 | 0.513 | 0.284 | 0.669 | 0.367 |
| Qwen3.5 | 4 | 0.856 | 0.505 | 0.273 | 0.659 | **0.980** |
| Llama | 4 | 0.845 | 0.610 | 0.326 | 0.411 | **1.000** |
| Gemma-2 | 3 | 0.782 | 0.095 | 0.570 | **0.978** | 0.823 |
| Gemma-3* | 3 | 0.903 | 0.525 | 0.203 | 0.449 | 0.819 |

\*不含 Gemma-3-1B。

- 全主组 eff/h 回归：`(A-I)/h ≈ −0.012 + 0.515×(E_Δ)/h`，R² **0.553**。
- 来源：Task6 列 `E_delta_*`、`A_delta_*`、`hidden_dim_a`。

### 4.3 边界条件
- **族内**分开拟合（Qwen2.5/3.5 R²>0.91）；16 Qwen 合并 R² 仅 0.43——**跨亚族混合稀释线性律**。
- Gemma-2 的 A-I rank95/h 极低（0.095）因 A 近 I，与 Qwen 机制不同但 R² 仍高。

### 4.4 反例（异常组谱形）
- 异常 5 对：E_Δ rank95/h **0.897** 但 e@5%h 中位仅 **0.203**（主组 0.397）——**高 rank95 + 低能量集中度**，奇异值分散，不宜纳入主回归。

---

## 5. Task2：hidden 匹配决定 GCorr 量级

### 5.1 现象
同系列内比较，**hidden 维相同** vs **不同** 的 cos 均值差约 **0.50**（0.873 vs 0.376），远大于 IT 扰动效应（Task1 中位 0.998）。

### 5.2 证据

| 子集 | n | `gcorr_E_cos` 均值 |
|------|---|-------------------|
| 同 hidden | 24 | **0.873** |
| 异 hidden | 86 | **0.376** |
| 异 hidden 且 cos<0.5 | 63 | — |

- 来源：`task2_model_series/summary.csv`；`_analysis/task2_hidden_dim_mismatch_pairs.csv`。

### 5.3 边界条件
- Task2 不能论证 IT 稳定性；仅说明 **规模/维度假设** 下几何可比。
- 同 hidden 24 对仍含 instruct×instruct 跨代，均值 0.873 < BI 0.995。

### 5.4 反例
- DeepSeek 系列 25 对均值 0.790 但中位 **0.999**——**双峰分布**，均值掩盖同维子集。

---

## 6. Task3：规模桶与负 euc GCorr

### 6.1 现象
176 对跨系列：`le_4b` 桶 cos 均值 **0.567**，`gt_4b_lt_27b` / `ge_27b` 约 **0.34–0.35**。23 对 `gcorr_E_euc_mean < 0`，cos 仍为正（0.14–0.64）。

### 6.2 证据

| scale_group | n | E cos mean |
|-------------|---|------------|
| le_4b | 108 | **0.567** |
| gt_4b_lt_27b | 34 | 0.345 |
| ge_27b | 34 | 0.343 |

- 负 euc 分布（23 对）：`le_4b` 2、`gt_4b_lt_27b` 10、`ge_27b` 11；**91% 在中大桶**。
- 列联细化：**gemma×qwen** 在 `ge_27b` 多格 **2/2 全负**（见 [`../../../gcorr/tables/gcorr_task3_series_scale_crosstab.csv`](../../../gcorr/tables/gcorr_task3_series_scale_crosstab.csv)）。
- 最强负值：`Gemma-2-9B-IT × Qwen3.5-9B-IT`，euc **−0.323**，cos **0.288**。
- 来源：`task3_cross_scale_groups/summary.csv`；`_analysis/task3_negative_gcorr_pairs.csv`；`pair_plan.csv`。

### 6.3 边界条件（谨慎解释）
- 负 euc 反映 **跨族距离向量 Pearson 相关为负**，不等同 embedding 无关；cos 为正说明方向结构仍有弱相关。
- 涉及 Gemma 23/23 对、Qwen 19/23 对——**族际 + 规模桶** 混杂，非 BI 失败。

### 6.4 反例
- 同族跨代最高：`Qwen3-4B × Qwen2.5-3B-Instruct` cos **0.898**——族内可比性可很高。
- Task3 summary **无负 cos**——负值现象 **仅出现在 euc 度量**。

---

## 7. Task5：为何仅 task1 来源 R² 高

### 7.0 Intra E→U（92 模型，与 pair 分层正交）

| actual_tied | n | R2_EU 均值 | R2&lt;0.5 |
|-------------|---|------------|----------|
| True | 44 | **1.000** | 0 |
| False | 48 | **0.348** | 47 |

- 来源：`summary_intra_EU.csv`；[`tables/affine_task5_intra_by_tied.csv`](../../tables/archive/affine_task5_intra_by_tied.csv)。
- **含义**：BI 上 E/U GCorr 同步（§3）≠ 模型内部 E→U 可仿射；untied 架构下后者普遍失效。

### 7.1 现象
338 对全体 `R2_E` 均值 **0.429**；按 `source_tasks` 分层后，**仅 task1 子集** 均值 **0.942**，与 Task6 一致。

### 7.2 证据

| source_tasks | n | R2_E mean |
|--------------|---|-----------|
| task1_base_instruct | 31 | **0.942** |
| task2_model_series | 110 | 0.411 |
| task3_cross_scale_groups | 176 | 0.358 |
| task4_moe_cross_family | 21 | 0.361 |

- Task1 子集与 Task6 `E_R2`：**r = 0.999**（n=31），max \|Δ\| **0.049**（`Gemma-4-31B`）。
- 来源：`task5_affine_subsampled/summary_pair.csv`。

### 7.3 边界条件
- Task5 用 **24k 行子采样**；对 BI 足够（与 full-vocab 几乎一一对应）。
- Task5 全体均值 **不能** 代表 BI 仿射质量——pair 来源以跨模型为主（307/338）。

### 7.4 反例
- Task2/3/4 来源即使 GCorr 偶尔 >0.9（如 Task4 Qwen3.5×3.6 cos 0.953），Task5 `R2_E` 仍随 **跨模型** 来源整体偏低——**同族相邻代际 MoE 是跨任务例外，非 BI 规律**。

---

## 8. Task6 vs Task5：子采样验证 full-vocab

### 8.1 现象
同一 31 对 BI 上，Task5 子采样 R² 与 Task6 full-vocab R² 几乎重合；Task6 额外提供 A-I SVD 诊断。

### 8.2 证据
- Pearson(Task5 `R2_E`, Task6 `E_R2`) = **0.999**；Qwen/Llama 多数 \|Δ\| < 0.01。
- Task6 主组 `E_R2` 0.991 vs Task5 task1 0.942 的差异主要来自 **异常 5 对** 在 full-vocab 下拟合更严（Gemma-4-31B Δ=−0.049）。

### 8.3 边界条件
- Task6 `n_fit = vocab_size`；Task5 `max_fit_rows=24000`——BI 上二者等价，跨模型不适用。

### 8.4 反例
- `Gemma-3-1B`：Task5 `R2_E` 0.404，Task6 0.375——双低，方向一致。

---

## 9. GCorr vs 仿射 vs SVD：三角定位

| 方法 | 度量对象 | BI 主组 | 跨模型 |
|------|----------|---------|--------|
| **GCorr** | token-pair 几何相关 | cos ≈0.995 | cos ≈0.38–0.57 |
| **仿射 R²** | 全局 `Y≈XA+b` | **0.991** | **0.36–0.41** |
| **SVD** | 差分谱结构 | A-I 低秩/高能 | 未系统计算 |

**可写入论文的一句话**（有数据支撑）：

> 对 26 个非异常 Base-Instruct pair，instruct 相对 base 的 embedding 变化可由接近单位阵的全局仿射解释（median R²≈0.997），且 A−I 的谱能量显著集中于低维子空间；GCorr 与 R² 在主组呈双高、在 Gemma-3-1B/Gemma-4 中同步下降，但 Pearson 相关不应作为核心证据。

---

## 10. 缺失数据说明

| 主题 | 状态 |
|------|------|
| Gemma-3-1B row-level cosine 重算 | 未纳入；`_analysis/gemma_series_row_norms.csv` 为历史审计 |
| Task2/3 跨模型 full-vocab R² | 未算；仅 Task5 子采样 R² |
| energy@p%h 曲线（p≠5%） | 仅单点 rank95 与 e@5%h |
| Pass2 事实复核 | 已完成；Gemma cos 0.935→0.921 等已修正 |
