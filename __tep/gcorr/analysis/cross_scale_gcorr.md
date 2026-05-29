# Task3：跨系列、跨规模桶 GCorr 分析

## 实验设定

| 项目 | 值 |
|------|-----|
| 配置 | `cross_scale_groups.yaml` + `model_series.yaml` |
| 输出 | `task3_cross_scale_groups/summary.csv`（**176** pairs） |
| 规模桶 | `le_4b`、`gt_4b_lt_27b`、`ge_27b` |
| 规则 | 跨系列，跳过同系列（避免与 Task2 重复） |

## 总体分布

| 指标 | 均值 | 中位数 | 最小 |
|------|------|--------|------|
| `gcorr_E_cos_mean` | 0.481 | 0.438 | 0.140 |
| `gcorr_E_euc_mean` | 0.361 | 0.330 | **-0.323** |

**证据**：`task3_cross_scale_groups/summary.csv` 全表聚合。

## 负 GCorr（euclidean）

- **`gcorr_E_euc_mean < 0` 的 pair：23 个**（占 13.1%）。
- 完整列表：`results/_analysis/task3_negative_gcorr_pairs.csv`（列 `gcorr_E_euc`, `gcorr_E_cos`, `scale_group`, `series_a/b`）。

### 负 euc 的模式（数据事实）

| 特征 | 观察 |
|------|------|
| 系列组合 | 多为 **Qwen × Gemma**（2.5/3 vs Gemma-2/3） |
| 规模桶 | 集中在 `gt_4b_lt_27b` 与 `ge_27b` |
| cos 符号 | 多数仍 **>0.24**（非“完全反相关”） |

典型极低 euc 行（来自 `_analysis/task3_negative_gcorr_pairs.csv`）：

| gcorr_E_euc | gcorr_E_cos | pair 示例 |
|-------------|-------------|-----------|
| -0.323 | 0.288 | Gemma-2-9B-IT × Qwen3.5-9B-IT |
| -0.280 | 0.285 | Qwen3-8B × Gemma-2-9B-IT |
| -0.142 | 0.458 | Llama-3.1-70B-IT × Gemma-4-31B-IT |

**解释假设**：不同族的 **距离标度与 anisotropy** 不一致时，采样 token-pair 的欧氏距离向量可在 Pearson 意义下反相关，即使 cos 仍为正。这不等于“embedding 无关”，而是 **metric 不可比**。

## 与 Task2 的关系

- Task3 中位数 cos（0.438）与 Task2（0.360）同量级，均远低于 Task1（0.998）。
- Task3 额外引入 **跨 tokenizer / 跨 vocab**（`align_mode=string` 占多数），进一步压低可比性。

## 对主线的含义

| 问题 | Task3 能否回答 |
|------|----------------|
| Base-Instruct 是否保持几何？ | **否**（非 Base-Instruct 设计） |
| 不同公司模型是否同构？ | 仅弱证据：cos 通常 0.3–0.5 |
| 哪些族组合最易“负相关”？ | **是**：Qwen–Gemma 桶需单独统计 |

## scale_group × 系列列联（Pass 迭代 R3）

**证据表**：[`tables/gcorr_task3_series_scale_crosstab.csv`](../tables/gcorr_task3_series_scale_crosstab.csv)（63 格，由 `pair_plan.csv` ⨝ `summary.csv` 聚合）。

| 观察 | 数据事实 |
|------|----------|
| 负 euc 集中桶 | **91%**（21/23）在 `gt_4b_lt_27b`+`ge_27b`；`le_4b` 仅 2/108 对负 euc |
| 负 euc 族组合 | 多格 **gemma×qwen**（2.5/3）在 `ge_27b` 为 **2/2 全负**；`gemma2×llama` 在 `le_4b` 亦 2/2 |
| 高 cos 例外 | 同族跨代如 Qwen3×Qwen2.5 在 `le_4b` 可达 cos **0.85–0.90** |

**解释假设**：中大桶 + 跨族（尤其 Gemma×Qwen）距离标度不可比 → euc GCorr 可为负而 cos 仍正。

## 建议（后续）

1. 与 Task4 MoE 结果对照：MoE 跨族低 gcorr 是否与 Task3 同族组合一致。
2. 可选：对列联表中 neg_euc=2/2 的格做 bootstrap（源 CSV 已有 `bootstrap_results.csv`）。
