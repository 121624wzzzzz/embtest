# Task2：同系列内 GCorr 分析

## 实验设定

| 项目 | 值 |
|------|-----|
| 配置 | `configs/model_series.yaml` |
| 输出 | `task2_model_series/summary.csv`（**110** pairs） |
| Pair 计划 | `pair_plan.csv`（系列内 C(n,2)） |

## 总体：跨规模几何相似性弱

| 指标 | 均值 | 中位数 | 最小 | 最大 |
|------|------|--------|------|------|
| `gcorr_E_cos_mean` | 0.484 | 0.360 | **0.057** | 1.000 |

**证据**：`task2_model_series/summary.csv` 全表聚合。

## Hidden dimension 不匹配是核心混杂

| 统计量 | 值 |
|--------|-----|
| `hidden_dim_a ≠ hidden_dim_b` 的 pair | **86 / 110** |
| 其中 `gcorr_E_cos_mean < 0.5` | **63** |
| `gcorr_E_cos_mean > 0.95` | **19** |

**证据**：`summary.csv` 列 `hidden_dim_a`、`hidden_dim_b`、`gcorr_E_cos_mean` 交叉计数（`computed_stats.json`）。

**解释假设**：GCorr 对绝对 hidden 空间维度和尺度敏感；不同规模模型即使用同一 tokenizer，直接比 E/E 几何也会偏低。同 hidden 的系列内对比更有信息量。

## 按系列（`model_a` 前缀）的 cos 均值

| 系列 | n | cos 均值 | 备注 |
|------|---|----------|------|
| DeepSeek | 25 | 0.790 | 中位数 0.999，分布极双峰 |
| Gemma-4 | 6 | 0.751 | 系列内较一致 |
| Gemma-3 | 6 | 0.656 | |
| GLM / MiniMax | 各 1 | >0.99 | 样本少 |
| Qwen3.5 | 13 | 0.441 | |
| Gemma-2 | 3 | 0.486 | |
| Qwen2.5 | 21 | 0.295 | |
| Qwen3 | 28 | 0.282 | |
| Llama-3.1 | 5 | **0.176** | 跨代际 instruct 对比 |

## 最低 cos 的典型 pair（跨规模）

| model_a | model_b | cos |
|---------|---------|-----|
| Llama-3.1-8B-Instruct | Llama-3.2-3B-Instruct | 0.057 |
| Qwen2.5-0.5B-Instruct | Qwen2.5-7B-Instruct | 0.059 |
| Qwen3-4B | Qwen3-14B | 0.073 |
| Llama-3.1-8B-Instruct | Llama-3.2-1B-Instruct | 0.074 |

**证据**：`summary.csv` 排序 `gcorr_E_cos_mean` 升序前 8 行。

## 与 Task1 的对比

| 对比类型 | Task1（Base↔IT 同规模） | Task2（同系列不同规模） |
|----------|-------------------------|-------------------------|
| cos 中位数 | 0.998 | 0.360 |
| 主要因素 | IT 扰动小（除 Gemma 异常） | hidden / 训练差异大 |

**结论（数据事实）**：同系列 **不同规模** 的 embedding 几何 **不** 像 Base-Instruct 那样保持；Task2 不能用于论证 IT 稳定性，只适合讨论“规模/架构漂移”。

## `_analysis` 补充

- `task2_hidden_dim_mismatch_pairs.csv`：预筛 hidden 不匹配 pair，便于 Pass 2 做 matched 子集分析。
