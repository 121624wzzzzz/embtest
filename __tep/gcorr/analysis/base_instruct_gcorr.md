# Task1：Base-Instruct GCorr 分析

## 实验设定

| 项目 | 值 |
|------|-----|
| 配置 | `configs/base_instruct_pairs.yaml` |
| 输出 | `cross_model_geometry/results/task1_base_instruct/summary.csv` |
| Pair 数 | **35** |
| 采样 | `n_tokens=20000`, `n_pairs=5e6`, `n_bootstrap=100` |
| 对齐 | 同 vocab → token id 对齐（`align_mode=id`） |

## 总体分布（数据事实）

原始 `summary.csv` 有35行；本文主聚合固定使用排除5个异常 pair 后的 BI-clean 30。分层见 [`../../../docs/分析口径与特殊案例.md`](../../../docs/分析口径与特殊案例.md)。

| 指标 | 均值 | 中位数 | 最小 | 最大 |
|------|------|--------|------|------|
| `gcorr_E_cos_mean` | 0.973 | **0.998** | 0.777 | 1.000 |
| `gcorr_E_euc_mean` | 0.932 | 0.999 | 0.441 | 1.000 |
| `gcorr_U_cos_mean` | 0.973 | 0.998 | 0.777 | 1.000 |

**证据**：`task1_base_instruct/summary.csv`，全列算术均值/中位数/min/max。

## 按系列分组（`gcorr_E_cos_mean`）

| 系列 | n | 均值 | 最小 |
|------|---|------|------|
| DeepSeek | 2 | 1.000 | 1.000 |
| Qwen2.5 | 7 | 0.999 | 0.998 |
| Qwen3.5 | 5 | 0.999 | 0.998 |
| Qwen3 | 6 | 0.993 | 0.987 |
| Llama-3.1 | 2 | 0.998 | 0.998 |
| Llama-3.2 | 2 | 0.984 | 0.981 |
| Gemma-2 | 3 | 0.999 | 0.998 |
| Gemma-3 | 4 | 0.934 | **0.777** |
| Gemma-4 | 4 | 0.849 | **0.788** |

**证据**：由 `model_a` 前缀映射 `series` 后对当前 `task1_base_instruct/summary.csv` 的 `gcorr_E_cos_mean` 分组求均值。

## E vs U：tied 与 untied

- **13/35** pair 的 `actual_tied_a` 或 `actual_tied_b` 为 `False`（典型：Qwen3-8B/14B/30B-A3B Base↔Instruct、Qwen2.5 大模型、Llama-3.1、DeepSeek V3/V3.1）。
- 全体 E-U cos 差值 `gcorr_E_cos_mean - gcorr_U_cos_mean` 的均值 ≈ **-0.0006**（几乎对称）；untied 对上 U 可略高于或低于 E，无统一“IT 只改 U”模式。
- **数据事实**：Task1 summary 列 `actual_tied_a/b`、`gcorr_E_cos_mean`、`gcorr_U_cos_mean`。

## 最低 GCorr 的 5 对（尾部）

| model_a | model_b | cos | euc |
|---------|---------|-----|-----|
| Gemma-3-1B | Gemma-3-1B-Instruct | 0.777 | 0.710 |
| Gemma-4-E4B | Gemma-4-E4B-Instruct | 0.788 | 0.531 |
| Gemma-4-26B-A4B | …-Instruct | 0.812 | 0.441 |
| Gemma-4-E2B | …-Instruct | 0.847 | 0.676 |
| Gemma-4-31B | …-Instruct | 0.951 | 0.571 |

**解读（假设）**：cos 仍 >0.77 的 Gemma-4 在“方向几何”上部分保留，但 **euc 已崩溃**（0.44–0.68），说明距离结构已严重改写；与 Task6 低 `E_R2` 一致。

## 与 Task6 的衔接

- 35 对全部进入当前 Task6 full-vocab 原始结果；`__tep` 的联表和结论只使用 BI-clean 30。
- **BI-clean**（去掉 Gemma-3-1B + Gemma-4 共 5 对，n=30）在 Task1 上 cos 中位数仍 ≈0.998，几乎不受尾部影响。

## 小结

> **数据事实**：除 Gemma-3-1B / Gemma-4 外，Base-Instruct 的 token 几何（GCorr）在 cos/euc 上均接近 1。  
> **解释假设**：Gemma 异常对不代表 IT 的一般行为，应分族报告。
