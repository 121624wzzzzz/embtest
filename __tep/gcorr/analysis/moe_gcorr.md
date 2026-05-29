# Task4：MoE 跨族 GCorr 分析

## 实验设定

| 项目 | 值 |
|------|-----|
| 配置 | `moe_cross_family.yaml` + `model_series.yaml` |
| 输出 | `task4_moe_cross_family/summary.csv`（**21** pairs） |
| 对齐 | 多为 `align_mode=string`（跨 vocab） |

## 总体

21 对 MoE / dense 代表模型的跨族 GCorr；`gcorr_E_cos_mean` 范围约 **0.38–0.95**，多数落在 0.4–0.55。

**证据**：`task4_moe_cross_family/summary.csv`。

## 突出 pair

### 同族/近缘高相关

| model_a | model_b | gcorr_E_cos | gcorr_U_cos |
|---------|---------|-------------|-------------|
| Qwen3.5-35B-A3B-Instruct | Qwen3.6-35B-A3B | **0.953** | 0.994 |
| DeepSeek-V2-Lite-Chat | MiniMax-M2.7 | 0.505 | 0.481 |

**数据事实**：Qwen3.5 IT 与 Qwen3.6 的 cos/euc 均接近 1，说明 **同生态 MoE 变体** 仍共享极强几何。

### 跨族低相关 + U 分化

| model_a | model_b | gcorr_E_cos | gcorr_U_cos |
|---------|---------|-------------|-------------|
| Qwen3-30B-A3B | Gemma-4-26B-A4B-Instruct | 0.438 | **0.360** |
| Qwen3.5-35B-A3B-Instruct | Gemma-4-26B-A4B-Instruct | 0.426 | **0.310** |
| Gemma-4-26B-A4B-Instruct | MiniMax-M2.7 | 0.422 | 0.324 |

**模式**：E cos ~0.42 时，U cos 常 **更低**（0.31–0.36）— 与 Task1 中“U 与 E 同步”形成对比。

**解释假设**：MoE / 不同族模型的 **lm_head 与 embed 解耦程度** 更高，跨族比较时输出层几何更不一致。

### 负 euc（Task4 内）

| pair | gcorr_E_euc |
|------|-------------|
| DeepSeek-R1-0528 × MiniMax-M2.7 | **-0.184** |
| Gemma-4-26B-A4B-IT × MiniMax-M2.7 | 0.034 |

**证据**：`summary.csv` 末行；与 Task3 负 euc 现象同类。

## 与 Base-Instruct 主线

Task4 **不** 支持“全局 embedding 同构”命题；仅说明：

1. **同系列 MoE 迭代**（Qwen3.5↔3.6）可达 0.95+ GCorr。
2. **Qwen MoE × Gemma-4** 稳定在低 0.4 带，与 Task3 跨族一致。
3. Gemma-4 在跨族对比中 U 侧更弱，可能与 PLE / tied 声明有关（见 [`gemma_anomalies.md`](gemma_anomalies.md)）。

## skipped_models

`skipped_models.csv` 记录因缺 extract 或 token 不足跳过的模型；Pass 2 引用 pair 数时需核对 metadata。
