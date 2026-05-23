# 按 tied 状态划分的 Base-Instruct 仿射结果

来源：`summary_pair.csv`，筛选条件为 `source_tasks` 包含 `task1_base_instruct`。

拟合关系为 `Y ~= X * A + b`。`R2_E` 衡量 Base embedding 到 Instruct embedding 的仿射拟合效果；`R2_U` 衡量 Base unembedding 到 Instruct unembedding 的仿射拟合效果。`R2` 越高、相对误差越低，说明仿射关系越强。

### 全部 Base-Instruct 模型对

| 数量 | R2_E 均值 | R2_E 中位数 | R2_E 最小值 | R2_E 最大值 | R2_U 均值 | R2_U 中位数 | R2_U 最小值 | R2_U 最大值 | rel_err_E 均值 | rel_err_U 均值 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 35 | 0.9487 | 0.9965 | 0.4038 | 1.0000 | 0.9486 | 0.9951 | 0.4038 | 1.0000 | 0.1270 | 0.1285 |

### Llama Base-Instruct 模型对

| 数量 | R2_E 均值 | R2_E 中位数 | R2_E 最小值 | R2_E 最大值 | R2_U 均值 | R2_U 中位数 | R2_U 最小值 | R2_U 最大值 | rel_err_E 均值 | rel_err_U 均值 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.9891 | 0.9902 | 0.9771 | 0.9990 | 0.9887 | 0.9898 | 0.9771 | 0.9982 | 0.0833 | 0.0880 |

#### Llama 明细

| 模型 A | 模型 B | tied A/B | 来源 | 对齐 | 公共 token 数 | 拟合行数 | R2_E | rel_err_E | R2_U | rel_err_U | 备注 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `Llama-3.1-70B-Base` | `Llama-3.1-70B-Instruct` | False/False | task1_base_instruct | id | 128000 | 24000 | 0.9990 | 0.0314 | 0.9982 | 0.0423 |  |
| `Llama-3.1-8B` | `Llama-3.1-8B-Instruct` | False/False | task1_base_instruct | id | 128000 | 24000 | 0.9987 | 0.0363 | 0.9979 | 0.0441 |  |
| `Llama-3.2-3B` | `Llama-3.2-3B-Instruct` | True/True | task1_base_instruct | id | 128000 | 24000 | 0.9817 | 0.1251 | 0.9817 | 0.1251 |  |
| `Llama-3.2-1B` | `Llama-3.2-1B-Instruct` | True/True | task1_base_instruct | id | 128000 | 24000 | 0.9771 | 0.1406 | 0.9771 | 0.1406 |  |

## 按 tied 状态划分的结果

### tied / tied

| 数量 | R2_E 均值 | R2_E 中位数 | R2_E 最小值 | R2_E 最大值 | R2_U 均值 | R2_U 中位数 | R2_U 最小值 | R2_U 最大值 | rel_err_E 均值 | rel_err_U 均值 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 22 | 0.9193 | 0.9896 | 0.4038 | 0.9998 | 0.9193 | 0.9896 | 0.4038 | 0.9998 | 0.1843 | 0.1843 |

#### tied / tied 明细

| 模型 A | 模型 B | tied A/B | 来源 | 对齐 | 公共 token 数 | 拟合行数 | R2_E | rel_err_E | R2_U | rel_err_U | 备注 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `Qwen2.5-3B` | `Qwen2.5-3B-Instruct` | True/True | task1_base_instruct | id | 151643 | 24000 | 0.9998 | 0.0151 | 0.9998 | 0.0151 |  |
| `Qwen2.5-1.5B` | `Qwen2.5-1.5B-Instruct` | True/True | task1_base_instruct | id | 151643 | 24000 | 0.9997 | 0.0164 | 0.9997 | 0.0164 |  |
| `Gemma-2-2B` | `Gemma-2-2B-Instruct` | True/True | task1_base_instruct | id | 255993 | 24000 | 0.9994 | 0.0224 | 0.9994 | 0.0224 |  |
| `Gemma-2-27B` | `Gemma-2-27B-Instruct` | True/True | task1_base_instruct | id | 255993 | 24000 | 0.9980 | 0.0439 | 0.9980 | 0.0439 |  |
| `Qwen3.5-0.8B-Base` | `Qwen3.5-0.8B-Instruct` | True/True | task1_base_instruct | id | 248044 | 24000 | 0.9979 | 0.0406 | 0.9979 | 0.0406 |  |
| `Qwen3.5-4B-Base` | `Qwen3.5-4B-Instruct` | True/True | task1_base_instruct | id | 248044 | 24000 | 0.9975 | 0.0468 | 0.9975 | 0.0468 |  |
| `Qwen3.5-2B-Base` | `Qwen3.5-2B-Instruct` | True/True | task1_base_instruct | id | 248044 | 24000 | 0.9969 | 0.0504 | 0.9969 | 0.0504 |  |
| `Qwen3-1.7B-Base` | `Qwen3-1.7B` | True/True | task1_base_instruct | id | 151643 | 24000 | 0.9944 | 0.0730 | 0.9944 | 0.0730 |  |
| `Gemma-2-9B` | `Gemma-2-9B-Instruct` | True/True | task1_base_instruct | id | 255995 | 24000 | 0.9925 | 0.0762 | 0.9925 | 0.0762 |  |
| `Qwen3-4B-Base` | `Qwen3-4B` | True/True | task1_base_instruct | id | 151643 | 24000 | 0.9911 | 0.0906 | 0.9911 | 0.0906 |  |
| `Qwen2.5-0.5B` | `Qwen2.5-0.5B-Instruct` | True/True | task1_base_instruct | id | 151643 | 24000 | 0.9906 | 0.0893 | 0.9906 | 0.0893 |  |
| `Qwen3-0.6B-Base` | `Qwen3-0.6B` | True/True | task1_base_instruct | id | 151643 | 24000 | 0.9887 | 0.1006 | 0.9887 | 0.1006 |  |
| `Llama-3.2-3B` | `Llama-3.2-3B-Instruct` | True/True | task1_base_instruct | id | 128000 | 24000 | 0.9817 | 0.1251 | 0.9817 | 0.1251 |  |
| `Llama-3.2-1B` | `Llama-3.2-1B-Instruct` | True/True | task1_base_instruct | id | 128000 | 24000 | 0.9771 | 0.1406 | 0.9771 | 0.1406 |  |
| `Gemma-3-27B` | `Gemma-3-27B-Instruct` | True/True | task1_base_instruct | id | 262137 | 24000 | 0.9770 | 0.1510 | 0.9770 | 0.1510 |  |
| `Gemma-3-12B` | `Gemma-3-12B-Instruct` | True/True | task1_base_instruct | id | 262137 | 24000 | 0.9721 | 0.1655 | 0.9721 | 0.1655 |  |
| `Gemma-3-4B` | `Gemma-3-4B-Instruct` | True/True | task1_base_instruct | id | 262137 | 24000 | 0.9620 | 0.1915 | 0.9620 | 0.1915 |  |
| `Gemma-4-31B` | `Gemma-4-31B-Instruct` | True/True | task1_base_instruct | id | 262120 | 24000 | 0.8252 | 0.4099 | 0.8252 | 0.4099 | Gemma4 |
| `Gemma-4-E4B` | `Gemma-4-E4B-Instruct` | True/True | task1_base_instruct | id | 262120 | 24000 | 0.7516 | 0.4676 | 0.7516 | 0.4676 | Gemma4 |
| `Gemma-4-26B-A4B` | `Gemma-4-26B-A4B-Instruct` | True/True | task1_base_instruct | id | 262120 | 24000 | 0.7399 | 0.4870 | 0.7399 | 0.4870 | Gemma4 |
| `Gemma-4-E2B` | `Gemma-4-E2B-Instruct` | True/True | task1_base_instruct | id | 262120 | 24000 | 0.6877 | 0.5150 | 0.6877 | 0.5150 | Gemma4 |
| `Gemma-3-1B` | `Gemma-3-1B-Instruct` | True/True | task1_base_instruct | id | 262137 | 24000 | 0.4038 | 0.7362 | 0.4038 | 0.7362 |  |

### untied / untied

| 数量 | R2_E 均值 | R2_E 中位数 | R2_E 最小值 | R2_E 最大值 | R2_U 均值 | R2_U 中位数 | R2_U 最小值 | R2_U 最大值 | rel_err_E 均值 | rel_err_U 均值 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | 0.9985 | 0.9990 | 0.9932 | 1.0000 | 0.9982 | 0.9989 | 0.9949 | 1.0000 | 0.0301 | 0.0339 |

#### untied / untied 明细

| 模型 A | 模型 B | tied A/B | 来源 | 对齐 | 公共 token 数 | 拟合行数 | R2_E | rel_err_E | R2_U | rel_err_U | 备注 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `DeepSeek-V3-Base` | `DeepSeek-V3` | False/False | task1_base_instruct|task2_model_series | id | 127998 | 24000 | 1.0000 | 0.0019 | 1.0000 | 0.0012 | MoE |
| `DeepSeek-V3.1-Base` | `DeepSeek-V3.1` | False/False | task1_base_instruct|task2_model_series | id | 127998 | 24000 | 1.0000 | 0.0032 | 1.0000 | 0.0027 | MoE |
| `Qwen2.5-32B` | `Qwen2.5-32B-Instruct` | False/False | task1_base_instruct | id | 151643 | 24000 | 0.9999 | 0.0099 | 0.9998 | 0.0149 |  |
| `Qwen2.5-14B` | `Qwen2.5-14B-Instruct` | False/False | task1_base_instruct | id | 151643 | 24000 | 0.9999 | 0.0113 | 0.9998 | 0.0152 |  |
| `Qwen2.5-72B-Base` | `Qwen2.5-72B-Instruct` | False/False | task1_base_instruct | id | 151643 | 24000 | 0.9998 | 0.0140 | 0.9995 | 0.0212 |  |
| `Qwen2.5-7B` | `Qwen2.5-7B-Instruct` | False/False | task1_base_instruct | id | 151643 | 24000 | 0.9998 | 0.0153 | 0.9994 | 0.0245 |  |
| `Llama-3.1-70B-Base` | `Llama-3.1-70B-Instruct` | False/False | task1_base_instruct | id | 128000 | 24000 | 0.9990 | 0.0314 | 0.9982 | 0.0423 |  |
| `Qwen3.5-9B-Base` | `Qwen3.5-9B-Instruct` | False/False | task1_base_instruct | id | 248044 | 24000 | 0.9988 | 0.0341 | 0.9985 | 0.0369 |  |
| `Qwen3.5-35B-A3B-Base` | `Qwen3.5-35B-A3B-Instruct` | False/False | task1_base_instruct | id | 248044 | 24000 | 0.9988 | 0.0341 | 0.9989 | 0.0312 | MoE |
| `Llama-3.1-8B` | `Llama-3.1-8B-Instruct` | False/False | task1_base_instruct | id | 128000 | 24000 | 0.9987 | 0.0363 | 0.9979 | 0.0441 |  |
| `Qwen3-14B-Base` | `Qwen3-14B` | False/False | task1_base_instruct | id | 151643 | 24000 | 0.9965 | 0.0583 | 0.9950 | 0.0702 |  |
| `Qwen3-8B-Base` | `Qwen3-8B` | False/False | task1_base_instruct | id | 151643 | 24000 | 0.9964 | 0.0593 | 0.9951 | 0.0680 |  |
| `Qwen3-30B-A3B-Base` | `Qwen3-30B-A3B` | False/False | task1_base_instruct | id | 151643 | 24000 | 0.9932 | 0.0820 | 0.9949 | 0.0689 | MoE |

### 混合 tied 状态

| 数量 | R2_E 均值 | R2_E 中位数 | R2_E 最小值 | R2_E 最大值 | R2_U 均值 | R2_U 中位数 | R2_U 最小值 | R2_U 最大值 | rel_err_E 均值 | rel_err_U 均值 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | - | - | - | - | - | - | - | - | - | - |

#### 混合 tied 状态明细

| 模型 A | 模型 B | tied A/B | 来源 | 对齐 | 公共 token 数 | 拟合行数 | R2_E | rel_err_E | R2_U | rel_err_U | 备注 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| - | - | - | - | - | - | - | - | - | - | - | 无数据行 |

## 备注

- `actual_tied=True` 表示该模型抽取出的 E 和 U 矩阵在数值上相等。
- 当前结果文件中没有混合 tied 状态的 Base-Instruct 模型对。
- tied / tied 组整体均值相对较低，主要由若干 Gemma Base-Instruct 模型对拉低；新增 MoE pair 中 DeepSeek V3/V3.1 与 Qwen MoE 的子采样仿射 R2 仍接近 1。
