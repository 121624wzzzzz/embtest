# Base-Instruct Affine Results by Tied Status

Source: `summary_pair.csv`, filtered by `source_tasks=task1_base_instruct`.

The fitted relation is `Y ~= X * A + b`. `R2_E` measures Base embedding to Instruct embedding affine fit; `R2_U` measures Base unembedding to Instruct unembedding affine fit. Higher R2 and lower relative error indicate a stronger affine relationship.

### All Base-Instruct Pairs

| n | R2_E mean | R2_E median | R2_E min | R2_E max | R2_U mean | R2_U median | R2_U min | R2_U max | rel_err_E mean | rel_err_U mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 31 | 0.9424 | 0.9964 | 0.4038 | 0.9999 | 0.9422 | 0.9950 | 0.4038 | 0.9998 | 0.1395 | 0.1417 |

### Llama Base-Instruct Pairs

| n | R2_E mean | R2_E median | R2_E min | R2_E max | R2_U mean | R2_U median | R2_U min | R2_U max | rel_err_E mean | rel_err_U mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.9891 | 0.9902 | 0.9771 | 0.9990 | 0.9887 | 0.9898 | 0.9771 | 0.9982 | 0.0833 | 0.0880 |

#### Llama Details

| model_a | model_b | tied A/B | align | n_common | n_fit | R2_E | rel_err_E | R2_U | rel_err_U | note |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `Llama-3.1-70B-Base` | `Llama-3.1-70B-Instruct` | False/False | id | 128000 | 24000 | 0.9990 | 0.0314 | 0.9982 | 0.0423 | |
| `Llama-3.1-8B` | `Llama-3.1-8B-Instruct` | False/False | id | 128000 | 24000 | 0.9987 | 0.0363 | 0.9979 | 0.0441 | |
| `Llama-3.2-3B` | `Llama-3.2-3B-Instruct` | True/True | id | 128000 | 24000 | 0.9817 | 0.1251 | 0.9817 | 0.1251 | |
| `Llama-3.2-1B` | `Llama-3.2-1B-Instruct` | True/True | id | 128000 | 24000 | 0.9771 | 0.1406 | 0.9771 | 0.1406 | |

## Results By Tied Status

### tied / tied

| n | R2_E mean | R2_E median | R2_E min | R2_E max | R2_U mean | R2_U median | R2_U min | R2_U max | rel_err_E mean | rel_err_U mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 22 | 0.9193 | 0.9896 | 0.4038 | 0.9998 | 0.9193 | 0.9896 | 0.4038 | 0.9998 | 0.1843 | 0.1843 |

#### tied / tied details

| model_a | model_b | tied A/B | align | n_common | n_fit | R2_E | rel_err_E | R2_U | rel_err_U | note |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `Qwen2.5-3B` | `Qwen2.5-3B-Instruct` | True/True | id | 151643 | 24000 | 0.9998 | 0.0151 | 0.9998 | 0.0151 | |
| `Qwen2.5-1.5B` | `Qwen2.5-1.5B-Instruct` | True/True | id | 151643 | 24000 | 0.9997 | 0.0164 | 0.9997 | 0.0164 | |
| `Gemma-2-2B` | `Gemma-2-2B-Instruct` | True/True | id | 255993 | 24000 | 0.9994 | 0.0224 | 0.9994 | 0.0224 | |
| `Gemma-2-27B` | `Gemma-2-27B-Instruct` | True/True | id | 255993 | 24000 | 0.9980 | 0.0439 | 0.9980 | 0.0439 | |
| `Qwen3.5-0.8B-Base` | `Qwen3.5-0.8B-Instruct` | True/True | id | 248044 | 24000 | 0.9979 | 0.0406 | 0.9979 | 0.0406 | |
| `Qwen3.5-4B-Base` | `Qwen3.5-4B-Instruct` | True/True | id | 248044 | 24000 | 0.9975 | 0.0468 | 0.9975 | 0.0468 | |
| `Qwen3.5-2B-Base` | `Qwen3.5-2B-Instruct` | True/True | id | 248044 | 24000 | 0.9969 | 0.0504 | 0.9969 | 0.0504 | |
| `Qwen3-1.7B-Base` | `Qwen3-1.7B` | True/True | id | 151643 | 24000 | 0.9944 | 0.0730 | 0.9944 | 0.0730 | |
| `Gemma-2-9B` | `Gemma-2-9B-Instruct` | True/True | id | 255995 | 24000 | 0.9925 | 0.0762 | 0.9925 | 0.0762 | |
| `Qwen3-4B-Base` | `Qwen3-4B` | True/True | id | 151643 | 24000 | 0.9911 | 0.0906 | 0.9911 | 0.0906 | |
| `Qwen2.5-0.5B` | `Qwen2.5-0.5B-Instruct` | True/True | id | 151643 | 24000 | 0.9906 | 0.0893 | 0.9906 | 0.0893 | |
| `Qwen3-0.6B-Base` | `Qwen3-0.6B` | True/True | id | 151643 | 24000 | 0.9887 | 0.1006 | 0.9887 | 0.1006 | |
| `Llama-3.2-3B` | `Llama-3.2-3B-Instruct` | True/True | id | 128000 | 24000 | 0.9817 | 0.1251 | 0.9817 | 0.1251 | |
| `Llama-3.2-1B` | `Llama-3.2-1B-Instruct` | True/True | id | 128000 | 24000 | 0.9771 | 0.1406 | 0.9771 | 0.1406 | |
| `Gemma-3-27B` | `Gemma-3-27B-Instruct` | True/True | id | 262137 | 24000 | 0.9770 | 0.1510 | 0.9770 | 0.1510 | |
| `Gemma-3-12B` | `Gemma-3-12B-Instruct` | True/True | id | 262137 | 24000 | 0.9721 | 0.1655 | 0.9721 | 0.1655 | |
| `Gemma-3-4B` | `Gemma-3-4B-Instruct` | True/True | id | 262137 | 24000 | 0.9620 | 0.1915 | 0.9620 | 0.1915 | |
| `Gemma-4-31B` | `Gemma-4-31B-Instruct` | True/True | id | 262120 | 24000 | 0.8252 | 0.4099 | 0.8252 | 0.4099 | |
| `Gemma-4-E4B` | `Gemma-4-E4B-Instruct` | True/True | id | 262120 | 24000 | 0.7516 | 0.4676 | 0.7516 | 0.4676 | |
| `Gemma-4-26B-A4B` | `Gemma-4-26B-A4B-Instruct` | True/True | id | 262120 | 24000 | 0.7399 | 0.4870 | 0.7399 | 0.4870 | |
| `Gemma-4-E2B` | `Gemma-4-E2B-Instruct` | True/True | id | 262120 | 24000 | 0.6877 | 0.5150 | 0.6877 | 0.5150 | |
| `Gemma-3-1B` | `Gemma-3-1B-Instruct` | True/True | id | 262137 | 24000 | 0.4038 | 0.7362 | 0.4038 | 0.7362 | |

### untied / untied

| n | R2_E mean | R2_E median | R2_E min | R2_E max | R2_U mean | R2_U median | R2_U min | R2_U max | rel_err_E mean | rel_err_U mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 0.9988 | 0.9990 | 0.9964 | 0.9999 | 0.9981 | 0.9985 | 0.9950 | 0.9998 | 0.0300 | 0.0375 |

#### untied / untied details

| model_a | model_b | tied A/B | align | n_common | n_fit | R2_E | rel_err_E | R2_U | rel_err_U | note |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `Qwen2.5-32B` | `Qwen2.5-32B-Instruct` | False/False | id | 151643 | 24000 | 0.9999 | 0.0099 | 0.9998 | 0.0149 | |
| `Qwen2.5-14B` | `Qwen2.5-14B-Instruct` | False/False | id | 151643 | 24000 | 0.9999 | 0.0113 | 0.9998 | 0.0152 | |
| `Qwen2.5-72B-Base` | `Qwen2.5-72B-Instruct` | False/False | id | 151643 | 24000 | 0.9998 | 0.0140 | 0.9995 | 0.0212 | |
| `Qwen2.5-7B` | `Qwen2.5-7B-Instruct` | False/False | id | 151643 | 24000 | 0.9998 | 0.0153 | 0.9994 | 0.0245 | |
| `Llama-3.1-70B-Base` | `Llama-3.1-70B-Instruct` | False/False | id | 128000 | 24000 | 0.9990 | 0.0314 | 0.9982 | 0.0423 | |
| `Qwen3.5-9B-Base` | `Qwen3.5-9B-Instruct` | False/False | id | 248044 | 24000 | 0.9988 | 0.0341 | 0.9985 | 0.0369 | |
| `Llama-3.1-8B` | `Llama-3.1-8B-Instruct` | False/False | id | 128000 | 24000 | 0.9987 | 0.0363 | 0.9979 | 0.0441 | |
| `Qwen3-14B-Base` | `Qwen3-14B` | False/False | id | 151643 | 24000 | 0.9965 | 0.0583 | 0.9950 | 0.0702 | |
| `Qwen3-8B-Base` | `Qwen3-8B` | False/False | id | 151643 | 24000 | 0.9964 | 0.0593 | 0.9951 | 0.0680 | |

### mixed tied status

无结果。

#### mixed tied status details

| model_a | model_b | tied A/B | align | n_common | n_fit | R2_E | rel_err_E | R2_U | rel_err_U | note |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| - | - | - | - | - | - | - | - | - | - | No rows |

## Notes

- `actual_tied=True` means the extracted E and U matrices are numerically equal for that model.
- There are no mixed tied-status Base-Instruct pairs in the current result file.
- The relatively low overall tied/tied mean is driven mainly by several Gemma Base-Instruct pairs with much lower R2 than Qwen and Llama.
