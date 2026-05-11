# 实验主方法与指标定义

本文档记录当前 `ijcai_clean` 主线实验的方法、数据接口、指标定义与运行口径。后续新增任务、指标或采样策略时，应优先更新本文档。

## 维护记录

- 2026-05-04：建立当前版本，覆盖 Task 1 Base/Instruct GCorr、`extracts/` 数据接口、bootstrap 统计、validate/csv-only 运行模式。
- 2026-05-10：清理重复段落，补充 Task 2-5 当前口径与主线入口。

## 当前任务

| 任务 | 配置 | 入口 | 主要输出 |
|---|---|---|---|
| Task 1 Base/Instruct GCorr | `configs/base_instruct_pairs.yaml` | `ijcai_clean/scripts/run_task1_base_instruct.py` | `ijcai_clean/results/task1_base_instruct/summary.csv` |
| Task 2 Model-Series GCorr | `configs/model_series.yaml` | `ijcai_clean/scripts/run_task2_model_series.py` | `ijcai_clean/results/task2_model_series/summary.csv` |
| Task 3 Cross-Scale GCorr | `configs/cross_scale_groups.yaml` + `configs/model_series.yaml` | `ijcai_clean/scripts/run_task3_cross_scale_groups.py` | `ijcai_clean/results/task3_cross_scale_groups/summary.csv` |
| Task 4 MoE Cross-Family GCorr | `configs/moe_cross_family.yaml` + `configs/model_series.yaml` | `ijcai_clean/scripts/run_task4_moe_cross_family.py` | `ijcai_clean/results/task4_moe_cross_family/summary.csv` |
| Task 5 Affine Relations | `configs/affine_pairs.yaml` | `ijcai_clean/scripts/run_task5_affine_relations.py` | `ijcai_clean/results/task5_affine_relations/summary_pair.csv` |

Task 2-4 会先生成 `generated_pairs.yaml` 和 `pair_plan.csv`，再复用 Task 1 的 GCorr runner。Task 5 读取 Task 1-4 的 pair 集合并做仿射拟合。

## 数据接口

当前主线不直接读取完整 HuggingFace / ModelScope 权重目录，而是读取标准化抽取后的 E/U 矩阵：

- `extracts/<model_name>.safetensors`
- `extracts/<model_name>.info.json`

其中 `model_name` 必须与 `configs/models.yaml` 中的简称一致。`*.info.json` 中的核心字段包括：

- `standardized_dims.embed`：标准化 E 形状。
- `standardized_dims.lm_head`：标准化 U 形状。
- `standardized_sources.embed`：E 在 safetensors 中的真实 key。
- `standardized_sources.lm_head`：U 在 safetensors 中的真实 key。
- `tie_word_embeddings`：模型配置声明的 tied 状态。

读取逻辑位于 `ijcai_clean/src/ijcai_clean/data.py`。

## 矩阵对象

| 符号 | 名称 | 形状 | 含义 |
|---|---|---|---|
| `E` | Embedding matrix | `vocab_size x hidden_dim` | 输入嵌入矩阵，将 token 映射到 hidden vector。 |
| `U` | Unembedding matrix / LM head | `vocab_size x hidden_dim` | 输出解嵌矩阵，用于将 hidden vector 映射回 token logits。 |

代码统一将 E/U 标准化为同一行口径：`[vocab_size, hidden_dim]`。

## Token 对齐

GCorr 与仿射拟合都要求两个矩阵的行具有可比较的 token 语义，因此先进行 token 对齐。

| 情况 | 对齐方式 |
|---|---|
| 两个模型 vocab size 相同 | 按 token id 对齐，即第 `i` 行对第 `i` 行。 |
| 两个模型 vocab size 不同 | 按 token string 求交集后对齐。 |
| 特殊 token | 排除 `bos/eos/pad/unk` 和 tokenizer 的 `all_special_ids`。 |
| 共同 token 过少 | GCorr 少于 1000 时跳过；Task 5 默认少于 5000 时跳过。 |

相关逻辑位于 `ijcai_clean/src/ijcai_clean/alignment.py`。

## GCorr 方法

GCorr（Global Correlation）衡量两个矩阵在 token 空间中诱导出的几何结构是否相似。它不直接比较单个 token 向量是否逐元素相等，而是比较 token-token 关系是否相似。

给定两个对齐后的矩阵：

- `X = E_a` 或 `U_a`
- `Y = E_b` 或 `U_b`

先采样一批 token pair：

```text
(i_1, j_1), (i_2, j_2), ..., (i_m, j_m), 其中 i_k < j_k
```

对每个 token pair，在 `X` 和 `Y` 中分别计算距离或相似度，形成两个长度为 `m` 的向量：

```text
d_X = [metric(X[i_1], X[j_1]), ..., metric(X[i_m], X[j_m])]
d_Y = [metric(Y[i_1], Y[j_1]), ..., metric(Y[i_m], Y[j_m])]
```

GCorr 定义为这两个向量的 Pearson correlation：

```text
GCorr_metric(X, Y) = PearsonCorr(d_X, d_Y)
```

当前实现计算三种 metric：

| 指标 | 定义 | 解释 |
|---|---|---|
| `cos` | `dot(x_i, x_j) / (||x_i|| * ||x_j||)` | 关注向量方向相似性，对向量长度较不敏感。 |
| `euc` | `||x_i - x_j||` | 欧氏距离，关注向量在空间中的绝对距离。 |
| `euc2` | `||x_i - x_j||^2` | 平方欧氏距离，放大较大的距离差异。 |

平方欧氏距离使用等价形式：

```text
||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * dot(x_i, x_j)
```

这样可以避免显式构造大规模差分张量，提高 GPU 计算效率。实现位于 `ijcai_clean/src/ijcai_clean/metrics.py`。

## Pearson 相关系数

对两个长度为 `m` 的数值向量 `a` 和 `b`：

```text
Pearson(a, b) =
  (m * sum(a*b) - sum(a) * sum(b))
  /
  sqrt((m * sum(a^2) - sum(a)^2) * (m * sum(b^2) - sum(b)^2))
```

当前实现用流式累加的方式统计：

- `sum_x`
- `sum_y`
- `sum_x2`
- `sum_y2`
- `sum_xy`

这样不需要保存所有 token-pair 的距离向量，适合 `n_pairs = 5000000` 的设置。

## 输出指标字段

每次 bootstrap 会输出以下 6 个核心 GCorr 指标：

| 字段 | 比较对象 | metric |
|---|---|---|
| `gcorr_E_cos` | `E_a` vs `E_b` | cosine similarity |
| `gcorr_E_euc` | `E_a` vs `E_b` | Euclidean distance |
| `gcorr_E_euc2` | `E_a` vs `E_b` | squared Euclidean distance |
| `gcorr_U_cos` | `U_a` vs `U_b` | cosine similarity |
| `gcorr_U_euc` | `U_a` vs `U_b` | Euclidean distance |
| `gcorr_U_euc2` | `U_a` vs `U_b` | squared Euclidean distance |

`summary.csv` 对每个指标输出：

- `mean`
- `std`
- `se`
- `ci95_low`
- `ci95_high`
- `median`

## Bootstrap 与断点续跑

默认 GCorr 参数：

| 参数 | 当前值 | 说明 |
|---|---:|---|
| `n_tokens` | `20000` | 每次 bootstrap 采样 token 数。 |
| `n_pairs` | `5000000` | 每次 bootstrap 采样 token pair 数。 |
| `n_bootstrap` | `100` | 每组模型 pair 的 bootstrap 次数。 |
| `seed` | `42` | 基础随机种子；第 `b` 次 bootstrap 使用 `seed + b`。 |
| `devices` | `auto` | 自动使用可见 GPU；可通过 `--exclude-gpus` 排除指定卡。 |
| `complete_mode` | `validate` | 结果完整时仍做小规模矩阵计算校验。 |
| `validation_n_tokens` | `1024` | validate 模式下的校验 token 数。 |
| `validation_n_pairs` | `10000` | validate 模式下的校验 token pair 数。 |

`bootstrap_results.csv` 是断点文件。脚本启动后会读取已完成的 `(model_a, model_b, bootstrap)` 三元组：

- 已存在的 bootstrap 不重复计算。
- 缺失的 bootstrap 会继续补齐。
- 完整时可用 `--complete-mode csv-only` 只重建 summary/metadata。

为了避免同时缓存多个大模型导致内存膨胀，当前实现按 pair 执行：加载当前 pair 的两个模型矩阵，补齐缺失 bootstrap，写 CSV，释放矩阵，再进入下一个 pair。

## Task 5 仿射关系

Task 5 对 Task 1-4 的 pair 并集做跨模型仿射拟合：

```text
Y ≈ X · A + b
```

分别报告 E↔E 与 U↔U 的：

- `R2`
- `rel_err`
- `norm_A`
- `norm_b`

同时对所有出现过的模型做模型内部 E→U 仿射拟合，用于判断该模型 E 与 U 是否近似仿射相关。默认最多采样 `max_fit_rows = 24000` 行，公共 token 下限为 `min_common_tokens = 5000`。

## Tied / Untied 定义

文档和结果中存在两类 tied 口径：

- `is_tied`：模型配置中的 `tie_word_embeddings` 声明。
- `actual_tied`：实际矩阵比较结果，即 `np.allclose(E, U, rtol=1e-5, atol=1e-5)`。

论文或报告分析时，优先使用 `actual_tied`，因为它来自真实 E/U 矩阵。

## 当前结果规模

截至 2026-05-10 当前结果目录：

| 任务 | pair/model 数 | 主结果规模 |
|---|---:|---|
| Task 1 | 31 pairs | `summary.csv` 31 行，`bootstrap_results.csv` 3100 行 |
| Task 2 | 110 pairs | `summary.csv` 110 行，`bootstrap_results.csv` 11000 行 |
| Task 3 | 176 pairs | `summary.csv` 176 行，`bootstrap_results.csv` 17600 行 |
| Task 4 | 21 pairs | `summary.csv` 21 行，`bootstrap_results.csv` 2100 行 |
| Task 5 | 338 unique pairs / 92 models | `summary_pair.csv` 338 行，`summary_intra_EU.csv` 92 行 |

## 结果解读建议

| 对比 | 解读重点 |
|---|---|
| `gcorr_E_*` 高 | 两个模型的输入词向量几何保持一致。 |
| `gcorr_U_*` 高 | 两个模型的输出词向量几何保持一致。 |
| `gcorr_U_*` 明显低于 `gcorr_E_*` | 训练或指令微调可能更多改变输出层几何。 |
| `cos` 高但 `euc` 较低 | 向量方向较一致，但长度或尺度结构发生变化。 |
| `euc` 和 `euc2` 都高 | 绝对距离结构也较稳定。 |
| Task 5 `R2` 高 | 一个模型空间可被另一个模型空间的线性/仿射变换较好解释。 |

## 后续更新约定

新增实验、指标或默认参数变化时，请同步更新：

1. “维护记录”
2. “当前任务”
3. “输出指标字段”
4. “当前结果规模”
5. 对新增指标补充数学定义和解读方式
# 实验方法与指标定义

本文档用于持续记录当前实验的核心方法、指标含义和参数设置。重点是解释 GCorr 方法本身，而不是代码结构。

## 更新记录

| 日期 | 更新内容 |
|------|----------|
| 2026-05-04 | 建立文档，记录 Task 1 Base/Instruct GCorr 的实验对象、方法原理、指标定义和参数设置。 |

## Task 1：Base/Instruct 几何相似性实验

本实验比较同一模型系列中 Base 模型与 Instruct 模型的词表几何结构是否保持一致。对每一组 Base/Instruct pair，分别比较输入嵌入矩阵 `E` 和输出解嵌矩阵 `U` 的全局几何结构。

| 项目 | 设置 |
|------|------|
| 任务名称 | Task 1 Base/Instruct GCorr |
| 实验对象 | Base 模型与对应 Instruct 模型 |
| 当前 pair 数 | 28 |
| 配置文件 | `configs/base_instruct_pairs.yaml` |
| 主要结果 | `ijcai_clean/results/task1_base_instruct/summary.csv` |
| bootstrap 明细 | `ijcai_clean/results/task1_base_instruct/bootstrap_results.csv` |

## 矩阵对象

对每个模型，实验关注两类矩阵：

| 符号 | 名称 | 形状 | 含义 |
|------|------|------|------|
| `E` | Embedding matrix | `vocab_size x hidden_dim` | 输入嵌入矩阵，将 token 映射到 hidden vector。 |
| `U` | Unembedding matrix / LM head | `vocab_size x hidden_dim` | 输出解嵌矩阵，用于将 hidden vector 映射回 token logits。 |

实验分别计算：

| 比较对象 | 含义 |
|----------|------|
| `E_base` vs `E_instruct` | Base 与 Instruct 的输入词向量几何是否一致。 |
| `U_base` vs `U_instruct` | Base 与 Instruct 的输出词向量几何是否一致。 |

## Token 对齐

GCorr 要求两个矩阵的行具有可比较的 token 语义，因此先进行 token 对齐。

| 情况 | 对齐方式 |
|------|----------|
| 两个模型 vocab size 相同 | 按 token id 对齐，即第 `i` 行对第 `i` 行。 |
| 两个模型 vocab size 不同 | 按 token string 求交集后对齐。 |
| 特殊 token | 排除 `bos/eos/pad/unk` 等 special tokens。 |
| 共同 token 过少 | 若共同 token 数少于 1000，则跳过该 pair。 |

当前 Task 1 的主要 Base/Instruct pair 基本为同 tokenizer / 同 vocab 口径，因此主要使用 token id 对齐。

## GCorr 核心方法

GCorr（Global Correlation）用于衡量两个矩阵诱导出的 token-token 几何关系是否相似。它不直接比较某个 token 向量是否逐元素相等，而是比较 token 之间的距离结构是否一致。

给定两个对齐后的矩阵 `X` 和 `Y`：

- 当比较输入嵌入时：`X = E_base`, `Y = E_instruct`
- 当比较输出解嵌时：`X = U_base`, `Y = U_instruct`

先采样若干 token pair：

```text
(i_1, j_1), (i_2, j_2), ..., (i_m, j_m), 其中 i_k < j_k
```

然后分别在 `X` 和 `Y` 中计算这些 token pair 的距离或相似度：

```text
d_X = [metric(X[i_1], X[j_1]), ..., metric(X[i_m], X[j_m])]
d_Y = [metric(Y[i_1], Y[j_1]), ..., metric(Y[i_m], Y[j_m])]
```

最后计算两个距离向量的 Pearson 相关系数：

```text
GCorr_metric(X, Y) = PearsonCorr(d_X, d_Y)
```

解释：

| GCorr 值 | 含义 |
|----------|------|
| 接近 `1` | 两个矩阵诱导出的 token-token 几何结构高度一致。 |
| 接近 `0` | 两个矩阵的几何结构相关性弱。 |
| 小于 `0` | 两个矩阵的几何关系呈反相关，通常不期望出现。 |

## 三类距离 / 相似度指标

当前实验对每个矩阵比较同时计算 `cos`、`euc` 和 `euc2` 三类 GCorr。

| 指标 | 定义 | 解释 |
|------|------|------|
| `cos` | `cos(x_i, x_j) = dot(x_i, x_j) / (||x_i|| * ||x_j||)` | 关注向量方向相似性，对向量长度较不敏感。 |
| `euc` | `euc(x_i, x_j) = ||x_i - x_j||` | 欧氏距离，关注向量在空间中的绝对距离。 |
| `euc2` | `euc2(x_i, x_j) = ||x_i - x_j||^2` | 平方欧氏距离，放大较大的距离差异。 |

平方欧氏距离在实现中使用等价形式：

```text
||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * dot(x_i, x_j)
```

这样可以避免显式构造大规模差分张量，提高 GPU 计算效率。

## 输出指标字段

每次 bootstrap 会输出以下 6 个核心 GCorr 指标：

| 字段 | 比较对象 | metric |
|------|----------|--------|
| `gcorr_E_cos` | `E_base` vs `E_instruct` | cosine similarity |
| `gcorr_E_euc` | `E_base` vs `E_instruct` | Euclidean distance |
| `gcorr_E_euc2` | `E_base` vs `E_instruct` | squared Euclidean distance |
| `gcorr_U_cos` | `U_base` vs `U_instruct` | cosine similarity |
| `gcorr_U_euc` | `U_base` vs `U_instruct` | Euclidean distance |
| `gcorr_U_euc2` | `U_base` vs `U_instruct` | squared Euclidean distance |

## Pearson 相关系数

GCorr 最后一步使用 Pearson correlation。对两个长度为 `m` 的向量 `a` 和 `b`：

```text
Pearson(a, b) =
  (m * sum(a*b) - sum(a) * sum(b))
  /
  sqrt((m * sum(a^2) - sum(a)^2) * (m * sum(b^2) - sum(b)^2))
```

实验中不会保存所有 pairwise 距离，而是流式累加 `sum_x`、`sum_y`、`sum_x2`、`sum_y2` 和 `sum_xy`，以支持百万级 token pair 计算。

## Bootstrap 设计

每组 Base/Instruct pair 运行 100 次 bootstrap。每次 bootstrap 重新采样 token 和 token pair，并计算一次完整的 6 个 GCorr 指标。

| 步骤 | 说明 |
|------|------|
| 1 | 从对齐后的 token 集合中采样 `n_tokens` 个 token。 |
| 2 | 在这批 token 内采样 `n_pairs` 个 token pair。 |
| 3 | 对 `E` 和 `U` 分别计算 `cos/euc/euc2` 三类 GCorr。 |
| 4 | 将单次结果写入 `bootstrap_results.csv`。 |
| 5 | 对 100 次结果汇总，生成 `summary.csv`。 |

## 汇总统计

`summary.csv` 对每个 GCorr 指标统计以下值：

| 后缀 | 含义 | 定义 |
|------|------|------|
| `_mean` | 均值 | 100 次 bootstrap 的平均值。 |
| `_std` | 标准差 | `np.std(values)`，当前为总体标准差口径。 |
| `_se` | 标准误 | `std / sqrt(n_bootstrap)`。 |
| `_ci95_low` | 95% 区间下界 | bootstrap 结果的 2.5 分位数。 |
| `_ci95_high` | 95% 区间上界 | bootstrap 结果的 97.5 分位数。 |
| `_median` | 中位数 | bootstrap 结果的 50 分位数。 |

例如：

| 字段 | 含义 |
|------|------|
| `gcorr_E_euc_mean` | E 矩阵欧氏 GCorr 的 bootstrap 均值。 |
| `gcorr_E_euc_ci95_low` | E 矩阵欧氏 GCorr 的 95% bootstrap 区间下界。 |
| `gcorr_U_cos_median` | U 矩阵 cosine GCorr 的 bootstrap 中位数。 |

## 当前实验参数

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `n_tokens` | `20000` | 每次 bootstrap 采样的 token 数。 |
| `n_pairs` | `5000000` | 每次 bootstrap 采样的 token pair 数。 |
| `n_bootstrap` | `100` | 每组模型 pair 的 bootstrap 次数。 |
| `seed` | `42` | 基础随机种子；第 `b` 次 bootstrap 使用 `seed + b`。 |
| `devices` | `auto` | 自动使用可见 GPU。 |
| `complete_mode` | `validate` | 结果完整时仍做小规模矩阵计算校验。 |
| `validation_n_tokens` | `1024` | validate 模式下的校验 token 数。 |
| `validation_n_pairs` | `10000` | validate 模式下的校验 token pair 数。 |

## 当前结果规模

| 项目 | 当前值 |
|------|--------|
| Base/Instruct pair 数 | `28` |
| 每组 bootstrap 数 | `100` |
| `bootstrap_results.csv` 行数 | `2800` |
| `summary.csv` 行数 | `28` |

## 结果解读建议

| 对比 | 解读重点 |
|------|----------|
| `gcorr_E_*` 高 | Base 与 Instruct 的输入词向量几何保持一致。 |
| `gcorr_U_*` 高 | Base 与 Instruct 的输出词向量几何保持一致。 |
| `gcorr_U_*` 明显低于 `gcorr_E_*` | 指令微调可能更多改变输出层几何。 |
| `cos` 高但 `euc` 较低 | 向量方向较一致，但长度或尺度结构发生变化。 |
| `euc` 和 `euc2` 都高 | 绝对距离结构也较稳定。 |

## 后续更新约定

后续新增实验、修改默认参数或新增指标时，请更新：

1. “更新记录”
2. “当前实验参数”
3. “输出指标字段”
4. “当前结果规模”
5. 对新增指标补充方法定义和解读方式
# 实验主方法与指标定义

本文档记录当前 `ijcai_clean` 实验仓的主方法、数据接口、指标定义与运行口径。后续新增任务、指标或采样策略时，应优先更新本文档。

## 维护记录

- 2026-05-04：建立当前版本。覆盖任务一 Base/Instruct GCorr、`extracts/` 数据接口、bootstrap 统计、validate/csv-only 运行模式。

## 当前主实验：Task 1 Base/Instruct GCorr

任务一比较同一模型家族内的 Base 模型与 Instruct 模型，例如 `Qwen3-0.6B-Base -> Qwen3-0.6B`。配置文件为：

- `configs/base_instruct_pairs.yaml`

运行入口为：

```bash
PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run_task1_base_instruct.py --devices auto
```

默认参数：

- `n_tokens = 20000`
- `n_pairs = 5000000`
- `n_bootstrap = 100`
- `seed = 42`
- `devices = auto`
- `complete_mode = validate`

结果目录：

- `ijcai_clean/results/task1_base_instruct/bootstrap_results.csv`
- `ijcai_clean/results/task1_base_instruct/summary.csv`
- `ijcai_clean/results/task1_base_instruct/metadata.json`

## 数据接口

当前实验不再直接读取完整 HuggingFace / ModelScope 权重目录，而是读取已经标准化抽取后的 E/U 矩阵：

- `extracts/<model_name>.safetensors`
- `extracts/<model_name>.info.json`

其中 `model_name` 必须与仓库 `configs/models.yaml` 里的简称一致。

### E 与 U

对每个模型，实验使用两类矩阵：

- `E`：input embedding，即 token id 到 hidden vector 的嵌入矩阵。
- `U`：unembedding / lm head，即 hidden vector 到 token logits 的输出矩阵。代码统一转置或标准化为与 `E` 相同的形状口径：`[vocab_size, hidden_dim]`。

`*.info.json` 中的核心字段：

- `standardized_dims.embed`：标准化 E 形状。
- `standardized_dims.lm_head`：标准化 U 形状。
- `standardized_sources.embed`：E 在 safetensors 中的真实 key。
- `standardized_sources.lm_head`：U 在 safetensors 中的真实 key。
- `tie_word_embeddings`：模型配置声明的 tied 状态。

读取逻辑位于：

- `ijcai_clean/src/ijcai_clean/data.py`

## Token 对齐

比较两个模型时，需要先确定比较哪些 token 行。

### 同词表 / 同 tokenizer 口径

若两个模型的 vocab size 相同，任务一按 token id 对齐：

- `E_a[token_id]` 对齐 `E_b[token_id]`
- `U_a[token_id]` 对齐 `U_b[token_id]`

代码会排除 tokenizer 的特殊 token，例如 `bos/eos/pad/unk` 和 `all_special_ids`。

### 跨 tokenizer 口径

如果 vocab size 不同，则按 token 字符串求交集：

- 将 token id 转为 token string。
- 取两个 tokenizer 的 token string 交集。
- 对同一 token string 的两侧 id 进行对齐。

如果共同 token 数少于 1000，则跳过该 pair。

相关逻辑位于：

- `ijcai_clean/src/ijcai_clean/alignment.py`

## GCorr：全局几何相关性

GCorr 衡量两个矩阵在 token 空间中诱导出的几何结构是否相似。直观地说，它不直接比较单个 token 向量是否相等，而是比较 token-token 关系是否相似。

给定两个对齐后的矩阵：

- `X = E_a` 或 `U_a`
- `Y = E_b` 或 `U_b`

先采样一批 token pair：

```text
(i_1, j_1), (i_2, j_2), ..., (i_m, j_m), 其中 i_k < j_k
```

对每个 token pair，在 `X` 和 `Y` 中分别计算距离或相似度，形成两个长度为 `m` 的向量：

```text
d_X = [metric(X[i_1], X[j_1]), ..., metric(X[i_m], X[j_m])]
d_Y = [metric(Y[i_1], Y[j_1]), ..., metric(Y[i_m], Y[j_m])]
```

GCorr 定义为这两个向量的 Pearson correlation：

```text
GCorr_metric(X, Y) = PearsonCorr(d_X, d_Y)
```

当前实现计算三种 metric：

- `cos`：余弦相似度。
- `euc`：欧氏距离。
- `euc2`：平方欧氏距离。

对应输出字段：

- `gcorr_E_cos`
- `gcorr_E_euc`
- `gcorr_E_euc2`
- `gcorr_U_cos`
- `gcorr_U_euc`
- `gcorr_U_euc2`

其中：

- `gcorr_E_*` 比较两个模型的 E 几何。
- `gcorr_U_*` 比较两个模型的 U 几何。

实现位于：

- `ijcai_clean/src/ijcai_clean/metrics.py`

## 距离与相似度定义

对两个 token 向量 `x_i, x_j`：

### Cosine

```text
cos(x_i, x_j) = dot(x_i, x_j) / (||x_i|| * ||x_j||)
```

### Squared Euclidean

```text
euc2(x_i, x_j) = ||x_i - x_j||^2
               = ||x_i||^2 + ||x_j||^2 - 2 * dot(x_i, x_j)
```

### Euclidean

```text
euc(x_i, x_j) = sqrt(euc2(x_i, x_j))
```

代码使用点积和预计算范数实现，避免显式构造 `x_i - x_j` 的大张量。

## Pearson 相关系数

对两个长度为 `m` 的数值向量 `a` 和 `b`，Pearson correlation 为：

```text
Pearson(a, b) =
  (m * sum(a*b) - sum(a) * sum(b))
  /
  sqrt((m * sum(a^2) - sum(a)^2) * (m * sum(b^2) - sum(b)^2))
```

当前实现用流式累加的方式统计：

- `sum_x`
- `sum_y`
- `sum_x2`
- `sum_y2`
- `sum_xy`

这样不需要保存所有 token-pair 的距离向量，适合 `n_pairs = 5000000` 的设置。

## Bootstrap 统计

一次 bootstrap 的流程：

1. 根据 `seed + bootstrap_index` 采样 token 行。
2. 在采样 token 内采样 `n_pairs` 个 token pair。
3. 分别计算 E 与 U 的三类 GCorr。
4. 将结果写入 `bootstrap_results.csv`。

当前默认 `n_bootstrap = 100`。对每个模型 pair，最终有 100 条 bootstrap 行。

`summary.csv` 将 100 次 bootstrap 聚合为统计量：

- `mean`：均值。
- `std`：标准差，当前使用 `np.std(values)`，即总体标准差口径。
- `se`：标准误，定义为 `std / sqrt(n_bootstrap)`。
- `ci95_low`：bootstrap 结果的 2.5 分位数。
- `ci95_high`：bootstrap 结果的 97.5 分位数。
- `median`：中位数。

字段示例：

- `gcorr_E_euc_mean`
- `gcorr_E_euc_std`
- `gcorr_E_euc_se`
- `gcorr_E_euc_ci95_low`
- `gcorr_E_euc_ci95_high`
- `gcorr_E_euc_median`

## Tied / Untied 定义

文档和结果中存在两类 tied 口径：

- `is_tied`：模型配置中的 `tie_word_embeddings` 声明。
- `actual_tied`：实际矩阵比较结果，即 `np.allclose(E, U, rtol=1e-5, atol=1e-5)`。

在任务一结果中记录的是：

- `actual_tied_a`
- `actual_tied_b`

论文或报告分析时，优先使用 `actual_tied`，因为它来自真实 E/U 矩阵。

## 运行模式

任务一入口支持两种完整结果后的行为：

### validate

默认模式：

```bash
python ijcai_clean/scripts/run_task1_base_instruct.py --devices auto
```

如果 `bootstrap_results.csv` 已经完整，仍会逐 pair 读取一小批 E/U 行并运行小规模 GCorr，用于确认当前代码、矩阵文件和 GPU 计算路径正常。

默认验证参数：

- `validation_n_tokens = 1024`
- `validation_n_pairs = 10000`

验证不会覆盖正式 bootstrap 结果。

### csv-only

快速模式：

```bash
python ijcai_clean/scripts/run_task1_base_instruct.py --devices auto --complete-mode csv-only
```

如果 `bootstrap_results.csv` 已经完整，只从 CSV 重建 `summary.csv` 和 `metadata.json`，不加载矩阵。

## 断点续跑与资源控制

`bootstrap_results.csv` 是任务一的断点文件。脚本启动后会读取已完成的 `(model_a, model_b, bootstrap)` 三元组：

- 已存在的 bootstrap 不重复计算。
- 缺失的 bootstrap 会继续补齐。

为了避免同时缓存多个大模型导致内存膨胀，当前实现按 pair 执行：

1. 加载当前 pair 的两个模型矩阵。
2. 并行补齐该 pair 缺失的 bootstrap。
3. 写入 CSV。
4. 释放矩阵，清理 GPU cache。
5. 进入下一个 pair。

这比一次性提交所有 pair 的任务更稳定，尤其适合 70B / 72B / Gemma 大模型。

## 当前任务一结果口径

截至本文档建立时，任务一结果为：

- pair 数：28
- 每组 bootstrap：100
- `bootstrap_results.csv` 行数：2800
- `summary.csv` 行数：28
- `n_tokens`：20000
- `n_pairs`：5000000

## 后续更新约定

新增实验或指标时，请按以下方式更新本文档：

1. 在“维护记录”添加日期和变更摘要。
2. 如果新增任务，在“当前主实验”后添加新的任务章节。
3. 如果新增指标，在“指标定义”附近补充数学定义、字段名和解释。
4. 如果改变默认参数，在对应任务章节和“当前结果口径”中同步更新。
