# 实验主方法与指标定义

本文档记录当前 `ijcai_clean` 主线实验的方法、数据接口、指标定义与运行口径。研究性结论和异常解释放在 `../analysis.md`，历史实验摘要放在 `historical/`。

## 维护记录

- 2026-05-04：建立 Task1 Base/Instruct GCorr、`extracts/` 数据接口、bootstrap 统计、validate/csv-only 运行模式说明。
- 2026-05-10：补充 Task2-5 当前口径与主线入口。
- 2026-05-18：清理重复旧段落，补充 Base-Instruct full-vocab 仿射 / `A-I` / SVD 诊断和 `energy@1%h/5%h/10%h`。
- 2026-05-22：同步主线入口为 `ijcai_clean/scripts/run.py task1..task6`，旧 `run_task*.py` wrapper 仅作为兼容入口保留。
- 2026-05-23：Base-Instruct pair 扩展到 35 组（补 Qwen MoE 与 DeepSeek V3/V3.1 MoE），Task5 / Task6 结果与文档口径同步；Task6 支持复用已有 CSV 行的增量计算。

## 当前任务

| 任务 | 配置 | 入口 | 主要输出 |
|---|---|---|---|
| Task1 Base/Instruct GCorr | `configs/base_instruct_pairs.yaml` | `ijcai_clean/scripts/run.py task1` | `ijcai_clean/results/task1_base_instruct/summary.csv` |
| Task2 Model-Series GCorr | `configs/model_series.yaml` | `ijcai_clean/scripts/run.py task2` | `ijcai_clean/results/task2_model_series/summary.csv` |
| Task3 Cross-Scale GCorr | `configs/cross_scale_groups.yaml` + `configs/model_series.yaml` | `ijcai_clean/scripts/run.py task3` | `ijcai_clean/results/task3_cross_scale_groups/summary.csv` |
| Task4 MoE Cross-Family GCorr | `configs/moe_cross_family.yaml` + `configs/model_series.yaml` | `ijcai_clean/scripts/run.py task4` | `ijcai_clean/results/task4_moe_cross_family/summary.csv` |
| Task5 Affine Relations (子采样) | `configs/affine_pairs.yaml` | `ijcai_clean/scripts/run.py task5` | `ijcai_clean/results/task5_affine_subsampled/summary_pair.csv` |
| Task6 Base-Instruct full-vocab affine / SVD | `configs/base_instruct_pairs.yaml` | `ijcai_clean/scripts/run.py task6` | `ijcai_clean/results/task6_base_instruct_full_vocab/summary_pair_base_instruct_full_vocab.csv` |

Task2-4 会先生成 `generated_pairs.yaml` 和 `pair_plan.csv`，再复用 Task1 的 GCorr runner。Task5 读取 Task1-4 的 pair 集合并做仿射拟合。full-vocab 诊断直接读取 `configs/base_instruct_pairs.yaml`，按完整词表 id 对齐，不采样 token 行。

推荐统一使用 `PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run.py <task>` 运行主线任务。`run_task*_*.py`（含 Task6 的 `run_task6_base_instruct_full_vocab_affine.py`）仍保留为旧命令兼容 wrapper。

## 实验设置与产物总览

这一节按实验说明“输入是什么、怎么跑、输出什么、看哪个文件”。除特别说明外，所有任务都从 `extracts/` 读取标准化 E/U 矩阵。

### Task1 Base/Instruct GCorr

| 项目 | 设置 |
|---|---|
| 目标 | 比较同一模型 Base 与 Instruct 版本的 E/U 几何结构是否保持。 |
| pair 来源 | `configs/base_instruct_pairs.yaml`。 |
| 对齐方式 | 通常同 vocab，按 token id 对齐；仍会走通用 token 对齐与特殊 token 排除逻辑。 |
| 核心方法 | GCorr，分别计算 E-E 与 U-U 的 `cos`、`euc`、`euc2` 几何相关。 |
| 默认采样 | `n_tokens=20000`，`n_pairs=5000000`，`n_bootstrap=100`。 |
| 入口 | `PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run.py task1 --devices auto`。 |
| 输出目录 | `ijcai_clean/results/task1_base_instruct/`。 |

主要产物：

- `summary.csv`：每个 Base/Instruct pair 的 GCorr 均值、标准差、标准误、95% CI 和 median；日常看结果优先读这个。
- `bootstrap_results.csv`：每次 bootstrap 的原始结果；用于断点续跑、误差条和复核。
- `metadata.json`：运行参数、pair 数、设备等元信息。

### Task2 Model-Series GCorr

| 项目 | 设置 |
|---|---|
| 目标 | 比较同一模型系列内部不同规模 / 版本之间的几何相似性。 |
| pair 来源 | `configs/model_series.yaml`，每个系列内部生成 `C(n,2)` pair；同条目同时有 base/instruct 候选时优先使用 instruct 侧代表。 |
| 对齐方式 | 同 vocab 用 token id；不同 vocab 用 token string 交集。 |
| 核心方法 | 与 Task1 相同的 GCorr runner。 |
| 默认采样 | 与 Task1 相同。 |
| 入口 | `PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run.py task2 --devices auto`。 |
| 输出目录 | `ijcai_clean/results/task2_model_series/`。 |

主要产物：

- `pair_plan.csv`：从配置展开后的 pair 计划，适合检查实际跑了哪些模型对。
- `generated_pairs.yaml`：脚本生成的 pair 配置，供 GCorr runner 复用。
- `summary.csv`：系列内 pair 的 GCorr 汇总。
- `bootstrap_results.csv`：bootstrap 明细。
- `metadata.json`：运行元信息。

### Task3 Cross-Scale GCorr

| 项目 | 设置 |
|---|---|
| 目标 | 比较不同模型系列、不同规模桶之间的几何相似性。 |
| pair 来源 | `configs/cross_scale_groups.yaml` + `configs/model_series.yaml`。 |
| 规模桶 | `le_4b`、`gt_4b_lt_27b`、`ge_27b`。 |
| pair 规则 | 跨系列生成 pair，跳过同系列 pair，避免重复 Task2。 |
| 核心方法 | 与 Task1 相同的 GCorr runner。 |
| 默认采样 | 与 Task1 相同。 |
| 入口 | `PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run.py task3 --devices auto`。 |
| 输出目录 | `ijcai_clean/results/task3_cross_scale_groups/`。 |

主要产物：

- `pair_plan.csv`：跨系列、跨规模桶 pair 计划。
- `generated_pairs.yaml`：生成后的 GCorr pair 配置。
- `summary.csv`：跨规模桶 GCorr 汇总。
- `bootstrap_results.csv`：bootstrap 明细。
- `metadata.json`：运行元信息。

### Task4 MoE Cross-Family GCorr

| 项目 | 设置 |
|---|---|
| 目标 | 专门比较 MoE 模型与 dense 模型、不同系列代表模型之间的几何关系。 |
| pair 来源 | `configs/moe_cross_family.yaml` + `configs/model_series.yaml`。 |
| pair 规则 | 由 MoE 配置指定核心 MoE / dense 代表，再展开跨 family pair。 |
| 核心方法 | 与 Task1 相同的 GCorr runner。 |
| 默认采样 | 与 Task1 相同。 |
| 入口 | `PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run.py task4 --devices auto`。 |
| 输出目录 | `ijcai_clean/results/task4_moe_cross_family/`。 |

主要产物：

- `pair_plan.csv`：MoE 跨系列 pair 计划。
- `generated_pairs.yaml`：生成后的 GCorr pair 配置。
- `summary.csv`：MoE 跨 family GCorr 汇总。
- `bootstrap_results.csv`：bootstrap 明细。
- `metadata.json`：运行元信息。

### Task5 Affine Relations

| 项目 | 设置 |
|---|---|
| 目标 | 检查两个模型空间是否可由全局仿射变换 `Y ~= X A + b` 解释。 |
| pair 来源 | `configs/affine_pairs.yaml` 指定要复用的任务来源，当前主要读取 Task1-4 pair 并集。 |
| 对齐方式 | 同 vocab 用 token id；不同 vocab 用 token string 交集。 |
| 拟合对象 | 跨模型 E-E、U-U；另对每个模型做内部 E→U。 |
| 默认采样 | `max_fit_rows=24000`，`min_common_tokens=5000`；Task5 是采样行拟合，不是 full-vocab。 |
| 核心指标 | `R2`、`rel_err`、`norm_A`、`norm_b`。 |
| 入口 | `PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run.py task5 --devices auto`。 |
| 输出目录 | `ijcai_clean/results/task5_affine_subsampled/`。 |

主要产物：

- `summary_pair.csv`：跨模型 pair 的 E-E / U-U 仿射结果，是 Task5 主结果。
- `summary_intra_EU.csv`：每个模型内部 E→U 仿射结果，用于检查 E/U 是否近似仿射相关。
- `metadata.json`：运行配置、pair 来源和设备等元信息。
- `base_instruct_affine_tied_report.md`：从 Task5 结果中抽取 Base/Instruct pair，按 tied/untied 口径整理的轻量报告。

### Task6 Base-Instruct Full-Vocab Affine / A-I / SVD

| 项目 | 设置 |
|---|---|
| 目标 | 对 Base/Instruct pair 做完整词表仿射拟合，并分析 `A-I` 与 `E_instruct - E_base` 的低秩结构。 |
| pair 来源 | `configs/base_instruct_pairs.yaml`。 |
| 对齐方式 | 完整 token id 对齐；要求 base/instruct vocab size 和 hidden dim 一致。 |
| 拟合对象 | `E_base -> E_instruct`；若 E/U 实际 tied，则 U 结果复用 E，否则另跑 U。 |
| 计算方式 | 流式 centered normal equations，避免一次性物化完整大矩阵到 GPU。 |
| 核心指标 | full-vocab `R2/rel_err`、`A` 诊断、`E_delta` SVD、`A_delta=A-I` SVD。 |
| 相对能量 | `energy_at_1pct_h / 5pct_h / 10pct_h`，用于跨 hidden dim 公平比较谱集中程度。 |
| 入口 | `PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run.py task6`。 |
| 输出目录 | `ijcai_clean/results/task6_base_instruct_full_vocab/`。 |

主要产物：

- `summary_pair_base_instruct_full_vocab.csv`：完整 CSV，包含 full-vocab 仿射、`A` 诊断、SVD rank/effective rank/energy 指标；后续统计分析优先读这个。
- `base_instruct_full_vocab_affine_report.md`：人类可读报告，列出各组均值、逐模型结果和关键 `energy@5%h`。
- `base_instruct_full_vocab_metadata.json`：运行参数、设备、batch size、输出路径、复用行数和新增计算行数等元信息。

## 数据接口

当前主线不直接读取完整 HuggingFace / ModelScope 权重目录，而是读取标准化抽取后的 E/U 矩阵：

- `extracts/<model_name>.safetensors`
- `extracts/<model_name>.info.json`

`model_name` 必须与 `configs/models.yaml` 中的简称一致。`*.info.json` 的核心字段包括：

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
| 特殊 token | GCorr / Task5 默认排除 `bos/eos/pad/unk` 和 tokenizer 的 `all_special_ids`。 |
| 共同 token 过少 | GCorr 少于 1000 时跳过；Task5 默认少于 5000 时跳过。 |
| Base-Instruct full-vocab | 同 vocab / 同 hidden dim 时使用完整 id 对齐，不排除 token、不采样。 |

相关逻辑位于 `ijcai_clean/src/ijcai_clean/alignment.py`；full-vocab 诊断核心实现位于 `ijcai_clean/src/ijcai_clean/experiments/full_vocab_affine.py`，CLI wrapper 位于 `ijcai_clean/scripts/run_task6_base_instruct_full_vocab_affine.py`。

## GCorr 方法

GCorr（Global Correlation）衡量两个矩阵在 token 空间中诱导出的几何结构是否相似。它不直接比较单个 token 向量是否逐元素相等，而是比较 token-token 关系是否相似。

给定两个对齐后的矩阵：

```text
X = E_a 或 U_a
Y = E_b 或 U_b
```

先采样 token pair：

```text
(i_1, j_1), (i_2, j_2), ..., (i_m, j_m), 其中 i_k < j_k
```

分别在 `X` 和 `Y` 中计算距离或相似度，得到两个长度为 `m` 的向量 `d_X` 和 `d_Y`，然后计算 Pearson correlation：

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

## Bootstrap 与输出

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

每次 bootstrap 输出 6 个核心指标：

- `gcorr_E_cos`
- `gcorr_E_euc`
- `gcorr_E_euc2`
- `gcorr_U_cos`
- `gcorr_U_euc`
- `gcorr_U_euc2`

`summary.csv` 对每个指标输出 `mean`、`std`、`se`、`ci95_low`、`ci95_high`、`median`。

## Task5 仿射关系

Task5 对 Task1-4 的 pair 并集做跨模型仿射拟合：

```text
Y ~= X A + b
```

分别报告 E↔E 与 U↔U：

- `R2`：`1 - SS_res / SS_tot`，越接近 1 说明仿射解释力越强。
- `rel_err`：`||Y - Y_pred|| / ||Y||`。
- `norm_A`：`A` 的 Frobenius norm。
- `norm_b`：平移向量 `b` 的 L2 norm。

Task5 同时对所有出现过的模型做模型内部 E→U 仿射拟合，用于判断该模型 E 与 U 是否近似仿射相关。默认最多采样 `max_fit_rows = 24000` 行，公共 token 下限为 `min_common_tokens = 5000`。

## Base-Instruct full-vocab 诊断

`run_task6_base_instruct_full_vocab_affine.py` 专门分析 `configs/base_instruct_pairs.yaml` 中的 Base-Instruct pair。因为 Base/Instruct 同 vocab、同 hidden dim，可按完整 token id 行直接对齐，相当于对完整矩阵做分析，不再回到 token 采样。脚本会复用已有 `summary_pair_base_instruct_full_vocab.csv` 中的 pair 行，只计算配置中新出现或缺失的 pair。

输出包括三类指标：

- full-vocab 仿射拟合质量：`E_R2`、`U_R2`、`E_rel_err`、`U_rel_err` 等。
- `A` 矩阵相对单位阵的诊断：`trace_A_over_d`、`rel_A_minus_I_over_I`、`identity_cosine`、`offdiag_norm_over_A`、`rel_orthogonality_error_over_I`。
- SVD 低秩能量：对 `E_delta = E_instruct - E_base` 和 `A_delta = A - I` 分别计算 rank / effective rank / energy 曲线。

### A 诊断

| 指标 | 定义 | 解读 |
|---|---|---|
| `rel_A_minus_I_over_I` | `||A - I||_F / ||I||_F` | 越小越接近单位阵。 |
| `identity_cosine` | `trace(A) / (||A||_F ||I||_F)` | 越接近 1，说明 `A` 越接近单位阵方向。 |
| `offdiag_norm_over_A` | `||offdiag(A)||_F / ||A||_F` | 越大说明坐标混合越强。 |
| `rel_orthogonality_error_over_I` | `||A^T A - I||_F / ||I||_F` | 越小越接近正交旋转/反射。 |

### SVD 能量指标

对矩阵 `M` 的奇异值 `s_i`，使用 squared singular value energy：

```text
energy_i = s_i^2 / sum_j s_j^2
cumulative_energy_k = sum_{i<=k} energy_i
```

当前输出：

- `rank_50 / rank_80 / rank_90 / rank_95 / rank_99`：达到指定累计能量所需最小 rank。
- `effective_rank`：基于 energy 分布熵的有效秩。
- `energy_at_1/5/10/20/50/100/200/500/1000`：固定 top-k 累计能量。
- `energy_at_1pct_h / 5pct_h / 10pct_h`：top `1% / 5% / 10%` hidden dim 的累计能量，更适合跨 hidden dimension 比较。
- `top_singular_values`：前若干奇异值，便于检查谱衰减形状。

## Tied / Untied 定义

文档和结果中存在两类 tied 口径：

- `is_tied`：模型配置中的 `tie_word_embeddings` 声明，在结果 CSV 中通常表现为 `is_tied_a/is_tied_b` 或来自 `*.info.json`。
- `actual_tied`：实际矩阵比较结果，即 `np.allclose(E, U, rtol=1e-5, atol=1e-5)`；当前主线代码位于 `ijcai_clean/src/ijcai_clean/data.py`。

论文、报告和分组统计都优先使用 `actual_tied`，因为它来自真实 E/U 矩阵，而不是模型名或配置声明。`is_tied` 只作为辅助证据；如果二者不一致，应以 `actual_tied` 为准并在脚注说明声明值与实际权重存在差异。

历史审计输出位于：

- `docs/historical/legacy_results_summary/exp1_model_tag_audit.csv`：旧实验逐模型 tied / untied 审计。
- `docs/historical/legacy_results_summary/untied_comparison_summary.csv`：旧实验 tied/untied 分组 R² 汇总。
- `audits/`：`tools/get_model_useful.py` 持续写入的当前模型审计 JSON。

Base-Instruct full-vocab 诊断中，如果 pair 两侧 E/U 都实际 tied，则 U 结果可能直接复用 E 结果；解释时应说明这是因为实际 E/U tied。

## 当前结果规模

当前结果目录规模：

| 任务 | pair/model 数 | 主结果规模 |
|---|---:|---|
| Task1 | 35 pairs | `summary.csv` 35 行，`bootstrap_results.csv` 3500 行 |
| Task2 | 110 pairs | `summary.csv` 110 行，`bootstrap_results.csv` 11000 行 |
| Task3 | 176 pairs | `summary.csv` 176 行，`bootstrap_results.csv` 17600 行 |
| Task4 | 21 pairs | `summary.csv` 21 行，`bootstrap_results.csv` 2100 行 |
| Task5 | 340 unique pairs / 94 models | `summary_pair.csv` 340 行，`summary_intra_EU.csv` 94 行 |
| Base-Instruct full-vocab | 35 pairs | `summary_pair_base_instruct_full_vocab.csv` 35 行 |

## 结果解读建议

| 对比 | 解读重点 |
|---|---|
| `gcorr_E_*` 高 | 两个模型的输入词向量几何保持一致。 |
| `gcorr_U_*` 高 | 两个模型的输出词向量几何保持一致。 |
| `gcorr_U_*` 明显低于 `gcorr_E_*` | 训练或指令微调可能更多改变输出层几何。 |
| `cos` 高但 `euc` 较低 | 向量方向较一致，但长度或尺度结构发生变化。 |
| `euc` 和 `euc2` 都高 | 绝对距离结构也较稳定。 |
| Task5 `R2` 高 | 一个模型空间可被另一个模型空间的全局仿射变换较好解释。 |
| `A-I` 低秩 / `energy@%h` 高 | 仿射变换相对单位阵的偏移集中在少量主方向。 |

## 后续更新约定

新增实验、指标或默认参数变化时，请同步更新：

1. “维护记录”
2. “当前任务”
3. 对新增指标补充数学定义和解读方式
4. “当前结果规模”
5. `../analysis.md` 中的研究结论（如果结果解读发生变化）
