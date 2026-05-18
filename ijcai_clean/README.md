# IJCAI Embedding Geometry（clean 子目录）

本目录整理自历史路径 `/root/shared-nvme/ijcai`，目标是把几何分析代码、轻量结果与文档留在可读的 `ijcai_clean/` 下；**大权重与缓存仍在仓库根目录**（见下文「数据布局」）。

## 目录结构（整理后）

| 路径 | 职责 |
|------|------|
| 根目录 [`configs/`](../configs) | 全局模型配置与任务配置：`base_instruct_pairs.yaml`（任务一）、`model_series.yaml`（任务二系列列表）、`cross_scale_groups.yaml`（任务三三档规模桶） |
| `src/ijcai_clean/` | **当前**分析代码包：加载 `extracts/`、对齐 tokenizer、GCorr、任务一/二/三汇总 |
| `scripts/` | 可执行入口，如 `run_task1_base_instruct.py`、`run_task2_model_series.py`、`run_task3_cross_scale_groups.py` |
| `results/` | 当前实验输出（如 `task1_base_instruct/`、`task2_model_series/`、`task3_cross_scale_groups/`） |
| `legacy/` | **旧参考**：原 `run_exp1_v4.py`、仿射实验、矩阵对比脚本（路径与环境仍为历史口径） |
| `docs/` | 当前方法、状态、审计说明；历史文档在 `docs/historical/` |
| `analysis.md` | 当前研究结论备忘：Base-Instruct full-vocab、Gemma 异常、SVD 低秩分析 |

## 子目录功能

- `src/ijcai_clean/data.py`：加载 `extracts/` 中的 E/U 矩阵和 `*.info.json`，判断 `actual_tied`。
- `src/ijcai_clean/alignment.py`：token id / token string 对齐与采样逻辑。
- `src/ijcai_clean/metrics.py`：GCorr 的 cosine、euclidean、squared euclidean 统计。
- `src/ijcai_clean/experiments/`：Task1-5 的实验实现、pair 生成、CSV 汇总和 metadata 写入。
- `scripts/_cli.py`：命令行入口共享的仓库根定位、`PYTHONPATH` bootstrap 和 GPU 参数解析。
- `scripts/run_task*_*.py`：Task1-5 的主入口。
- `scripts/run_base_instruct_full_vocab_affine.py`：Base-Instruct 专用 full-vocab 仿射、`A-I` 诊断和 SVD 低秩分析入口。
- `results/task*_*/`：各任务当前结果；`task5_affine_relations/` 同时保存全量仿射结果和 Base-Instruct full-vocab 诊断结果。
- `docs/README.md`：文档索引，区分当前主线、研究结论和历史归档。
- `docs/methods_and_metrics.md`：当前实验方法、指标定义和运行口径。
- `docs/current_state.md`：仓库结构、入口脚本、结果目录和运行环境速查。
- `analysis.md`：阶段性研究判断和结果解读。
- `legacy/`：历史路径迁移来的旧脚本，可能仍依赖历史目录或完整 HF 模型目录。

## 当前推荐入口（基于 extracts）

仓库根目录有 [`configs/models.yaml`](/home/wz/projects/mypro/get_useful/configs/models.yaml)（下载与模型简称映射）、[`extracts/`](/home/wz/projects/mypro/get_useful/extracts/)（按模型简称命名的 `*.safetensors` + `*.info.json`）。

**任务一（Base–Instruct GCorr）**：

```bash
cd /path/to/get_useful
PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run_task1_base_instruct.py --devices auto
```

配置：[`../configs/base_instruct_pairs.yaml`](../configs/base_instruct_pairs.yaml)。结果：`results/task1_base_instruct/`。

**任务二（系列组内 GCorr）**：

```bash
cd /path/to/get_useful
PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run_task2_model_series.py --devices auto
```

配置：[`../configs/model_series.yaml`](../configs/model_series.yaml)。每个系列内部生成 `C(n,2)` 组 pair；若某个条目同时列出 base/instruct 候选，则使用 instruct 侧模型与其他代表模型比较。结果：`results/task2_model_series/`。

**任务三（三档规模桶跨系列 GCorr）**：

```bash
cd /path/to/get_useful
PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run_task3_cross_scale_groups.py --devices auto
```

配置：[`../configs/cross_scale_groups.yaml`](../configs/cross_scale_groups.yaml)。规模桶分为 `le_4b`、`gt_4b_lt_27b`、`ge_27b`；同样优先选择 instruct 侧代表模型，并跳过同系列 pair，避免重复任务二的系列内部计算。结果：`results/task3_cross_scale_groups/`。

**任务五（仿射 R²，复用 Task1-4 pair 并集）**：

```bash
cd /path/to/get_useful
PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run_task5_affine_relations.py --devices auto
```

配置：[`../configs/affine_pairs.yaml`](../configs/affine_pairs.yaml)。结果：`results/task5_affine_relations/summary_pair.csv`、`summary_intra_EU.csv`。

**Base-Instruct full-vocab 仿射 / A-I / SVD 诊断**：

```bash
cd /path/to/get_useful
PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run_base_instruct_full_vocab_affine.py
```

该入口只分析 Task1 的 Base-Instruct pair，按完整词表 id 对齐，不采样 token 行；输出 full-vocab 仿射 R²、`A-I` 诊断，以及 `E_instruct - E_base` / `A-I` 的 SVD 低秩能量。结果：`results/task5_affine_relations/summary_pair_base_instruct_full_vocab.csv` 与 `base_instruct_full_vocab_affine_report.md`。

**下载与抽取权重**：在仓库根目录运行 `python tools/get_model_useful.py`。

## 旧实验（Legacy）

全局几何实验 V4、跨模型仿射、矩阵对比脚本已迁至 [`legacy/`](legacy/)，便于与当前 `extracts` 流程区分：

- `legacy/exp1_global_geometry/run_exp1_v4.py`
- `legacy/exp2_affine_cross_model/run_affine_cross_model.py`
- `legacy/matrix_comparison/compare_emb_unemb_matrices.py`

这些脚本历史上默认读取完整 HF 目录与 `/root/shared-nvme/tools/models.yaml`；**论文复核可追溯**，日常计算请优先用 `ijcai_clean` 包 + `extracts/`。

## tied / untied 审计

- `is_tied`：来自 `config.tie_word_embeddings`（元数据 `*.info.json` / `config`）。
- `actual_tied`：来自 `np.allclose(E, U, rtol=1e-5, atol=1e-5)`。

历史审计 CSV 见 `results_summary/exp1_model_tag_audit.csv`；说明见 `docs/model_tag_audit.md`。当前新结果中的 tied 分组仍优先使用 `actual_tied`。

## 关键历史结果（results_summary）

- `results_summary/exp1_global_summary.csv` 等：来自原实验导出的轻量汇总，**与当前 Task1-5 / full-vocab 主线规模不一致**，只作历史参考。
- 历史摘要已移至 `docs/historical/results_summary.md`；cleanup 过程记录已移至 `docs/historical/cleanup_inventory.md`。日常阅读优先看 `docs/README.md`、`docs/methods_and_metrics.md`、`docs/current_state.md` 与 `analysis.md`。

## 依赖

`torch`、`transformers`、`safetensors`、`numpy`、`pandas`、`scipy`、`pyyaml`、`tqdm`；下载脚本另需 `modelscope`。

## 其他

文档入口见 [`docs/README.md`](docs/README.md)。迁移与排除说明见 [`MANIFEST.md`](MANIFEST.md)。仓库级状态速查见 [`docs/current_state.md`](docs/current_state.md)。实验方法和指标定义见 [`docs/methods_and_metrics.md`](docs/methods_and_metrics.md)。研究结论与异常解释见 [`analysis.md`](analysis.md)。
