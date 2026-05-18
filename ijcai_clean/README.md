# IJCAI Embedding Geometry（clean 子目录）

本目录整理自历史路径 `/root/shared-nvme/ijcai`，目标是把几何分析代码、轻量结果与文档留在可读的 `ijcai_clean/` 下；**大权重与缓存仍在仓库根目录**（见下文「数据布局」）。

## 目录结构（整理后）

| 路径 | 职责 |
|------|------|
| 根目录 [`configs/`](../configs) | 全局模型配置与任务配置：`models.yaml`、`base_instruct_pairs.yaml`、`model_series.yaml`、`cross_scale_groups.yaml`、`moe_cross_family.yaml`、`affine_pairs.yaml` |
| `src/ijcai_clean/` | 分析代码包：加载 `extracts/`、对齐 tokenizer、GCorr、Task1-5 与 full-vocab 仿射诊断 |
| `scripts/` | CLI 入口：统一派发入口 `run.py {task1..6\|list}`，以及 `run_task1..5_*.py`、`run_base_instruct_full_vocab_affine.py` 等 thin wrapper |
| `results/` | 各任务实验输出（`task1_base_instruct/`、`task2_model_series/`、`task3_cross_scale_groups/`、`task4_moe_cross_family/`、`task5_affine_subsampled/`、`task6_base_instruct_full_vocab/`） |
| `audits/` | 模型审计 JSON（由 `tools/get_model_useful.py` 写入：`all_models_summary.json` 等） |
| `docs/` | 方法学、状态、审计说明；历史文档在 `docs/historical/` |
| `analysis.md` | 阶段性研究结论：Base-Instruct full-vocab、Gemma 异常、SVD 低秩分析 |

> 旧 IJCAI V4 实验代码（`run_exp1_v4.py`、`run_affine_cross_model.py` 等）已归档至仓库根目录 `archive/ijcai_cleanup_2026-05-10/`，复现旧实验请到那里查看。

## 子目录功能

- `src/ijcai_clean/data.py`：加载 `extracts/` 中的 E/U 矩阵和 `*.info.json`，判断 `actual_tied`。
- `src/ijcai_clean/alignment.py`：token id / token string 对齐与采样逻辑。
- `src/ijcai_clean/metrics.py`：GCorr 的 cosine、euclidean、squared euclidean 统计。
- `src/ijcai_clean/experiments/`：Task1-5 的实验实现、pair 生成、CSV 汇总和 metadata 写入。
- `scripts/_cli.py`：命令行入口共享的仓库根定位、`PYTHONPATH` bootstrap 和 GPU 参数解析。
- `scripts/run_task*_*.py`：Task1-5 的主入口。
- `scripts/run_base_instruct_full_vocab_affine.py`：Base-Instruct 专用 full-vocab 仿射、`A-I` 诊断和 SVD 低秩分析入口。
- `results/task*_*/`：各任务当前结果。Task5 子采样仿射在 `task5_affine_subsampled/`；Base-Instruct full-vocab 仿射 / `A-I` / SVD 诊断在 `task6_base_instruct_full_vocab/`。
- `docs/README.md`：文档索引，区分当前主线、研究结论和历史归档。
- `docs/methods_and_metrics.md`：实验方法、指标定义和运行口径。
- `docs/current_state.md`：仓库结构、入口脚本、结果目录和运行环境速查。
- `analysis.md`：阶段性研究判断和结果解读。

## 当前推荐入口（基于 extracts）

仓库根目录有 [`configs/models.yaml`](/home/wz/projects/mypro/get_useful/configs/models.yaml)（下载与模型简称映射）、[`extracts/`](/home/wz/projects/mypro/get_useful/extracts/)（按模型简称命名的 `*.safetensors` + `*.info.json`）。

推荐用统一派发入口（`run.py list` 查看可用任务，`run.py taskN --help` 看各任务参数）：

```bash
cd /path/to/get_useful
PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run.py task1 --devices auto
PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run.py task5 --devices auto
PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run.py task6
```

旧 `run_task*_*.py` 与 `run_base_instruct_full_vocab_affine.py` 入口仍可向后兼容运行：

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

**任务四（MoE 跨系列 GCorr）**：

```bash
cd /path/to/get_useful
PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run_task4_moe_cross_family.py --devices auto
```

配置：[`../configs/moe_cross_family.yaml`](../configs/moe_cross_family.yaml) 与 [`../configs/model_series.yaml`](../configs/model_series.yaml)。用于生成 MoE / dense 代表模型之间的跨 family pair。结果：`results/task4_moe_cross_family/`。

**任务五（仿射 R²，复用 Task1-4 pair 并集）**：

```bash
cd /path/to/get_useful
PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run_task5_affine_relations.py --devices auto
```

配置：[`../configs/affine_pairs.yaml`](../configs/affine_pairs.yaml)。结果：`results/task5_affine_subsampled/summary_pair.csv`、`summary_intra_EU.csv`。

**Base-Instruct full-vocab 仿射 / A-I / SVD 诊断**：

```bash
cd /path/to/get_useful
PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run_base_instruct_full_vocab_affine.py
```

该入口只分析 Task1 的 Base-Instruct pair，按完整词表 id 对齐，不采样 token 行；输出 full-vocab 仿射 R²、`A-I` 诊断，以及 `E_instruct - E_base` / `A-I` 的 SVD 低秩能量。结果：`results/task6_base_instruct_full_vocab/summary_pair_base_instruct_full_vocab.csv` 与 `base_instruct_full_vocab_affine_report.md`。

**下载与抽取权重**：在仓库根目录运行 `python tools/get_model_useful.py`。

## tied / untied 审计

- `is_tied`：来自 `config.tie_word_embeddings`（元数据 `*.info.json` / `config`）。
- `actual_tied`：来自 `np.allclose(E, U, rtol=1e-5, atol=1e-5)`。

当前新结果中的 tied 分组优先使用 `actual_tied`。详细口径见 `docs/model_tag_audit.md`，历史审计 CSV 见 `docs/historical/legacy_results_summary/exp1_model_tag_audit.csv`。

## 历史结果与归档

- `docs/historical/legacy_results_summary/`：被 Task1-5 替代的旧实验 1/2 轻量汇总 CSV，**与当前主线规模不一致**，只作历史参考。
- `docs/historical/results_summary.md`、`docs/historical/cleanup_inventory.md`：旧实验摘要与 2026-05-10 cleanup 清单。
- 仓库根 `archive/ijcai_cleanup_2026-05-10/`：归档的旧 V4 实验代码（`legacy/`、`source_notes/`、结果快照 bundle 等）。
- 日常阅读优先看 `docs/README.md`、`docs/methods_and_metrics.md` 与 `analysis.md`。

## 依赖

`torch`、`transformers`、`safetensors`、`numpy`、`pandas`、`scipy`、`pyyaml`、`tqdm`；下载脚本另需 `modelscope`。

## 其他

文档入口见 [`docs/README.md`](docs/README.md)。迁移与排除说明见 [`MANIFEST.md`](MANIFEST.md)。仓库级状态速查见 [`docs/current_state.md`](docs/current_state.md)。实验方法和指标定义见 [`docs/methods_and_metrics.md`](docs/methods_and_metrics.md)。研究结论与异常解释见 [`analysis.md`](analysis.md)。
