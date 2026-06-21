# Cross-model Embedding Geometry

本目录保留当前跨模型 embedding / unembedding 几何分析代码、轻量结果和文档。大权重、下载缓存和标准化抽取矩阵仍在仓库根目录。

## 快速入口

当前推荐使用统一派发脚本：

```bash
cd /path/to/get_useful
PYTHONPATH=cross_model_geometry/src python cross_model_geometry/scripts/run.py list
PYTHONPATH=cross_model_geometry/src python cross_model_geometry/scripts/run.py task1 --devices auto
PYTHONPATH=cross_model_geometry/src python cross_model_geometry/scripts/run.py task5 --devices auto
PYTHONPATH=cross_model_geometry/src python cross_model_geometry/scripts/run.py task6
```

旧入口仍保留为 thin wrapper，方便复用已有命令：

```bash
PYTHONPATH=cross_model_geometry/src python cross_model_geometry/scripts/run_task1_base_instruct.py --devices auto
PYTHONPATH=cross_model_geometry/src python cross_model_geometry/scripts/run_task2_model_series.py --devices auto
PYTHONPATH=cross_model_geometry/src python cross_model_geometry/scripts/run_task3_cross_scale_groups.py --devices auto
PYTHONPATH=cross_model_geometry/src python cross_model_geometry/scripts/run_task4_moe_cross_family.py --devices auto
PYTHONPATH=cross_model_geometry/src python cross_model_geometry/scripts/run_task5_affine_relations.py --devices auto
PYTHONPATH=cross_model_geometry/src python cross_model_geometry/scripts/run_task6_base_instruct_full_vocab_affine.py
```

下载与抽取权重在仓库根目录运行：

```bash
python tools/get_model_useful.py
```

## 数据布局

| 路径 | 职责 |
|---|---|
| `../configs/` | 全局模型配置与任务配置：`models.yaml`、`base_instruct_pairs.yaml`、`model_series.yaml`、`cross_scale_groups.yaml`、`moe_cross_family.yaml`、`affine_pairs.yaml` |
| `../downloaded_models/` | HuggingFace / ModelScope 下载缓存 |
| `../extracts/` | 标准化抽取矩阵：`<model>.safetensors` + `<model>.info.json` |
| `src/cross_model_geometry/` | 分析代码包：加载 E/U、token 对齐、GCorr、Task1-6 实现 |
| `scripts/` | CLI 入口：`run.py` 与向后兼容的 thin wrapper |
| `results/` | 当前实验输出 |
| `audits/` | `tools/get_model_useful.py` 写入的模型审计 JSON |
| `docs/` | 方法学文档与历史结果归档 |
| `analysis.md` | 阶段性研究结论与异常解释 |

## 当前任务

| 任务 | 配置 / 来源 | 主输出 |
|---|---|---|
| Task1 Base/Instruct GCorr | `../configs/base_instruct_pairs.yaml` | `results/task1_base_instruct/summary.csv` |
| Task2 Model-Series GCorr | `../configs/model_series.yaml` | `results/task2_model_series/summary.csv` |
| Task3 Cross-Scale GCorr | `../configs/cross_scale_groups.yaml` + `../configs/model_series.yaml` | `results/task3_cross_scale_groups/summary.csv` |
| Task4 MoE Cross-Family GCorr | `../configs/moe_cross_family.yaml` + `../configs/model_series.yaml` | `results/task4_moe_cross_family/summary.csv` |
| Task5 Affine Relations（子采样） | `../configs/affine_pairs.yaml` | `results/task5_affine_subsampled/summary_pair.csv`、`summary_intra_EU.csv` |
| Task6 Base-Instruct full-vocab affine / SVD | Task1 Base/Instruct pair | `results/task6_base_instruct_full_vocab/summary_pair_base_instruct_full_vocab.csv` |

详细实验设置、指标定义和结果口径见 `docs/methods_and_metrics.md`。研究性解释、Gemma 异常分析和低秩结论见 `analysis.md`。

## 版本化策略

- 跟踪源码、配置、`summary.csv`、`metadata.json`、`pair_plan.csv`、`generated_pairs.yaml`、轻量报告和历史轻量 CSV。
- 不再跟踪 `results/**/bootstrap_results.csv`、`*.log`、`__pycache__/` 和大权重 / 缓存；bootstrap 明细体量大，必要时由配置、seed 和 runner 重算。
- `results/task5_affine_subsampled/` 保存 Task5 子采样仿射结果；`results/task6_base_instruct_full_vocab/` 保存 Base-Instruct full-vocab 诊断结果。
- `docs/historical/legacy_results_summary/` 只用于追溯旧实验 1/2，不代表当前 Task1-6 覆盖范围。

## 环境提示

建议使用包含 `torch`、`transformers`、`safetensors`、`numpy`、`pandas`、`scipy`、`pyyaml`、`tqdm` 的 Python 环境（例如本机 conda 环境 `wzall`）。Task1-4 为 GPU 并行 bootstrap；Task5 与 Task6 会处理大矩阵，建议使用 CUDA。

入口脚本会向上查找仓库根目录的 `configs/models.yaml`；目录结构特殊时可设置 `REPO_ROOT`。

## 历史归档

旧 IJCAI V4 实验代码（`run_exp1_v4.py`、`run_affine_cross_model.py` 等）已归档至仓库根目录 `archive/ijcai_cleanup_2026-05-10/`。旧实验摘要见 `docs/historical/results_summary.md`，旧轻量汇总 CSV 见 `docs/historical/legacy_results_summary/`。

## 文档入口

- `docs/README.md`：文档索引。
- `docs/methods_and_metrics.md`：方法、指标、任务设置和 tied / untied 口径。
- `analysis.md`：当前研究判断和结果解读。
