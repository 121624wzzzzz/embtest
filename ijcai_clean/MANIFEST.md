# IJCAI Clean Manifest

入口与目录详解见 [`README.md`](README.md)；本文件只列出版本控制 / 发布相关的产物清单。

## 纳入版本库的源码

- `src/ijcai_clean/`：分析包（`data.py`、`alignment.py`、`metrics.py`、`paths.py`、`experiments/`）。
- `scripts/`：CLI 入口（`run_task1..5_*.py`、`run_base_instruct_full_vocab_affine.py`、`_cli.py`）。
- 仓库根 `configs/`：模型与任务配置（`models.yaml`、`base_instruct_pairs.yaml`、`model_series.yaml`、`cross_scale_groups.yaml`、`moe_cross_family.yaml`、`affine_pairs.yaml`）。

## 数据与工具（仓库根）

- `configs/models.yaml`：模型列表、`repo_id`、缓存路径。
- `downloaded_models/`：下载缓存。
- `extracts/`：标准化抽取矩阵（`*.safetensors` + `*.info.json`）。
- `tools/get_model_useful.py`、`tools/audit.py`、`tools/cleanup_redundant.py`：下载、审计与清理脚本。

## 当前结果产物

- `results/task{1,2,3,4,5}_*/`：各任务 `bootstrap_results.csv`（量大）、`summary.csv`、`metadata.json`、`generated_pairs.yaml`、`pair_plan.csv` 等。
- `results/task5_affine_subsampled/`：Task5 跨模型子采样仿射（`summary_pair.csv`、`summary_intra_EU.csv`、`metadata.json`、`base_instruct_affine_tied_report.md`）。
- `results/task6_base_instruct_full_vocab/`：Base-Instruct full-vocab 仿射 / `A-I` / SVD 诊断（`summary_pair_base_instruct_full_vocab.csv`、`base_instruct_full_vocab_affine_report.md`、`base_instruct_full_vocab_metadata.json`）。
- `audits/`：`tools/get_model_useful.py` 持续写入的模型审计 JSON。

## 文档

- `README.md`：入口与子目录功能。
- `docs/README.md`：文档索引。
- `docs/methods_and_metrics.md`：方法学与指标定义。
- `docs/model_tag_audit.md`：tied / untied 标签审计口径。
- `analysis.md`：阶段性研究结论。
- `docs/historical/`：旧实验摘要 (`results_summary.md`)、cleanup 记录 (`cleanup_inventory.md`)、被替代的旧 CSV 汇总 (`legacy_results_summary/`)。

## 归档（不在本目录）

- 仓库根 `archive/ijcai_cleanup_2026-05-10/`：旧 V4 实验代码、`source_notes/`、结果快照 bundle。

## 已排除（体积或临时）

- 完整权重分片、HF 全缓存：体积大，默认不纳入本目录文档关注点。
- `__pycache__/`、`*.log`、`.ipynb_checkpoints/` 等：见 `.gitignore`。
