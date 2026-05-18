# IJCAI Clean Manifest

## 当前主代码（extracts 流程）

- `src/ijcai_clean/`：从 `extracts/*.safetensors` 与 `*.info.json` 加载 E/U，支持 GCorr、Task1-5 任务实现。
- `scripts/run_task1_base_instruct.py`：任务一入口（需 `PYTHONPATH=ijcai_clean/src`）。
- `scripts/run_task2_model_series.py`：任务二入口，按系列组内生成 pair 后复用 GCorr runner。
- `scripts/run_task3_cross_scale_groups.py`：任务三入口，按三档规模桶生成跨系列 pair 后复用 GCorr runner。
- `scripts/run_task4_moe_cross_family.py`：任务四入口，生成 MoE 跨系列 pair 后复用 GCorr runner。
- `scripts/run_task5_affine_relations.py`：任务五入口，对 Task1-4 pair 并集做仿射关系分析。
- `scripts/run_base_instruct_full_vocab_affine.py`：Base-Instruct full-vocab 仿射、`A-I` 诊断和 SVD 能量分析入口。
- `../configs/`：全局模型配置与任务配置（`models.yaml`、`base_instruct_pairs.yaml`、`model_series.yaml`、`cross_scale_groups.yaml` 等）。

## 数据与下载工具（仓库根目录）

- `../configs/models.yaml`：模型列表、`repo_id`、缓存路径。
- `downloaded_models/`：下载缓存。
- `extracts/`：标准化抽取矩阵。
- `tools/get_model_useful.py`、`tools/audit.py`、`tools/cleanup_redundant.py`：下载、审计与清理脚本。

## Legacy（参考 / 历史复现）

- `legacy/exp1_global_geometry/`：实验 1 V4 主入口与提取脚本（旧路径与整仓加载）。
- `legacy/exp2_affine_cross_model/`：实验 2 仿射与 tied/untied 分析。
- `legacy/matrix_comparison/`：E/U 矩阵对比脚本。

## 文档与摘要

- `docs/README.md`：文档索引，区分当前主线、研究分析和历史归档。
- `docs/current_state.md`：仓库与实验接口说明。
- `docs/methods_and_metrics.md`：当前方法、指标和运行口径。
- `docs/model_tag_audit.md`：tied / untied 标签审计口径。
- `analysis.md`：当前研究结论备忘，包含 Base-Instruct full-vocab、Gemma 异常和 SVD 低秩分析。
- `docs/historical/`：历史结果摘要与 cleanup 清单。
- `results_summary/`：从历史 CSV 抽取的轻量摘要，仅用于追溯旧实验。

## 已排除（体积或临时）

- `archive/v1_preliminary/`：旧探索代码（若存在）不在主路径。
- 完整权重分片、HF 全缓存：体积大，默认不纳入本目录文档关注点。
- `.ipynb_checkpoints`、临时日志等。
