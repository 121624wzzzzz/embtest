# IJCAI Clean Manifest

## 当前主代码（extracts 流程）

- `src/ijcai_clean/`：从 `extracts/*.safetensors` 与 `*.info.json` 加载 E/U、GCorr、任务一/二。
- `scripts/run_task1_base_instruct.py`：任务一入口（需 `PYTHONPATH=ijcai_clean/src`）。
- `scripts/run_task2_model_series.py`：任务二入口，按系列组内生成 pair 后复用 GCorr runner。
- `configs/base_instruct_pairs.yaml`、`configs/model_series.yaml`：任务配置（任务一/二）。

## 数据与下载工具（仓库根目录）

- `models.yaml`：模型列表、`repo_id`、缓存路径。
- `downloaded_models/`：下载缓存。
- `extracts/`：标准化抽取矩阵。
- `tools/download/`、`tools/audit/`、`tools/cleanup/`：下载与审计脚本；根目录同名脚本为兼容包装。

## Legacy（参考 / 历史复现）

- `legacy/exp1_global_geometry/`：实验 1 V4 主入口与提取脚本（旧路径与整仓加载）。
- `legacy/exp2_affine_cross_model/`：实验 2 仿射与 tied/untied 分析。
- `legacy/matrix_comparison/`：E/U 矩阵对比脚本。

## 文档与摘要

- `docs/source_notes/`：原始实验设计与结果解释副本。
- `results_summary/`：从历史 CSV 抽取的轻量摘要。
- `docs/current_state.md`：仓库与实验接口说明。

## 已排除（体积或临时）

- `archive/v1_preliminary/`：旧探索代码（若存在）不在主路径。
- 完整权重分片、HF 全缓存：体积大，默认不纳入本目录文档关注点。
- `.ipynb_checkpoints`、临时日志等。
