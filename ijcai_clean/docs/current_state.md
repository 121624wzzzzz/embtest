# 仓库与实验状态（整理时审计）

## 数据层

- **模型注册与下载配置**：仓库根目录 [`models.yaml`](/home/wz/projects/mypro/get_useful/models.yaml)（`config.cache_dir` / `extracts_dir`、`model_groups`、`model_repo_ids`）。
- **下载缓存**：`downloaded_models/<owner>/<repo>/`（部分仓名中 `.` 会落成 `___`）。
- **标准化抽取**：`extracts/<模型简称>.safetensors` + `extracts/<模型简称>.info.json`（与 [`models.yaml`](/home/wz/projects/mypro/get_useful/models.yaml) 中简称一致）。

## 分析层（IJCAI clean）

- **任务一结果**：[`ijcai_clean/results/task1_base_instruct/`](/home/wz/projects/mypro/get_useful/ijcai_clean/results/task1_base_instruct/) — `bootstrap_results.csv`、`summary.csv`、`metadata.json`。
- **任务一配置**：[`ijcai_clean/configs/base_instruct_pairs.yaml`](/home/wz/projects/mypro/get_useful/ijcai_clean/configs/base_instruct_pairs.yaml)（应与 `summary.csv` 中 pair 一致）。
- **任务二系列**：[`ijcai_clean/configs/model_series.yaml`](/home/wz/projects/mypro/get_useful/ijcai_clean/configs/model_series.yaml)（与根目录 `models.yaml` 的 `model_groups` 对齐；组内生成 `C(n,2)` pair，base/instruct 同时存在时选 instruct 侧）。
- **入口脚本**：[`ijcai_clean/scripts/run_task1_base_instruct.py`](/home/wz/projects/mypro/get_useful/ijcai_clean/scripts/run_task1_base_instruct.py)、[`ijcai_clean/scripts/run_task2_model_series.py`](/home/wz/projects/mypro/get_useful/ijcai_clean/scripts/run_task2_model_series.py)，依赖包路径：[`ijcai_clean/src/`](/home/wz/projects/mypro/get_useful/ijcai_clean/src/) 下的 `ijcai_clean` 包。
- **方法与指标文档**：[`ijcai_clean/docs/methods_and_metrics.md`](/home/wz/projects/mypro/get_useful/ijcai_clean/docs/methods_and_metrics.md)，后续新增实验方法、指标或默认参数时优先更新。

## 参考层（旧实验）

- **Legacy**：[`ijcai_clean/legacy/`](/home/wz/projects/mypro/get_useful/ijcai_clean/legacy/) — 原 `run_exp1_v4.py`、`run_affine_cross_model.py` 等，路径与数据接口以历史环境为准，**复现请优先用 `extracts` + 新包**。

## 工具层

- **下载/抽取**：[`tools/download/get_model_useful.py`](/home/wz/projects/mypro/get_useful/tools/download/get_model_useful.py)；根目录 `get_model_useful.py` 为薄包装，保持旧命令可用。
- **审计/清理**：[`tools/audit/`](/home/wz/projects/mypro/get_useful/tools/audit/)、[`tools/cleanup/`](/home/wz/projects/mypro/get_useful/tools/cleanup/)。

## 环境与已知提示

- **Python**：建议使用含 `torch`、`transformers`、`safetensors`、`yaml`、`numpy` 等依赖的环境（例如 `conda env wzall`）。任务一汇总使用标准库 `csv`，不强制 `pandas`。
- **仓库根定位**：任务入口脚本会向上查找 `models.yaml`；若目录结构特殊，可设环境变量 `REPO_ROOT`。
- **硬件**：任务一/二为 GPU 并行 bootstrap；无 GPU 时脚本可退化（较慢）。
- **metadata**：历史运行可能在 `metadata.json` 中记录绝对路径；整理后以仓库相对路径与 `git_commit`（若可用）为准。

## 整理日期

- 文档生成于仓库结构整理批次；后续变更请在本文件追加小节。
