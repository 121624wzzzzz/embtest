# 仓库与实验状态

本文档是当前 `ijcai_clean` 主线的状态速查；历史 cleanup 过程见 `historical/cleanup_inventory.md`，旧实验摘要见 `historical/results_summary.md`。

## 数据层

- **模型注册与下载配置**：仓库根目录 [`configs/models.yaml`](/home/wz/projects/mypro/get_useful/configs/models.yaml)（`config.cache_dir` / `extracts_dir`、`model_groups`、`model_repo_ids`）。
- **下载缓存**：`downloaded_models/<owner>/<repo>/`（部分仓名中 `.` 会落成 `___`）。
- **标准化抽取**：`extracts/<模型简称>.safetensors` + `extracts/<模型简称>.info.json`（与 [`configs/models.yaml`](/home/wz/projects/mypro/get_useful/configs/models.yaml) 中简称一致）。

## 分析层（IJCAI clean）

- **任务一结果**：[`ijcai_clean/results/task1_base_instruct/`](/home/wz/projects/mypro/get_useful/ijcai_clean/results/task1_base_instruct/) — `bootstrap_results.csv`、`summary.csv`、`metadata.json`。
- **任务一配置**：[`configs/base_instruct_pairs.yaml`](/home/wz/projects/mypro/get_useful/configs/base_instruct_pairs.yaml)（应与 `summary.csv` 中 pair 一致）。
- **任务二系列**：[`configs/model_series.yaml`](/home/wz/projects/mypro/get_useful/configs/model_series.yaml)（与 `configs/models.yaml` 的 `model_groups` 对齐；组内生成 `C(n,2)` pair，base/instruct 同时存在时选 instruct 侧）。
- **任务三跨规模**：[`configs/cross_scale_groups.yaml`](/home/wz/projects/mypro/get_useful/configs/cross_scale_groups.yaml)（三档规模桶：`le_4b`、`gt_4b_lt_27b`、`ge_27b`；只生成跨系列 pair，避免重复任务二同系列内部计算）。
- **任务四 MoE 跨系列**：[`configs/moe_cross_family.yaml`](/home/wz/projects/mypro/get_useful/configs/moe_cross_family.yaml)（若存在）与 `configs/model_series.yaml` 共同生成 MoE / dense 跨系列 pair。
- **任务五仿射**：[`configs/affine_pairs.yaml`](/home/wz/projects/mypro/get_useful/configs/affine_pairs.yaml)，读取 Task1-4 pair 并集，输出 `results/task5_affine_relations/summary_pair.csv` 与 `summary_intra_EU.csv`。
- **Base-Instruct full-vocab 诊断**：[`ijcai_clean/scripts/run_base_instruct_full_vocab_affine.py`](/home/wz/projects/mypro/get_useful/ijcai_clean/scripts/run_base_instruct_full_vocab_affine.py)，对 Task1 Base-Instruct pair 按完整词表 id 对齐，输出仿射 R²、`A-I` 诊断、`E_instruct - E_base` / `A-I` 的 SVD 能量与 `energy@1%h/5%h/10%h`。
- **入口脚本**：[`ijcai_clean/scripts/run_task1_base_instruct.py`](/home/wz/projects/mypro/get_useful/ijcai_clean/scripts/run_task1_base_instruct.py)、[`run_task2_model_series.py`](/home/wz/projects/mypro/get_useful/ijcai_clean/scripts/run_task2_model_series.py)、[`run_task3_cross_scale_groups.py`](/home/wz/projects/mypro/get_useful/ijcai_clean/scripts/run_task3_cross_scale_groups.py)、[`run_task4_moe_cross_family.py`](/home/wz/projects/mypro/get_useful/ijcai_clean/scripts/run_task4_moe_cross_family.py)、[`run_task5_affine_relations.py`](/home/wz/projects/mypro/get_useful/ijcai_clean/scripts/run_task5_affine_relations.py)；依赖包路径：[`ijcai_clean/src/`](/home/wz/projects/mypro/get_useful/ijcai_clean/src/) 下的 `ijcai_clean` 包。
- **方法与指标文档**：[`ijcai_clean/docs/methods_and_metrics.md`](/home/wz/projects/mypro/get_useful/ijcai_clean/docs/methods_and_metrics.md)，后续新增实验方法、指标或默认参数时优先更新。
- **研究分析备忘**：[`ijcai_clean/analysis.md`](/home/wz/projects/mypro/get_useful/ijcai_clean/analysis.md)，记录当前结果解读、Gemma 异常判断和 SVD 低秩结论。

## 参考层（旧实验）

- **Legacy**：[`ijcai_clean/legacy/`](/home/wz/projects/mypro/get_useful/ijcai_clean/legacy/) — 原 `run_exp1_v4.py`、`run_affine_cross_model.py` 等，路径与数据接口以历史环境为准，**复现请优先用 `extracts` + 新包**。
- **文档归档**：[`ijcai_clean/docs/historical/`](/home/wz/projects/mypro/get_useful/ijcai_clean/docs/historical/) — 旧结果摘要和 2026-05-10 cleanup 清单，仅用于追溯。

## 工具层

- **下载/抽取**：[`tools/get_model_useful.py`](/home/wz/projects/mypro/get_useful/tools/get_model_useful.py)。
- **审计/清理**：[`tools/audit.py`](/home/wz/projects/mypro/get_useful/tools/audit.py)、[`tools/cleanup_redundant.py`](/home/wz/projects/mypro/get_useful/tools/cleanup_redundant.py)。

## 环境与已知提示

- **Python**：建议使用含 `torch`、`transformers`、`safetensors`、`yaml`、`numpy` 等依赖的环境（例如 conda 环境 `wzall`）。任务汇总尽量使用标准库 `csv`，不强制 `pandas`。
- **仓库根定位**：任务入口脚本会向上查找 `configs/models.yaml`；若目录结构特殊，可设环境变量 `REPO_ROOT`。
- **硬件**：Task1-4 为 GPU 并行 bootstrap；Task5 和 full-vocab 诊断会处理大矩阵，建议使用 CUDA。
- **metadata**：历史运行可能在 `metadata.json` 中记录绝对路径；整理后以仓库相对路径与 `git_commit`（若可用）为准。

## 维护记录

- 2026-05-10：仓库结构整理批次生成初版。
- 2026-05-18：补充 Task4/Task5、Base-Instruct full-vocab 诊断、`analysis.md` 和 `docs/archive/` 口径。
