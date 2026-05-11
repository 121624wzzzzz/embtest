# IJCAI Clean 整理清单

本文档用于在归档旧版本代码前确认 `ijcai_clean` 的新旧文件关系。

执行状态：第一版归档已在 `2026-05-10` 执行，归档目录为 `archive/ijcai_cleanup_2026-05-10/`。

## 时间与状态口径

- 当前主线整理起点：`2026-05-05` 的提交 `bc11913` 首次加入 `ijcai_clean`。
- 当前新增主线：`2026-05-07` 之后新增 Task 3/4/5、根目录 `configs/`、根目录 `tools/`。
- 当前文档/入口刷新：`2026-05-10` 更新 README、MANIFEST、Task CLI、`paths.py`。
- 旧实验来源：`2026-01-28` 到 `2026-03-16` 的 legacy 代码与 source notes。

## 新主线文件

这些文件是后续重构应围绕的主线。

| 路径 | 类型 | 时间范围 | 关系 | 建议动作 |
|---|---|---:|---|---|
| `ijcai_clean/src/ijcai_clean/` | 当前 Python 包 | 2026-05-03 到 2026-05-10 | 当前 Task1-5 的核心实现 | 保留并重构 |
| `ijcai_clean/scripts/run_task1_base_instruct.py` | 当前 CLI | 2026-05-10 | 调用 `experiments/task1.py` | 保留，后续抽公共 CLI |
| `ijcai_clean/scripts/run_task2_model_series.py` | 当前 CLI | 2026-05-10 | 调用 `experiments/task2.py` | 保留，后续抽公共 CLI |
| `ijcai_clean/scripts/run_task3_cross_scale_groups.py` | 当前 CLI | 2026-05-10，未跟踪 | 新增 Task3 入口 | 保留，纳入主线 |
| `ijcai_clean/scripts/run_task4_moe_cross_family.py` | 当前 CLI | 2026-05-10，未跟踪 | 新增 Task4 入口 | 保留，纳入主线 |
| `ijcai_clean/scripts/run_task5_affine_relations.py` | 当前 CLI | 2026-05-10，未跟踪 | 新增 Task5 入口 | 保留，纳入主线 |
| `configs/` | 当前权威配置 | 2026-05-07 到 2026-05-10，未跟踪 | 替代旧 `ijcai_clean/configs/` 与根目录散落配置 | 保留，纳入主线 |
| `tools/README.md`、`tools/audit.py`、`tools/cleanup_redundant.py`、`tools/get_model_useful.py`、`tools/paths.py` | 当前工具层 | 2026-05-10 | 替代根目录散落脚本和旧 `tools/audit|cleanup|download` 层级 | 保留，纳入主线 |
| `ijcai_clean/README.md`、`MANIFEST.md`、`docs/current_state.md` | 当前说明 | 2026-05-10 | 描述当前 extracts + Task1-5 口径 | 保留并持续更新 |

## 当前结果文件

这些是当前实验运行产物，不是源码；需要决定是否纳入版本库或移到归档/发布包。

| 路径 | 类型 | 时间范围 | 关系 | 建议动作 |
|---|---|---:|---|---|
| `ijcai_clean/results/task1_base_instruct/` | 当前 Task1 结果 | 2026-05-07，已跟踪但有修改 | Task1 bootstrap/summary/metadata | 保留结果，考虑只跟踪 summary/metadata |
| `ijcai_clean/results/task2_model_series/` | 当前 Task2 结果 | 2026-05-07，已跟踪但有修改 | Task2 generated pairs + Task1 runner 产物 | 保留结果，考虑只跟踪 summary/metadata/pair plan |
| `ijcai_clean/results/task3_cross_scale_groups/` | 当前 Task3 结果 | 2026-05-07，未跟踪 | 新增 Task3 输出 | 保留或归入结果发布包 |
| `ijcai_clean/results/task4_moe_cross_family/` | 当前 Task4 结果 | 2026-05-07，未跟踪 | 新增 Task4 输出 | 保留或归入结果发布包 |
| `ijcai_clean/results/task5_affine_relations/` | 当前 Task5 结果 | 2026-05-07，未跟踪 | 新增仿射实验输出 | 保留或归入结果发布包 |
| `ijcai_clean/results_summary/` | 历史轻量汇总 | 2026-01-29 到 2026-05-07 | 旧 exp1/exp2 摘要与审计，部分与当前 Task 规模不一致 | 保留为历史摘要，标注非主线 |

## 旧版本与快照

这些是臃肿来源，建议归档或从主线移除。

| 路径 | 类型 | 时间范围 | 与新文件关系 | 建议动作 |
|---|---|---:|---|---|
| `ijcai_clean/legacy/` | 旧实验复现代码 | 2026-01-28 到 2026-03-16，另有 pycache 到 2026-05-04 | 旧 HF/历史路径实现；当前主线已改为 `extracts/` + `src/ijcai_clean` | 已归档到 `archive/ijcai_cleanup_2026-05-10/legacy/` |
| `ijcai_clean/docs/source_notes/` | 旧实验笔记 | 2026-01-29 | 原始分析/设计草稿；当前结论应以 README/current docs 为准 | 已归档到 `archive/ijcai_cleanup_2026-05-10/docs/source_notes/` |
| `ijcai_clean/results/s1_four_tasks_bundle/` | 结果快照包 | 2026-05-07，未跟踪 | 内含重复 scripts/configs/docs/result；task1-4 summary 与当前结果相同，但 scripts/configs/docs 已与当前主线不同 | 已归档到 `archive/ijcai_cleanup_2026-05-10/results/s1_four_tasks_bundle/` |
| `ijcai_clean/scripts/gen_latex_table.py` | 旧报告脚本 | 2026-05-07 | 仍读取旧 `results/exp1_global_v4_complete.csv`，不适配当前 Task1-5 结果目录 | 已归档到 `archive/ijcai_cleanup_2026-05-10/scripts/gen_latex_table.py` |
| `ijcai_clean/scripts/run_task2_cron.sh`、`run_task2_cron_once.sh` | 本机调度脚本 | 2026-05-07 | 定时任务封装，可能包含本机路径假设，不属于通用 CLI | 已归档到 `archive/ijcai_cleanup_2026-05-10/scripts/` |
| `ijcai_clean/**/__pycache__/` | Python 缓存 | 2026-05-03 到 2026-05-10 | 自动生成，无源码价值 | 已删除 |

## 已迁移或替代的旧位置

| 旧路径/状态 | 新路径 | 关系 | 建议动作 |
|---|---|---|---|
| `ijcai_clean/configs/base_instruct_pairs.yaml`、`ijcai_clean/configs/model_series.yaml` 当前为删除 | `configs/base_instruct_pairs.yaml`、`configs/model_series.yaml` | 配置上移到仓库根，作为权威配置 | 接受删除旧位置 |
| 根目录 `models.yaml` 当前为删除 | `configs/models.yaml` | 模型注册配置统一到 `configs/` | 接受删除旧位置 |
| 根目录 `get_model_useful.py`、`audit_downloads.py`、`audit_tied_redundancy.py`、`cleanup_redundant.py` 当前为删除 | `tools/get_model_useful.py`、`tools/audit.py`、`tools/cleanup_redundant.py` | 工具脚本集中到 `tools/` | 接受删除旧散落脚本 |
| `tools/audit/*`、`tools/cleanup/*`、`tools/download/*` 当前为删除 | `tools/*.py` | 工具层级压平 | 若无外部调用依赖，接受删除 |
| 根目录 `all_models_summary.json`、`model_tied_matrix_summary.json`、`model_tied_untied_audit.json` 当前为删除 | `ijcai_clean/results_summary/audits/` | 审计 JSON 移入 IJCAI 汇总目录 | 接受删除旧散落 JSON |

## 快照一致性检查

`ijcai_clean/results/s1_four_tasks_bundle/` 与当前主线对比：

- `task1_base_instruct/summary.csv`、`task2_model_series/summary.csv`、`task3_cross_scale_groups/summary.csv`、`task4_moe_cross_family/summary.csv` 与当前对应结果一致。
- bundle 内 `scripts/run_task1..4.py` 与当前 `ijcai_clean/scripts/` 不一致。
- bundle 内 `configs/*.yaml` 与当前根目录 `configs/` 不一致。
- bundle 内 `docs/methods_and_metrics.md` 与当前文档不一致。

因此 bundle 更适合作为某次结果发布快照，不适合作为主线源码的一部分。

## 建议归档批次

1. 删除本地缓存：`ijcai_clean/**/__pycache__/`。
2. 归档旧实现：`ijcai_clean/legacy/`、`ijcai_clean/docs/source_notes/`。
3. 归档发布快照：`ijcai_clean/results/s1_four_tasks_bundle/`。
4. 归档或重写旧报告/调度脚本：`gen_latex_table.py`、`run_task2_cron*.sh`。
5. 接受已迁移路径的删除：旧 `ijcai_clean/configs/`、根目录散落配置/工具/审计 JSON。
6. 决定结果文件版本策略：是否继续跟踪完整 `bootstrap_results.csv` 和 `run.log`，或只跟踪 `summary.csv`、`metadata.json`、`pair_plan.csv`。

## 待确认

- 是否继续版本化大体量 CSV，例如 `bootstrap_results.csv`。
- 是否将 `archive/ijcai_cleanup_2026-05-10/` 纳入版本库，还是只作为本地临时归档。
- 是否需要将 `gen_latex_table.py` 基于当前 Task1-5 summary 重写为新的报告工具。
