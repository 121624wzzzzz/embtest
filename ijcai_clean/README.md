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
| `docs/` | 文档与说明 |

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

历史审计 CSV 见 `results_summary/exp1_model_tag_audit.csv`；说明见 `docs/model_tag_audit.md`（其中部分「src/…」路径请对应到 `legacy/…`）。

## 关键历史结果（results_summary）

- `results_summary/exp1_global_summary.csv` 等：来自原实验导出的轻量汇总，**与当前 `task1` 的 28 组 pair 规模可能不一致**，以 `results/task1_base_instruct/summary.csv` 为准。

## 依赖

`torch`、`transformers`、`safetensors`、`numpy`、`pandas`、`scipy`、`pyyaml`、`tqdm`；下载脚本另需 `modelscope`。

## 其他

迁移与排除说明见 [`MANIFEST.md`](MANIFEST.md)。仓库级状态速查见 [`docs/current_state.md`](docs/current_state.md)。实验方法和指标定义见 [`docs/methods_and_metrics.md`](docs/methods_and_metrics.md)，后续新增任务或指标时优先更新该文档。
