# get_useful：嵌入 / 解嵌矩阵下载与分析

## 布局

| 位置 | 说明 |
|------|------|
| [`configs/`](configs) | 全局模型配置 `models.yaml` 与任务配置 YAML |
| [`downloaded_models/`](downloaded_models) | 下载缓存 |
| [`extracts/`](extracts) | 每模型 `*.safetensors` + `*.info.json` |
| [`tools/`](tools) | 下载、审计、清理 |
| [`docs/`](docs) | 全仓口径、排除规则与特殊案例统一说明 |
| [`ijcai_clean/`](ijcai_clean) | 当前主线分析子项目：包 `ijcai_clean`、脚本、Task1-6 结果与方法文档 |
| [`analysis_eu_geometry/`](analysis_eu_geometry) | 全库 E/U checkpoint 几何审计（94 模型） |
| [`archive/`](archive) | 2026-05-10 cleanup 归档，仅作追溯；日常分析不用看 |

## 目录功能

### `configs/`

- [`models.yaml`](configs/models.yaml)：模型简称、ModelScope/HF 仓库映射、下载缓存和抽取目录等全局配置。
- [`base_instruct_pairs.yaml`](configs/base_instruct_pairs.yaml)：Task1 Base-Instruct pair 配置，也是 full-vocab Base-Instruct 仿射分析的 pair 来源。
- [`model_series.yaml`](configs/model_series.yaml)：Task2 系列内模型列表，用于生成系列内两两 pair。
- [`cross_scale_groups.yaml`](configs/cross_scale_groups.yaml)：Task3 跨系列、跨规模桶分组。
- [`moe_cross_family.yaml`](configs/moe_cross_family.yaml)：Task4 MoE / 跨 family 分组。
- [`affine_pairs.yaml`](configs/affine_pairs.yaml)：Task5 仿射分析的 pair 来源清单，基本复用 Task1-4 生成的 pair。

### 数据目录

- [`downloaded_models/`](downloaded_models)：模型原始下载缓存；通常体积较大，按 `.gitignore` 不纳入版本管理。
- [`extracts/`](extracts)：从模型权重抽取出的 embedding / lm_head 矩阵和元信息，是当前分析脚本的主要输入。

### `tools/`

- [`tools/`](tools)：仓库级维护工具，包括模型下载抽取、完整性审计和冗余缓存清理；具体脚本见 [`tools/README.md`](tools/README.md)。

### `docs/`

- [`ANALYSIS_SCOPES_AND_SPECIAL_CASES.md`](docs/ANALYSIS_SCOPES_AND_SPECIAL_CASES.md)：全仓统一口径，包括全库模型数、tied/untied 数、GCorr Task1-4 pair 数、BI 35/30/26 口径，以及 Gemma-3-1B、Gemma-4、DeepSeek-V4 的排除 / 例外原因。

### `ijcai_clean/`

- [`src/ijcai_clean/`](ijcai_clean/src/ijcai_clean)：当前 Python 分析包，包含矩阵加载、token 对齐、GCorr 指标和任务实现。
- [`scripts/`](ijcai_clean/scripts)：主线任务入口，推荐使用 `run.py` 统一运行 Task1-6。
- [`results/`](ijcai_clean/results)：当前实验输出目录，保存 CSV、metadata 和轻量报告。
- [`docs/`](ijcai_clean/docs)：当前方法、指标、状态说明和 tied 审计；旧结果摘要与 cleanup 清单在 `docs/historical/`。
- [`analysis.md`](ijcai_clean/analysis.md)：当前研究结论备忘，包括 Base-Instruct full-vocab、Gemma 异常和 SVD 低秩分析。

### `analysis_eu_geometry/`

- [`analysis_eu_geometry/`](analysis_eu_geometry)：全库 E/U checkpoint 几何审计（行范数、μ-ratio、谱分析；BI 70 + 其他 24）；结论见 [`docs/FINDINGS.md`](analysis_eu_geometry/docs/FINDINGS.md)。不属于 `ijcai_clean` Task1-6 主线结果。

### `archive/`

- [`archive/ijcai_cleanup_2026-05-10/`](archive/ijcai_cleanup_2026-05-10)：第一轮 cleanup 归档，创建于 2026-05-10；包含旧 `legacy/`、旧实验草稿、2026-05-07 的 `s1_four_tasks_bundle` 结果快照和本机 cron/LaTeX 脚本。当前主线已有对应代码和结果，除非需要追溯历史状态，日常阅读和实验不用看这里。

## 快速命令

```bash
conda activate wzall

# 下载/抽取
python tools/get_model_useful.py

# 查看可运行任务
PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run.py list

# 任务一：Base–Instruct GCorr
PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run.py task1 --devices auto

# 任务二：系列组内 GCorr
PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run.py task2 --devices auto

# 任务三：跨规模桶 GCorr
PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run.py task3 --devices auto

# 任务四：MoE / 跨 family GCorr
PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run.py task4 --devices auto

# 任务五：Task1-4 pair 并集仿射 R²
PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run.py task5 --devices auto

# 任务六：Base-Instruct full-vocab 仿射 / A-I / SVD 诊断
PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run.py task6
```

若脚本无法自动定位仓库根，可设 `export REPO_ROOT=/path/to/get_useful`。

推荐运行环境为 conda 环境 `wzall`；当前已确认该环境可导入 `torch`、`transformers`、`safetensors`、`numpy`、`yaml`、`tqdm`，且 CUDA 可用。

更多说明见 [`ijcai_clean/README.md`](ijcai_clean/README.md)、[`ijcai_clean/docs/README.md`](ijcai_clean/docs/README.md)、[`ijcai_clean/docs/methods_and_metrics.md`](ijcai_clean/docs/methods_and_metrics.md)、[`ijcai_clean/analysis.md`](ijcai_clean/analysis.md) 与 [`analysis_eu_geometry/README.md`](analysis_eu_geometry/README.md)。
