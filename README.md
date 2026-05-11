# get_useful：嵌入 / 解嵌矩阵下载与分析

## 布局

| 位置 | 说明 |
|------|------|
| [`configs/`](configs) | 全局模型配置 `models.yaml` 与任务配置 YAML |
| [`downloaded_models/`](downloaded_models) | 下载缓存 |
| [`extracts/`](extracts) | 每模型 `*.safetensors` + `*.info.json` |
| [`tools/`](tools) | 下载、审计、清理 |
| [`ijcai_clean/`](ijcai_clean) | 当前分析子项目：包 `ijcai_clean`、脚本、结果、**`legacy/` 旧实验** |

## 快速命令

```bash
conda activate wzall

# 下载/抽取
python tools/get_model_useful.py

# 任务一：Base–Instruct GCorr
PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run_task1_base_instruct.py --devices auto

# 任务二：系列组内 GCorr
PYTHONPATH=ijcai_clean/src python ijcai_clean/scripts/run_task2_model_series.py --devices auto
```

若脚本无法自动定位仓库根，可设 `export REPO_ROOT=/path/to/get_useful`。

推荐运行环境为 conda 环境 `wzall`；当前已确认该环境可导入 `torch`、`transformers`、`safetensors`、`numpy`、`yaml`、`tqdm`，且 CUDA 可用。

更多说明见 [`ijcai_clean/README.md`](ijcai_clean/README.md)、[`ijcai_clean/docs/current_state.md`](ijcai_clean/docs/current_state.md) 与 [`ijcai_clean/docs/methods_and_metrics.md`](ijcai_clean/docs/methods_and_metrics.md)。
