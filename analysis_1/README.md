# Analysis 1：静态 E/U checkpoint 谱 + GCorr 证据包

本目录为 **可独立抽走** 的自包含包：分析叙事、模型配置、原始结果 CSV、特殊案例索引。  
数据字典见 [`DATA.md`](DATA.md)；文件清单见 [`MANIFEST.json`](MANIFEST.json)。
全仓统一口径与特殊案例见 [`../docs/分析口径与特殊案例.md`](../docs/分析口径与特殊案例.md)。

## 分析笔记

| 文件 | 内容 |
|------|------|
| [`notes/01_untied_u_vs_e_static_spectrum.md`](notes/01_untied_u_vs_e_static_spectrum.md) | **untied 48 对**：U mean 更强、raw rank1 48/48 更高；仅 rank1 更集中 |
| [`notes/02_tied_static_spectrum_like_u.md`](notes/02_tied_static_spectrum_like_u.md) | **tied 34 个**（排除 Gemma-4、Gemma-3-1B）：共享矩阵谱更像 untied U |
| [`notes/03_gcorr_bi_confidence.md`](notes/03_gcorr_bi_confidence.md) | **Task1 BI**：untied E_euc 几乎全 1；tied E_cos 同量级极高、E_euc 有 range |
| [`notes/04_insight1_main_evidence.md`](notes/04_insight1_main_evidence.md) | **Insight 1 主证据**：Task2 37/37 + 静态谱（均不依赖 Gemma-4 BI E_euc） |

## 配置（模型注册与子集）

| 文件 | 内容 |
|------|------|
| [`configs/analysis_subsets.yaml`](configs/analysis_subsets.yaml) | **各 note 用的 n= 口径**（48/34/13/17/37）与排除规则 |
| [`configs/bi_pairs.yaml`](configs/bi_pairs.yaml) | 更新后的 BI 分析分层（main/extended/excluded、tied 标记） |
| [`configs/base_instruct_pairs.yaml`](configs/base_instruct_pairs.yaml) | Task1 全库 35 对 Base→Instruct |
| [`configs/model_series.yaml`](configs/model_series.yaml) | Task2 系列内 cross-model 配对 |

## 数据（原始 CSV）

| 路径 | 用于 |
|------|------|
| [`data/static/layer3_spectral.csv`](data/static/layer3_spectral.csv) | notes 01–02 静态谱 |
| [`data/static/all_models_eu_features.csv`](data/static/all_models_eu_features.csv) | 谱特征宽表（补充） |
| [`data/task1_base_instruct/summary.csv`](data/task1_base_instruct/summary.csv) | note 03 Task1 GCorr |
| [`data/task1_base_instruct/bootstrap_results.csv`](data/task1_base_instruct/bootstrap_results.csv) | note 03 组级 CI |
| [`data/task2_model_series/summary.csv`](data/task2_model_series/summary.csv) | note 04 Task2 全 110 对 |
| [`data/task2_model_series/tied_x_untied_pairs.csv`](data/task2_model_series/tied_x_untied_pairs.csv) | note 04 **37 对**子集 |

## 文档

| 文件 | 内容 |
|------|------|
| [`docs/SPECIAL_CASES.md`](docs/SPECIAL_CASES.md) | 旧链接兼容入口，转向全仓统一口径文档 |
| [`../docs/分析口径与特殊案例.md`](../docs/分析口径与特殊案例.md) | 全仓口径、排除规则与特殊案例权威说明 |

## 排除约定

| 分组 | 排除 | 原因 |
|------|------|------|
| untied 静态谱 | DeepSeek-V4-Flash / Pro | [`dsv4_hc_head`](../docs/分析口径与特殊案例.md#63-dsv4_hc_head) |
| tied 静态谱 / Task1 BI | Gemma-4 全系、Gemma-3-1B | [`gemma4_checkpoint_gauge`](../docs/分析口径与特殊案例.md#62-gemma4_checkpoint_gauge) / [`gemma3_1b_rewrite`](../docs/分析口径与特殊案例.md#61-gemma3_1b_rewrite) |

完整模型名单见 [`configs/analysis_subsets.yaml`](configs/analysis_subsets.yaml)。

## 上游（重算时用，非抽走必需）

| 资源 | 仓库原路径 |
|------|------------|
| GPU SVD 脚本 | [`analysis_eu_geometry/scripts/eu_geometry/spectral.py`](../analysis_eu_geometry/scripts/eu_geometry/spectral.py) |
| GCorr 流水线 | [`cross_model_geometry/`](../cross_model_geometry/) |
