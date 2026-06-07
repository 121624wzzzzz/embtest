# E/U checkpoint 几何审计

本目录是一个独立的 **checkpoint 静态几何审计** 子项目：读取仓库根目录的
`configs/` 与 `extracts/`，对 `configs/models.yaml` 中全库 **94 个模型** 的
存盘 embedding（E）与 lm_head / unembedding（U）做 descriptive geometry audit。

它回答的问题是：checkpoint 里的 E/U 行范数、mean vector、谱结构是否存在异常，
以及这些静态几何特征如何解释主线实验中的特殊现象，例如 Gemma-4 的 row-norm
gauge 和 DeepSeek-V4 的非传统 output head。

本目录 **不运行 GCorr**，也 **不做 affine fit / LoRA budget 实验**；这些属于
[`../ijcai_clean/`](../ijcai_clean/) 和 [`../__tep/affine/`](../__tep/affine/)。
这里产出的静态几何表可被它们引用或用于解释异常。

**核心结论**见 [`docs/FINDINGS.md`](docs/FINDINGS.md)。全仓口径与特殊案例见
[`../docs/分析口径与特殊案例.md`](../docs/分析口径与特殊案例.md)。
静态 E/U 谱专题证据包见 [`../analysis_1/`](../analysis_1/)。

---

## 目录结构

```
analysis_eu_geometry/
├── README.md
├── docs/                       # 结论与特殊案例说明
├── scripts/
│   ├── audit_eu_geometry.py    # CLI
│   └── eu_geometry/            # common + row_norms + mu_ratio + spectral + features
└── results/
    ├── all_models_eu_features.csv      ← 主表：188 行 × 全特征
    ├── ALL_MODELS_EU_FEATURES_SUMMARY.md
    ├── bi_row_norms.csv              ← BI 70 模型长表（每 model×E/U 一行）
    ├── bi_pair_delta.csv             ← BI 35 对 Δ%
    ├── BI_ROW_NORMS_SUMMARY.md
    ├── other_models_row_norms.csv    ← 非 BI 24 中间表
    └── layers/                       ← 三层分析长表
        ├── layer1_row_norms.csv
        ├── layer2_mu_ratio.csv
        └── layer3_spectral.csv
```

`__pycache__/` 和 `*.pyc` 是 Python 运行缓存，不属于项目内容。

## 文件职责

### 文档

| 文件 | 功能 |
|------|------|
| [`README.md`](README.md) | 本目录入口：说明定位、数据流、脚本、结果文件与边界 |
| [`docs/FINDINGS.md`](docs/FINDINGS.md) | 核心发现：Gemma-4 gauge、untied E/U 静态谱差异、BI 行范数漂移、DeepSeek-V4 例外 |
| [`../docs/分析口径与特殊案例.md`](../docs/分析口径与特殊案例.md) | 全仓统一口径与特殊案例：Gemma-3-1B、Gemma-4、DeepSeek-V4 |
| [`docs/SPECIAL_CASES.md`](docs/SPECIAL_CASES.md) | 旧链接兼容入口，转向全仓统一口径文档 |

### 脚本

| 文件 | 功能 |
|------|------|
| [`scripts/audit_eu_geometry.py`](scripts/audit_eu_geometry.py) | CLI 总入口；负责调用 row-norm、μ-ratio、SVD 谱分析和 merge |
| [`scripts/eu_geometry/__init__.py`](scripts/eu_geometry/__init__.py) | 暴露本子项目内部 runner，供 CLI 统一 import |
| [`scripts/eu_geometry/common.py`](scripts/eu_geometry/common.py) | 公共工具：仓库根定位、结果路径、CSV I/O、模型目录、BI / other 模型划分、基础统计 |
| [`scripts/eu_geometry/row_norms.py`](scripts/eu_geometry/row_norms.py) | Layer 1：计算 E/U 行范数 mean、var、min、max；生成 BI pair delta 与行范数汇总 |
| [`scripts/eu_geometry/mu_ratio.py`](scripts/eu_geometry/mu_ratio.py) | Layer 2：计算 `||μ||` 和 `||μ|| / mean(row norm)`；支持 CPU 多进程 |
| [`scripts/eu_geometry/spectral.py`](scripts/eu_geometry/spectral.py) | Layer 3：GPU Gram economy SVD；计算 rank1、PR/d、effective rank、centered spectrum 等 |
| [`scripts/eu_geometry/features.py`](scripts/eu_geometry/features.py) | 合并层：按 `(model, matrix)` 合并 Layer 1/2/3 为总表，并生成 markdown 浏览版 |

### 结果

| 文件 | 功能 |
|------|------|
| [`results/all_models_eu_features.csv`](results/all_models_eu_features.csv) | **主表**：每行一个 `(model, matrix)`，合并 row-norm、μ-ratio、spectral 全部特征 |
| [`results/ALL_MODELS_EU_FEATURES_SUMMARY.md`](results/ALL_MODELS_EU_FEATURES_SUMMARY.md) | 主表的人类可读浏览版，便于快速扫模型关键指标 |
| [`results/bi_row_norms.csv`](results/bi_row_norms.csv) | Base-Instruct 相关 70 个模型的 E/U 行范数长表 |
| [`results/bi_pair_delta.csv`](results/bi_pair_delta.csv) | 35 对 Base-Instruct pair 的行范数变化百分比：`delta_E_pct` / `delta_U_pct` |
| [`results/BI_ROW_NORMS_SUMMARY.md`](results/BI_ROW_NORMS_SUMMARY.md) | BI 行范数 markdown 汇总，按模型族分组 |
| [`results/other_models_row_norms.csv`](results/other_models_row_norms.csv) | 非 BI 模型的行范数审计结果 |
| [`results/rank1_vs_affine_untied_gpu.csv`](results/rank1_vs_affine_untied_gpu.csv) | 后验交叉表：13 对 untied BI 的静态谱指标 × affine 分解指标 |
| [`results/rank2_vs_rank1c_untied_gpu.csv`](results/rank2_vs_rank1c_untied_gpu.csv) | 后验验证表：检查 centered rank1(c) 与 raw rank2 的关系 |
| [`results/layers/layer1_row_norms.csv`](results/layers/layer1_row_norms.csv) | Layer 1 全库长表：行范数 mean、var、min、max |
| [`results/layers/layer2_mu_ratio.csv`](results/layers/layer2_mu_ratio.csv) | Layer 2 全库长表：mean-vector norm 与 μ-ratio |
| [`results/layers/layer3_spectral.csv`](results/layers/layer3_spectral.csv) | Layer 3 全库长表：GPU SVD 谱和各向同性指标 |

说明：tied 模型 E=U。Layer 2 / Layer 3 对 tied 模型只计算 E；合并主表时，
`features.py` 会把 tied 模型 E 的 μ/谱指标沿用到 U 行，因此主表仍保持
`94 models × 2 matrices = 188` 行。

---

## 数据流

```text
extracts/ + configs/  ──► audit_eu_geometry.py
                              │
                              ├─► bi_row_norms.csv  ─┐
                              ├─► other_models_row_norms.csv   ─┼─► layers/layer1_row_norms.csv
                              ├─► layers/layer2_mu_ratio.csv   ─┤
                              └─► layers/layer3_spectral.csv   ─┴─► all_models_eu_features.csv
```

---

## 脚本

```bash
conda activate wzall
python analysis_eu_geometry/scripts/audit_eu_geometry.py              # 层 1 全量 + merge
python analysis_eu_geometry/scripts/audit_eu_geometry.py --merge-only # 只合并已有 CSV
python analysis_eu_geometry/scripts/audit_eu_geometry.py --mu-ratio-only --workers 8
CUDA_VISIBLE_DEVICES=0 python analysis_eu_geometry/scripts/audit_eu_geometry.py --svd-only
```

CLI 参数含义：

| 参数 | 功能 |
|------|------|
| `--scope all` | 默认；跑 BI 模型、other 模型，并合并为全库 layer1 / feature 表 |
| `--scope base_instruct` | 只跑 Base-Instruct 相关模型的 row-norm 审计 |
| `--scope other_models` | 只跑非 BI 模型的 row-norm 审计 |
| `--merge-only` | 不重算矩阵统计，只把已有 CSV 合并成 layer1 / 主表 |
| `--mu-ratio-only` | 只跑 Layer 2 mean-vector ratio |
| `--workers N` | `--mu-ratio-only` 的 CPU 并行进程数 |
| `--svd-only` | 只跑 Layer 3 GPU 谱分析 |
| `--device cuda:0` | `--svd-only` 使用的设备 |

### 三层分析

| 层 | 输出文件 | 内容 |
|----|----------|------|
| 1 | `layers/layer1_row_norms.csv` | mean, var, min, max |
| 2 | `layers/layer2_mu_ratio.csv` | mean_vec_norm, mu_over_row_norm |
| 3 | `layers/layer3_spectral.csv` | rank1, PR/d, σ_ratio, … |
| 合并 | **`all_models_eu_features.csv`** | 以上三层按 `(model, matrix)` 合一 |

---

## 主要结果文件

| 文件 | 何时用 |
|------|--------|
| **`all_models_eu_features.csv`** | 全库分析、画图、筛模型（首选） |
| `bi_pair_delta.csv` | BI 配对 ΔE/ΔU |
| `BI_ROW_NORMS_SUMMARY.md` | 按系列浏览 BI 行范数 |
| `layers/layer1/2/3_*.csv` | 只重跑某一层后的中间表 |
| `rank1_vs_affine_untied_gpu.csv` | 静态谱与 affine 分解的后验交叉检查 |
| `rank2_vs_rank1c_untied_gpu.csv` | centered rank1(c) 与 raw rank2 的后验验证 |

---

## 与主线实验的关系

本目录独立于 `ijcai_clean/results/task*/`。它主要用于解释 checkpoint 几何异常，
并为 `analysis_1/`、`bi_analysis/`、`__tep/` 中的论文叙事提供静态 E/U 证据。

主要边界如下：

| 目录 | 角色 |
|------|------|
| [`../ijcai_clean/`](../ijcai_clean/) | Task1-6 主实验代码与结果 |
| [`../analysis_eu_geometry/`](.) | 全库 E/U checkpoint 静态几何审计 |
| [`../analysis_1/`](../analysis_1/) | 可独立抽走的静态谱 + GCorr 证据包 |
| [`../bi_analysis/`](../bi_analysis/) | Base-Instruct 口径、排除规则和叙事表格 |
| [`../__tep/`](../__tep/) | GCorr / affine 两条论文线的洞察、派生表和脚本 |
