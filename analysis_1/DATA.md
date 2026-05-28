# Analysis 1 数据与配置说明

本目录为 **可独立抽走** 的自包含包：`notes/` 叙事 + `configs/` 模型注册 + `data/` 原始结果 CSV。  
无需再回读仓库根目录的 `ijcai_clean/` 或 `analysis_eu_geometry/results/` 即可复核文中数字。

---

## 目录结构

```text
analysis_1/
├── README.md                 # 总览与 note 索引
├── DATA.md                   # 本文件：数据字典
├── MANIFEST.json             # 打包文件清单与 upstream git commit
├── configs/
│   ├── analysis_subsets.yaml # ★ 各 note 用的模型子集（48/34/13/17/37 等）
│   ├── base_instruct_pairs.yaml  # Task1 全库 35 对 Base→Instruct
│   ├── bi_pairs.yaml             # ★ 更新后的分析分层（main/extended/excluded、tied 标记）
│   └── model_series.yaml         # Task2 系列内 cross-model 配对来源
├── data/
│   ├── static/               # 静态 checkpoint E/U 谱（非 BI ΔW）
│   ├── task1_base_instruct/  # Task1 GCorr（Base–Instruct BI）
│   └── task2_model_series/   # Task2 GCorr（同系列 Instruct 跨模型）
├── docs/
│   └── SPECIAL_CASES.md      # Gemma-4 gauge、DSV4 架构例外
└── notes/
    ├── 01_untied_u_vs_e_static_spectrum.md
    ├── 02_tied_static_spectrum_like_u.md
    ├── 03_gcorr_bi_confidence.md
    └── 04_insight1_main_evidence.md
```

---

## 配置文件

| 文件 | 作用 | 与 note 关系 |
|------|------|--------------|
| [`configs/analysis_subsets.yaml`](configs/analysis_subsets.yaml) | **汇总本包所有子集**：48 untied、34 tied、Task1 13+17、Task2 37 对等；含排除规则与模型名单 | 所有 note 的「n=」口径 |
| [`configs/bi_pairs.yaml`](configs/bi_pairs.yaml) | BI 分析注册表（2026-05-24）：`analysis_tier`（main/extended/excluded）、`tied`、`exclude_reason` | note 03 的 untied 13 / tied 17 来自 `analysis_untied` + 排除 Gemma-4/3-1B |
| [`configs/base_instruct_pairs.yaml`](configs/base_instruct_pairs.yaml) | Task1 上游 35 对（含 Gemma-4、Gemma-3-1B） | `data/task1_*` 的配对来源 |
| [`configs/model_series.yaml`](configs/model_series.yaml) | Task2 按系列列模型 | `data/task2_*` 的 110 对计划来源 |

**子集速查**（详见 `analysis_subsets.yaml`）：

| 子集 key | n | 用于 |
|----------|---|------|
| `static_untied_48_comparable` | 48 | note 01（排除 DSV4-Flash/Pro） |
| `static_tied_34_spectrum` | 34 | note 02（排除 Gemma-4 全系、Gemma-3-1B） |
| `task1_untied_13` | 13 | note 03 untied E_euc≈1 |
| `task1_tied_17_excl_gemma4_3_1b` | 17 | note 03 tied E_cos / E_euc |
| `task2_tied_x_untied_37` | 37 | note 04 Insight 1 主证据 A |

---

## 静态谱数据（notes 01–02）

### `data/static/layer3_spectral.csv`

| 项 | 说明 |
|----|------|
| **生成** | `analysis_eu_geometry/scripts/eu_geometry/spectral.py`（economy SVD on checkpoint E/U） |
| **行粒度** | 每模型 × 矩阵（untied：`E` 与 `U` 各一行；tied：仅 `E`，因 E=U 共享） |
| **规模** | 94 模型 → 144 行（50 untied + 44 tied） |
| **关键列** | `model`, `tied`, `matrix`, `mu_over_row_norm`, `rank1_energy_frac`, `cos_v1_mu`, `rank5_energy_frac`, `rank1_centered_energy_frac`, … |
| **note 01** | 筛 `tied=False`，去 DSV4 → 48 模型 × 2 矩阵 |
| **note 02** | 筛 `tied=True`，去 Gemma-4/3-1B → 34 模型 |

### `data/static/all_models_eu_features.csv`

| 项 | 说明 |
|----|------|
| **内容** | 谱 + 行范数 + μ 比等合并宽表（每模型一行，E/U 分列） |
| **用途** | 补充查询；note 01–02 主表直接读 `layer3_spectral.csv` |

---

## Task1 BI GCorr（note 03）

### `data/task1_base_instruct/summary.csv`

| 项 | 说明 |
|----|------|
| **实验** | 同模型 Base vs Instruct，token 对齐后 E–E、U–U 的 GCorr |
| **行粒度** | 35 对（与 `base_instruct_pairs.yaml` 一致） |
| **关键列** | `model_a`, `actual_tied_a`, `gcorr_E_cos_mean`, `gcorr_E_euc_mean`, `gcorr_E_*_ci95_low/high`, 及 U 侧同名列 |
| **note 03** | untied：`actual_tied_a=False`（13 对）；tied：True 且非 Gemma-4/3-1B（17 对） |

### `data/task1_base_instruct/bootstrap_results.csv`

| 项 | 说明 |
|----|------|
| **行粒度** | 35 对 × 100 bootstrap × 指标 |
| **关键列** | `bootstrap`, `gcorr_E_cos`, `gcorr_E_euc`, `gcorr_E_euc2`, … |
| **note 03** | 按 bootstrap 索引对子集求 mean，得组级 CI 与 untied−tied 差值 CI |

### `data/task1_base_instruct/metadata.json`

运行参数快照：`n_tokens=20000`, `n_pairs=5000000`, `n_bootstrap=100`, `pairs_file`, `git_commit`。

---

## Task2 跨模型 GCorr（note 04）

### `data/task2_model_series/summary.csv`

| 项 | 说明 |
|----|------|
| **实验** | 同系列 Instruct 模型两两 GCorr（非 BI） |
| **行粒度** | 110 对（全 Task2） |
| **关键列** | 同 Task1；另含 `actual_tied_a/b` 标识各侧是否 tied |

### `data/task2_model_series/tied_x_untied_pairs.csv`

| 项 | 说明 |
|----|------|
| **内容** | 从 `summary.csv` 筛出 **一侧 tied、一侧 untied** 的 **37 对** |
| **note 04** | Insight 1 主证据 A：37/37 U 侧 > E 侧 |

### `data/task2_model_series/pair_plan.csv`

Task2 110 对的系列、源 Base/Instruct 选型记录（审计用）。

### `data/task2_model_series/metadata.json`

`n_pairs_config=110`, `series_file`, `git_commit` 等。

---

## 特殊案例文档

[`docs/SPECIAL_CASES.md`](docs/SPECIAL_CASES.md) — Gemma-4 near-unit-sphere gauge、DeepSeek-V4 hc_head 架构；解释为何静态谱 / Task1 排除某些模型，以及 **Insight 1 不依赖 Gemma-4 Task1 E_euc**。

---

## 指标缩写（GCorr）

| 列前缀 | 含义 |
|--------|------|
| `gcorr_E_cos` | input embedding 方向几何相关 |
| `gcorr_E_euc` | input embedding 欧氏距离几何相关 |
| `gcorr_E_euc2` | 平方欧氏距离 GCorr |
| `gcorr_U_*` | lm_head（untied）或共享矩阵 U 侧（tied 时 U=E） |

Task2 比较 tied 共享矩阵 vs untied E/U 时：tied 侧只有一行矩阵，与 untied 的 U 列、E 列分别比较。

---

## 从本包复现数字（最小依赖）

```bash
# 例：note 04 Task2 37/37
python3 - <<'PY'
import csv, statistics as st
rows = list(csv.DictReader(open("data/task2_model_series/tied_x_untied_pairs.csv")))
uc = [float(r["gcorr_U_cos_mean"]) for r in rows]
ec = [float(r["gcorr_E_cos_mean"]) for r in rows]
print(len(rows), sum(u>e for u,e in zip(uc,ec)), st.mean(uc), st.mean(ec))
PY
```

仅需 Python 3 + 本目录内 CSV；无需 GPU 或 checkpoint。

---

## 上游来源（完整重跑时用）

| 本包副本 | 仓库原路径 |
|----------|------------|
| `data/static/*` | `analysis_eu_geometry/results/` |
| `data/task1_*` | `ijcai_clean/results/task1_base_instruct/` |
| `data/task2_*` | `ijcai_clean/results/task2_model_series/` |
| `configs/base_instruct_pairs.yaml` | `configs/base_instruct_pairs.yaml` |
| `configs/bi_pairs.yaml` | `bi_analysis/bi_pairs.yaml` |
| `configs/model_series.yaml` | `configs/model_series.yaml` |

重算静态谱需 checkpoint 与 `analysis_eu_geometry/scripts/`；重算 GCorr 需 `ijcai_clean/` 与 `extracts/`。
