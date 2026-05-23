# E/U checkpoint 几何审计

对 `configs/models.yaml` 全库 **94 模型** 的存盘 embedding（E）与 lm_head（U）做 **descriptive 几何审计**。三层分析：**行范数** → **mean-vector ratio** → **谱 / 各向同性**。起源是解释 Gemma-4 在 Task1 中的 raw Euclidean GCorr 异常；现以全库 E/U checkpoint 几何扫描为主轴。

**核心结论**见 [`docs/FINDINGS.md`](docs/FINDINGS.md)。特殊排错见 [`docs/SPECIAL_CASES.md`](docs/SPECIAL_CASES.md)。

| 项 | 说明 |
|----|------|
| **范围** | 94 模型（BI 70 + 其他 24） |
| **数据来源** | `extracts/*.safetensors` + `ijcai_clean.data.load_E_U_matrices` |
| **与主线** | 独立于 `ijcai_clean/results/task*/`；不写入 Task1–6 正式结果 |

---

## 目录结构

```
analysis_eu_geometry/
├── README.md                 # 本页：定位、命令、产物、指标
├── docs/
│   ├── FINDINGS.md           # 核心发现与对 GCorr 分析的建议
│   └── SPECIAL_CASES.md      # Gemma-4 gauge、DeepSeek-V4 HC head 排错
├── scripts/
│   ├── audit_row_norms.py    # 唯一 CLI 入口
│   └── row_norm_audit_common.py  # 加载、统计、汇总、μ-ratio、GPU 谱分析
└── results/row_norms/
    ├── base_instruct/        # BI 70 模型（行范数）
    ├── other_models/         # 非 BI 24 模型（行范数）
    ├── all_models/           # 94 模型合并总表
    ├── mu_ratio/             # ‖μ‖ / mean(row norm)
    └── spectral/             # GPU economy SVD / 各向同性指标
```

| 路径 | 说明 |
|------|------|
| `docs/FINDINGS.md` | 核心发现（Gemma-4、BI Δ、tied/untied、谱分析摘要） |
| `docs/SPECIAL_CASES.md` | Gemma-4 / DeepSeek-V4 架构例外与解读边界 |
| `scripts/audit_row_norms.py` | 唯一入口脚本 |
| `scripts/row_norm_audit_common.py` | 共用库 |
| `results/row_norms/base_instruct/` | Base–Instruct 70 模型 |
| `results/row_norms/other_models/` | models.yaml 中其余 24 模型 |
| `results/row_norms/all_models/` | 94 模型合并总表 |
| `results/row_norms/mu_ratio/` | mean-vector ratio 汇总 |
| `results/row_norms/spectral/` | GPU economy SVD / 各向同性谱指标 |

---

## 数据流

```text
configs/models.yaml              ─┐
configs/base_instruct_pairs.yaml ─┼─► audit_row_norms.py
extracts/<model>/*.safetensors   ─┘         │
                                            ├─► base_instruct/  (行范数, 70 models)
                                            ├─► other_models/   (行范数, 24 models)
                                            ├─► all_models/     (merge + 汇总 MD)
                                            ├─► mu_ratio/       (--mu-ratio-only)
                                            └─► spectral/       (--svd-only, GPU)
```

环境：`conda activate wzall`。脚本通过 `bootstrap_repo()` 自动把 `ijcai_clean/src` 加入 `sys.path`。

---

## 脚本

```bash
conda activate wzall

# 全库：BI 70 + other 24 + 合并总表（默认，行范数）
python analysis_eu_geometry/scripts/audit_row_norms.py

# 只跑 BI 70
python analysis_eu_geometry/scripts/audit_row_norms.py --scope base_instruct

# 只跑 other 24
python analysis_eu_geometry/scripts/audit_row_norms.py --scope other_models

# 只合并已有 CSV
python analysis_eu_geometry/scripts/audit_row_norms.py --merge-only

# mean-vector ratio（CPU 并行）
python analysis_eu_geometry/scripts/audit_row_norms.py --mu-ratio-only --workers 8

# GPU 谱分析
CUDA_VISIBLE_DEVICES=0 python analysis_eu_geometry/scripts/audit_row_norms.py --svd-only --device cuda:0
```

输入：`extracts/`、`configs/base_instruct_pairs.yaml`、`configs/models.yaml`。

### 三层分析

| 层 | 模式 | 算什么 |
|----|------|--------|
| 1 | 默认 / `--scope` | 每行 L2 范数的 mean、var、min、max |
| 2 | `--mu-ratio-only` | `‖μ‖ / mean(row norm)`，衡量行向量公共 mean 方向强度 |
| 3 | `--svd-only` | PR/d、rank1、centered rank1、effective_rank、σ_ratio 等各向同性/异性指标 |

**行范数**（层 1）按 BI / other 分批写盘再 merge；**μ-ratio / 谱分析**（层 2–3）直接扫全库 94 模型。tied 模型在层 2–3 只分析 E（U=E，不重复）。

### 谱指标（层 3）

对 n×d 矩阵在 GPU 上做 **d 维 economy SVD**（via `G = MᵀM/n`，远快于 full n-SVD）。未中心化（原始 M）与中心化（M − μ）各算一套：

| 指标 | 含义 |
|------|------|
| `rank1_energy_frac` | σ₁² / Σσᵢ²；越大越各向异性 |
| `isotropy_pr_over_d` = PR/d | →1 各向同性；→1/d 极度异性 |
| `effective_rank` | 谱熵有效秩 |
| `sigma_ratio` | σ_max / σ_min |
| `cos_v1_mu` | 最大奇异向量与 μ 方向对齐度 |

完整列名见 `spectral/all_models_spectral.csv`。

---

## 主要结果文件

### 行范数（层 1）

| 文件 | 说明 |
|------|------|
| `all_models/all_models_row_norms.csv` | **94 模型 × E/U 长表**（含 `subset`） |
| `all_models/ALL_MODELS_ROW_NORMS_SUMMARY.md` | 全库一页汇总：表 1 BI 35 对 + 表 2 其他 24 + 表 3 Untied Δ% |
| `base_instruct/base_instruct_row_norms_by_pair.csv` | 35 对 BI + `delta_E_pct` / `delta_U_pct` |
| `base_instruct/BASE_INSTRUCT_ROW_NORMS_SUMMARY.md` | BI 按系列分组汇总 |
| `other_models/other_models_row_norms.csv` | 非 BI 24 模型长表 |
| `other_models/OTHER_MODELS_ROW_NORMS_SUMMARY.md` | 按 model_group 分组 |

各子目录另有 `*_by_model.csv` 宽表（展示用，可由长表生成）。

### μ-ratio / 谱分析（层 2–3）

| 文件 | 说明 |
|------|------|
| `mu_ratio/all_models_mu_ratio.csv` | 全库 `‖μ‖ / mean(row norm)`（144 行：tied 仅 E） |
| `mu_ratio/ALL_MODELS_MU_RATIO_SUMMARY.md` | 全表 Markdown |
| `spectral/all_models_spectral.csv` | 全库 E/U 谱指标（144 行，32 列） |
| `spectral/ALL_MODELS_SPECTRAL_SUMMARY.md` | PR/d、rank1、eff_rank、σ_ratio 摘要 |

> 行范数长表 188 行（94×E/U）；μ-ratio / spectral 对 tied 模型只保留 E，故 144 行。

---

## 结论速查

详见 [`docs/FINDINGS.md`](docs/FINDINGS.md)；此处仅列索引。

| 主题 | 要点 |
|------|------|
| **Gemma-4** | 全库唯一 near-unit-sphere Base E（mean≈1，var~10⁻⁹）；Instruct 径向漂移 → raw Euclidean GCorr 误导 |
| **BI Δ mean** | 常态 ±1% 以内；13 untied 中 \|ΔU\| 中位 0.87% > \|ΔE\| 0.31%；Gemma-4 为 +12%～+52% outlier |
| **untied 谱** | 48/50 模型 `rank1(U) > rank1(E)`；例外 DeepSeek-V4（hc_head 架构） |
| **DSV4** | 裸 U 不等价传统 lm_head；对本目录 embedding 主线解释意义较低 |

架构细节见 [`docs/SPECIAL_CASES.md`](docs/SPECIAL_CASES.md)。

---

## 已知限制

1. 只读存盘权重，不含 runtime norm、量化、adapter
2. 谱分析需 GPU（`--svd-only`）
3. tied 模型 U 与 E 相同，μ-ratio / 谱只报 E
4. DSV4 有效输出头（`hc_head` 动态混合）暂无法从 checkpoint 静态还原

---

## 与主线实验的关系

本目录只做 E/U **存盘几何**的 descriptive 审计，用于解释 Gemma-4 等指标异常。Task1–6 正式结果仍在 `ijcai_clean/results/`。
