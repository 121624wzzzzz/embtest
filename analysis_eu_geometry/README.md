# E/U checkpoint 几何审计

对 `configs/models.yaml` 全库 **94 模型** 的存盘 embedding（E）与 lm_head（U）做 **descriptive 几何审计**。三层分析：**行范数** → **mean-vector ratio** → **谱 / 各向同性**。

**核心结论**见 [`docs/FINDINGS.md`](docs/FINDINGS.md)。特殊排错见 [`docs/SPECIAL_CASES.md`](docs/SPECIAL_CASES.md)。  
静态 E/U 谱专题分析见 [`../analysis_1/`](../analysis_1/)。

---

## 目录结构

```
analysis_eu_geometry/
├── README.md
├── docs/                       # FINDINGS、SPECIAL_CASES
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

---

## 与主线实验的关系

独立于 `ijcai_clean/results/task*/`；用于解释 Gemma-4 等 checkpoint 几何异常。
