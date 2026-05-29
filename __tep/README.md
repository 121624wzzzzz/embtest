# __tep/

`__tep/` 是两条论文线的整理区。源数据只读 `../ijcai_clean/results/` 与 `../extracts/`；本目录只放实验洞察、派生表和复现实验脚本。

## 入口

| 模块 | 目录 | 先读 | 说明 |
|------|------|------|------|
| A. GCorr / AGD 诊断 | [`gcorr/`](gcorr/) | [`gcorr/INSIGHTS.md`](gcorr/INSIGHTS.md) | 保留为论文 A 的数据审计与改稿清单 |
| B. Affine / low-rank update | [`affine/`](affine/) | [`affine/INSIGHTS.md`](affine/INSIGHTS.md) | 当前主线，E/U + tied/untied 故事已收束 |
| 全局统计 | [`data/`](data/) | [`data/key_metrics.csv`](data/key_metrics.csv) | 少量全局统计 JSON/CSV |
| 派生脚本 | [`scripts/`](scripts/) | [`scripts/verify_metrics.py`](scripts/verify_metrics.py) | 派生表与复核脚本 |

## 当前主结论

主分析组为 31 个 Base-Instruct pair 排除 `Gemma-3-1B` 与 `Gemma-4-*` 后的 26 对。

| 结论 | 数字 |
|------|------|
| GCorr 与 affine R2 在 BI 主组双高 | GCorr cos mean ≈ **0.995**；E affine R2 mean ≈ **0.991** |
| 普通 R2 不是核心证据 | identity R2 本来接近 1；要看 update-scale |
| E 侧 affine 优势存在但有边界 | `P/D` median **0.120**；低 W-rank 窗口内较强 |
| U/lm_head 侧更支持 affine | `P/D` median **0.315**；hybrid 在 `rW=2/4/8` 为 **26/26** 胜 |
| tied/untied 是解释变量 | tied: E/U 等价；untied: E 侧弱、U 侧强 |

最关键的总表：

- [`affine/tables/final/model_level_e_u_affine_lora_summary.csv`](affine/tables/final/model_level_e_u_affine_lora_summary.csv) ：逐模型 E/U 全指标。
- [`affine/tables/final/model_level_e_u_by_tied_summary.csv`](affine/tables/final/model_level_e_u_by_tied_summary.csv) ：tied/untied 汇总。
- [`affine/tables/final/model_level_e_u_by_family_size_summary.csv`](affine/tables/final/model_level_e_u_by_family_size_summary.csv) ：按模型族/尺寸汇总。

## 符号约定

- 主分析组：31 对 BI 排除 `Gemma-3-1B` 与 `Gemma-4-*`，即 n=26。
- `E` = input embedding；`U` / `lm_head` = unembedding。
- `D=Y_c-X_c`，`P=X_c(A-I)`，`R=D-P`。
- `P/D` 或 `full_affine_gain` 指 `||P||_F^2 / ||D||_F^2`，是当前 affine 叙事的核心 update-scale 指标。

## 目录结构

```text
__tep/
├── README.md
├── affine/
│   ├── INSIGHTS.md
│   ├── README.md
│   ├── analysis/
│   └── tables/  final/ · e/ · u/ · archive/
├── gcorr/
│   ├── INSIGHTS.md
│   ├── analysis/
│   ├── tables/
│   └── main_zh_neurips_full.tex
├── data/
└── scripts/     active scripts + archive/
```

## 复现

关键派生脚本在 [`scripts/`](scripts/)；核心数字可用：

```bash
python3 scripts/verify_metrics.py
```

Affine E/U 与 LoRA budget 的新结果主要由以下脚本生成：

- [`scripts/compute_affine_pred_delta_svd.py`](scripts/compute_affine_pred_delta_svd.py)
- [`scripts/evaluate_w_rank_budget.py`](scripts/evaluate_w_rank_budget.py)
- [`scripts/evaluate_pred_delta_common_spectrum.py`](scripts/evaluate_pred_delta_common_spectrum.py)
- [`scripts/evaluate_hybrid_affine_w_budget.py`](scripts/evaluate_hybrid_affine_w_budget.py)
