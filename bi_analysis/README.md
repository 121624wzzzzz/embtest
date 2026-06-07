# BI Analysis

Base→Instruct（BI）相关的问题梳理、结论与备忘。与 `ijcai_clean/`（Task1–6 实验）、`__tep/affine/`（仿射分解主线）并列，侧重**叙事与判断**，不重复存放原始数据与计算脚本。

## 目录

| 文件 | 内容 |
|------|------|
| [`bi_pairs.yaml`](bi_pairs.yaml) | **纳入分析的 BI 对**：按模型族排列，标注 tied / 主组 tier |
| [`tables/README.md`](tables/README.md) | **数据文件总目录**：每表行数、含义、生成方式 |
| [`tables/DELTA_W_R2_ONE_MINUS.md`](tables/DELTA_W_R2_ONE_MINUS.md) | 30 对的 ΔW 相关 R²（**1−R²** 形式） |
| [`tables/EXTENDED_DPR_COMMON_SPECTRUM.md`](tables/EXTENDED_DPR_COMMON_SPECTRUM.md) | extended 4 对 D/P/R 谱摘录（全 30 对见 `dpr_common_spectrum_*.csv`） |
| [`tables/AFFINE_EFFECTIVE_SUBSPACE.md`](tables/AFFINE_EFFECTIVE_SUBSPACE.md) | raw ΔW 有效子空间 R²_aff,k（30 对） |
| [`tables/AFFINE_LORA_BUDGET.md`](tables/AFFINE_LORA_BUDGET.md) | Aff/LoRA vs Vocab LoRA；n/d 与 aff@W1（**30 对**） |
| [`../docs/分析口径与特殊案例.md`](../docs/分析口径与特殊案例.md) | 全仓统一口径、BI 35/30/26 分层、排除规则与特殊案例原因 |
| [`notes/01_excluded_models_gemma4_dsv4.md`](notes/01_excluded_models_gemma4_dsv4.md) | 旧链接兼容入口，转向全仓统一口径文档 |
| [`notes/02_affine_effective_update_insight.md`](notes/02_affine_effective_update_insight.md) | **主 Insight**：affine 为何能解释 U 侧更新（R²_aff,K + rank-L + 实测） |
| [`notes/03_full_vocab_affine_geometry.md`](notes/03_full_vocab_affine_geometry.md) | Task6 full-vocab affine：`A-I`、`E_delta`、归一化 rank / effective-rank、energy@%h |

## 主分析组约定

当前 BI 仿射 / 有效子空间等结论默认 **non-excluded 30 对**（见 `bi_pairs.yaml`）：

- **17 tied**（E=U 同矩阵；有效子空间只算 E）
- **13 untied**（main 9 + extended 4；E/U 分列）

分层：

| tier | n | 说明 |
|------|---|------|
| **non-excluded（当前主叙事）** | **30** | main 26 + extended 4；排除 Gemma-4×4、Gemma-3-1B |
| main | 26 | 历史 aff/LoRA 批次；数字已并入 30 行 `affine_lora_budget_summary.csv` |
| extended | 4 | DeepSeek V3/V3.1、Qwen3-30B-A3B、Qwen3.5-35B-A3B；budget 已补算并入 30 行主表 |
| excluded | 5 | 见 [`BI-excluded`](../docs/分析口径与特殊案例.md#42-bi-excluded-5-对) |

全库注册 **35 对**（`configs/base_instruct_pairs.yaml`）。

DeepSeek-V4 **无 Base–Instruct 配对**，不在 35 对内；其问题属于**静态 E/U 几何不可比**，见 [`dsv4_hc_head`](../docs/分析口径与特殊案例.md#63-dsv4_hc_head)。

**结论一致性**：extended 4 并入 30 对后，E/U 分裂、U affine-friendly、tied≈untied U 等结论**方向不变**（untied U vs E 仍 13/13）。

## 本目录数据表（均为 **30 对** 尺度）

完整说明见 [`tables/README.md`](tables/README.md)。

| 表 | 行数 | 说明 |
|----|------|------|
| `tables/affine_effective_subspace.csv` | 43 | 有效子空间（17 tied×1 + 13 untied×2） |
| `tables/affine_lora_budget_summary.csv` | **30** | Aff/LoRA budget 逐模型（26 main + 4 extended 合并） |
| `tables/affine_lora_by_tied_summary.csv` | 2 组 | tied / untied 汇总 |
| `tables/delta_w_r2_one_minus.csv` | **30** | 1−R² 变化量 |
| `tables/dpr_common_spectrum_{e,u}.csv` | 各 **30** | D/P/R common-k 谱 |
| `tables/task6_decomposition_{e,u}.csv` | 各 **30** | Task6 centered 分解 |

## 外部引用（脚本与原始流水线仍在原处）

- 异常 pair 汇总：`ijcai_clean/results/_analysis/anomaly_group_base_instruct.csv`
- 行范数 / 谱 audit：`analysis_eu_geometry/results/BI_ROW_NORMS_SUMMARY.md`、[`docs/分析口径与特殊案例.md`](../docs/分析口径与特殊案例.md)
- 仿射主组过滤与 Task6 脚本：`__tep/scripts/*`、`__tep/affine/analysis/tasks/task6_pred_delta_probe.md`
- 完整 aff/LoRA 流水线原文：`__tep/affine/INSIGHTS.md`、`__tep/affine/README.md`
