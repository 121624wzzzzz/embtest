# bi_analysis/tables 数据目录

**默认尺度：non-excluded 30 对 BI pair**（17 tied + 13 untied；见 [`../bi_pairs.yaml`](../bi_pairs.yaml)）。  
排除 5 对：Gemma-3-1B、Gemma-4×4。

除非文件内另有说明，**不要**把本目录表当作 26 对 main 组使用；26 只是历史计算批次标签（`analysis_tier: main`），叙事以 30 对为准。

---

## 一览

| 文件 | 行数 / 规模 | 内容 | 生成方式 |
|------|-------------|------|----------|
| [`affine_effective_subspace.csv`](affine_effective_subspace.csv) | 43 行 | raw ΔW=Y−X 在 Z=[X,1] 有效子空间指标（η=90/95/99%） | `__tep/scripts/evaluate_affine_effective_subspace.py` |
| [`AFFINE_EFFECTIVE_SUBSPACE.md`](AFFINE_EFFECTIVE_SUBSPACE.md) | — | 上表可读汇总 + 与 insight 链接 | 手工维护 |
| [`affine_lora_budget_summary.csv`](affine_lora_budget_summary.csv) | **30 行** | 逐模型 E/U：P/D、Task6 谱、Aff vs W-rank budget、hybrid 稳定性 | [`../scripts/build_affine_lora_budget_30.py`](../scripts/build_affine_lora_budget_30.py) |
| [`affine_lora_by_tied_summary.csv`](affine_lora_by_tied_summary.csv) | 2 组 | tied / untied 分组中位数、aff wins | 同上脚本 |
| [`affine_lora_by_family_size_summary.csv`](affine_lora_by_family_size_summary.csv) | 多组 | 按 family×size 汇总 | 同上脚本 |
| [`AFFINE_LORA_BUDGET.md`](AFFINE_LORA_BUDGET.md) | — | Aff/LoRA 实验可读汇总（30 对） | 手工维护 |
| [`w_rank_budget_extended_e.csv`](w_rank_budget_extended_e.csv) | 4×8 | extended 4 对 **E 侧** W-rank sweep（中间产物） | `__tep/scripts/evaluate_w_rank_budget.py` |
| [`w_rank_budget_extended_u.csv`](w_rank_budget_extended_u.csv) | 4×8 | extended 4 对 **U 侧** W-rank sweep（中间产物） | 同上 |
| [`delta_w_r2_one_minus.csv`](delta_w_r2_one_minus.csv) | **30 行** | 1−R²_identity、1−R²_full_affine、R² gain | 合并 Task6 + rank1 表 |
| [`DELTA_W_R2_ONE_MINUS.md`](DELTA_W_R2_ONE_MINUS.md) | — | 上表按族排列的可读版 | 手工维护 |
| [`task6_decomposition_e.csv`](task6_decomposition_e.csv) | **30 行** | E 侧 centered D/P/R 分解与谱（Task6） | 自 `__tep/affine/tables/e/` 复制 |
| [`task6_decomposition_u.csv`](task6_decomposition_u.csv) | **30 行** | U/lm_head 侧 Task6 分解 | 自 `__tep/affine/tables/u/` 复制 |
| [`dpr_common_spectrum_e.csv`](dpr_common_spectrum_e.csv) | **30 行** | E 侧 D/P/R **common-k** 累积能量谱 C(k) | 自 `__tep/affine/tables/e/` 复制 |
| [`dpr_common_spectrum_u.csv`](dpr_common_spectrum_u.csv) | **30 行** | U 侧 common-k 谱 | 自 `__tep/affine/tables/u/` 复制 |
| [`extended_untied_dpr_common_spectrum.csv`](extended_untied_dpr_common_spectrum.csv) | 8 行 | extended 4 对 E/U 长表（**`dpr_common_spectrum_*` 的子集**） | `merge_extended_common_spectrum.py` |
| [`EXTENDED_DPR_COMMON_SPECTRUM.md`](EXTENDED_DPR_COMMON_SPECTRUM.md) | — | extended 4 对 D/P/R 谱可读摘录 | 手工维护 |

---

## 按分析主题

### 有效 Affine 子空间（线 C，主叙事）

- **问**：raw ΔW 有多少能量落在 Z=[X,1] 的低维有效子空间 Q_K？
- **表**：`affine_effective_subspace.csv` + `AFFINE_EFFECTIVE_SUBSPACE.md`
- **笔记**：[`../notes/02_affine_effective_update_insight.md`](../notes/02_affine_effective_update_insight.md)

### Aff/LoRA vs Vocab LoRA（Task6 budget）

- **问**：同 W-rank 参数预算下，hidden Aff/LoRA 是否比 W-form 解释更多 Δ？
- **表**：`affine_lora_budget_summary.csv` + `AFFINE_LORA_BUDGET.md`
- **extended 4 对**：2025-05 补跑 `w_rank_budget_extended_*.csv` 后并入 30 行主表
- **26 对 main**：budget 数字来自 `__tep/affine/tables/final/` 原始汇总（未改矩阵）

### 变化量 R²（1−R² 口径）

- **问**：BI 改动相对 Instruct 能量有多大？仿射拟合提升多少？
- **表**：`delta_w_r2_one_minus.csv`

### D/P/R 分解与 common-k 谱（centered，线 A）

- **问**：centered Δ 里 P=X_c(A−I) 占多少？谱上 C_P vs C_D？
- **表**：`task6_decomposition_*.csv`、`dpr_common_spectrum_*.csv`
- **extended 子集**：`extended_untied_dpr_common_spectrum.csv`（与全表重复，便于单独查阅 MoE/DeepSeek）

**Raw ΔW = Y−X 对照（勿只读 centered）**

线 A 主指标 **P/D = ‖P‖²/‖D‖²** 在 **centered** 变化量 D=Y_c−X_c 上定义（P=X_c(A−I)，不含 bias 项 1_n b^T）。  
但 BI 的真实更新首先是 **raw ΔW = Y−X**（行不去中心化）。二者通过正交分解相连：

```text
‖ΔW‖² = ‖D‖² + n‖μ_Y−μ_X‖²     （mean-shift 占 raw Δ 通常仅 2–8%）
```

因此在 **raw ΔW** 上应同时报告仿射可解释占比，避免两种误读：

| 误读 | 实际情况（untied 13 对中位，η=95%） |
|------|--------------------------------------|
| 「只有 centered P/D 高，raw 不行」 | raw 上 **R²_aff,full ≈ 36%**、**R²_aff,K ≈ 33%**（U）；与 **P/D ≈ 32%** 同档 |
| 「只有 raw 高、一 centered 就没了」 | E/U 分裂在 raw 与 centered **同向**：E **~5%**，U **~30–36%**；mean-shift 很小，去中心化几乎不改变结论 |

| 口径 | 对象 | U untied 中位 | E untied 中位 | 表 / 列 |
|------|------|---------------|---------------|---------|
| **Raw ΔW** | ΔW=Y−X 落在 Z=[X,1] 的比例 | R²_aff,full **~36%**；R²_aff,K **~33%** | **~5%** / **~4%** | `affine_effective_subspace.csv`（线 C）；[`AFFINE_EFFECTIVE_SUBSPACE.md`](AFFINE_EFFECTIVE_SUBSPACE.md) |
| **Centered D** | P/D=‖X_c(A−I)‖²/‖D‖²（线性项） | **~32%** | **~5%** | `affine_lora_budget_summary.csv` 的 `*_P_over_D_full_affine_gain`；`dpr_*`.`full_affine_gain_over_delta` |
| **Mean-shift** | n‖μ_Y−μ_X‖² / ‖ΔW‖² | **~8%** | **~2%** | `task6_decomposition_*`.`Mean_shift_over_raw_delta_energy` |

**叙事要点**：U 侧 affine 结构在 **raw ΔW 与 centered D 上同时 substantial**（约三分之一量级），不是某一口径独有；E 侧两侧均弱。完整推导与 rank-L 分解见 [`../notes/02_affine_effective_update_insight.md`](../notes/02_affine_effective_update_insight.md) §5.4。

---

## 列名速查

### `affine_lora_budget_summary.csv`

| 列 | 含义 |
|----|------|
| `analysis_tier` | `main`（26）或 `extended`（4） |
| `*_P_over_D_full_affine_gain` | ‖P‖²/‖D‖²，D=Y_c−X_c，P=X_c(A−I) |
| `*_aff_vs_W_ratio_r{k}` | 同 centered 解释量下 aff gain / W gain（>1 = affine 优） |
| `*_aff_rank_budget_r{k}` | W rank=k 时参数量匹配的 aff rank（≈ k·(n+d)/(2h)） |
| `*_hybrid_stable_small_budget_both` | rW∈{1,2,4,8} 四档 aff centered gain 均 > W |

### `affine_effective_subspace.csv`

| 列 | 含义 |
|----|------|
| `side` | `E` 或 `U`（tied 仅 E 行） |
| `eta` | 有效子空间能量阈值 0.90 / 0.95 / 0.99 |
| `K` | Q_K 维度 |
| `r1,r5,r10` | ΔW 前 L 奇异方向能量占比 |
| `a1_k,A5_k,R2_aff_k` | 对齐度与全谱仿射可解释占比 |

---

## 重建命令

```bash
# 30 对 Aff/LoRA 主表（需先有 w_rank_budget_extended_*.csv）
python3 bi_analysis/scripts/build_affine_lora_budget_30.py

# extended 4 对 W-rank sweep（GPU，各 ~1–2 min）
python3 __tep/scripts/evaluate_w_rank_budget.py \
  --models DeepSeek-V3-Base DeepSeek-V3.1-Base Qwen3-30B-A3B-Base Qwen3.5-35B-A3B-Base \
  --matrix-kind embed --w-ranks 1 2 4 8 16 32 64 128 --device cuda:5 \
  --out bi_analysis/tables/w_rank_budget_extended_e.csv
# lm_head 侧把 --matrix-kind embed 改为 lm_head，--out ..._u.csv
```

有效子空间与 common-k 谱的**全量重算**仍在 `__tep/scripts/`；本目录以**归档 + 30 对合并**为主。

---

## 仍只在 `__tep/affine/` 的数据

- 全量 W-rank / hybrid sweep 明细（main 26，`unembed_w_rank_budget_sweep_all_main.csv` 等）
- 理论推导、Task 流水线文档
- 若需逐 rank 曲线，到 `__tep/affine/tables/` 查原表
