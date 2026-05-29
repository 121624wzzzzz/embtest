# Task5：子采样仿射关系分析

## 实验设定

| 项目 | 值 |
|------|-----|
| 配置 | `configs/affine_pairs.yaml`（Task1–4 pair 并集） |
| 输出 | `summary_pair.csv`（338 pairs）、`summary_intra_EU.csv`（92 models） |
| 拟合 | `Y ≈ XA + b`，最多 `max_fit_rows=24000` 行 |
| 指标 | `R2_E`, `rel_err_E`, `norm_A_E`, `norm_b_E`（U 同理） |

## 全体 pair：跨模型仿射普遍差

| 范围 | R2_E 均值 | R2_E 中位数 |
|------|-----------|-------------|
| 338 pairs | **0.429** | 0.354 |

**证据**：`task5_affine_subsampled/summary_pair.csv` 列 `R2_E`。

**解释假设**：跨模型 / 跨 vocab / 跨 hidden 时，全局仿射不足以刻画空间差异；与子采样噪声叠加。

## Task1 子集：与 Task6 高度一致

| 子集 | n | R2_E 均值 | R2_E 最小 |
|------|---|-----------|-----------|
| `source_tasks` 含 `task1_base_instruct` | 31 | **0.942** | 0.404 |

- Task5 vs Task6（同 31 对）`R2_E` ↔ `E_R2`：**r ≈ 0.999**（n=31）。
- **结论（数据事实）**：对 Base-Instruct，**2.4 万行子采样** 已足以估计 full-vocab 仿射质量；Task6 主要用于 A-I / SVD 诊断而非刷新 R²。

## 模型内部 E→U（intra）

`summary_intra_EU.csv`：92 个模型的 E→U 仿射 `R2_EU`。汇总表：[`tables/affine_task5_intra_by_tied.csv`](../../tables/archive/affine_task5_intra_by_tied.csv)。

| actual_tied | n | R2_EU 均值 | 中位数 | R2&lt;0.5 |
|-------------|---|------------|--------|----------|
| **True** | 44 | **1.000** | 1.000 | 0 |
| **False** | 48 | **0.348** | 0.290 | **47** |

**证据**：`task5_affine_subsampled/summary_intra_EU.csv` 列 `actual_tied`, `R2_EU`（2026-05-19 聚合）。

| 现象 | 数据 |
|------|------|
| tied 模型 E→U **完美** | 共享矩阵下 E=U，R²≡1 为定义结果，非「几何奇迹」 |
| untied 普遍低 R² | 47/48 &lt;0.5；与 `_analysis/task5_low_intra_EU_untied.csv` 一致 |
| 极低示例 | Qwen2.5-7B `R2_EU≈0.206`；Qwen3-8B ≈0.255；DeepSeek-R1-0528 ≈0.473 |

**解释假设**：声明 untied 时 E/U 几何本就不追求全局仿射一致；与 Task1 中 untied 对 **BI 上** E≈U GCorr 仍高并不矛盾（GCorr 比 token-pair 结构，非 E→U 线性映射）。

## Tied 报告

`base_instruct_affine_tied_report.md`：从 Task5 抽取 Base-Instruct，按 tied/untied 汇总 — Pass 2 可与之对照 `actual_tied` 列。

## 使用注意

| 正确用法 | 错误用法 |
|----------|----------|
| Task5 Task1 子集 ↔ Task6 互相验证 R² | 用 Task5 全体均值 0.43 否定 Base-Instruct 仿射 |
| Task5 intra 讨论 E/U 解耦 | 用 Task5 跨模型 R² 代表 IT 效应 |
