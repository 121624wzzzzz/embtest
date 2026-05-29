# Task6：Base-Instruct Full-Vocab 仿射与 SVD

## 实验设定

| 项目 | 值 |
|------|-----|
| 入口 | `run_base_instruct_full_vocab_affine.py` |
| 输出 | `summary_pair_base_instruct_full_vocab.csv`（31 行） |
| 对齐 | 全词表 id，`n_fit = n_common = vocab_size` |
| 对象 | `E_base → E_instruct`；tied 时 U 复用 E |

## Full-vocab R²（数据事实）

| 分组 | n | E_R2 均值 | E_R2 中位数 |
|------|---|-----------|-------------|
| 全体 | 31 | 0.936 | 0.996 |
| 主分析组（无 Gemma-3-1B/4） | 26 | **0.991** | 0.997 |
| 异常组 | 5 | 0.650 | 0.707 |
| Qwen only | 16 | 0.996 | 0.998 |
| Llama only | 4 | 0.988 | 0.989 |
| Gemma only | 11 | 0.830 | 0.958 |
| Gemma only（无 1B/Gemma-4） | 6 | 0.981 | 0.981 |
| 主组 untied | 9 | **0.998** | 0.999 |
| 主组 tied | 17 | 0.988 | 0.991 |

**证据**：列 `E_R2`；主组/异常分组见 [`laws_old.md`](laws_old.md) §1。

**主组内部差异**：去掉 Gemma 异常后，低一档 pair 基本都在 `actual_tied=True` 中；untied 的 9 对 min 仍为 0.9956。tied 仍包含高 R² pair，因此 tied 更像调节变量，而不是单独的失效条件。

## A 矩阵诊断（主分析组 n=26）

| 指标 | 典型范围 | 含义 |
|------|----------|------|
| `E_rel_A_minus_I_over_I` | 0.007–0.11 | 近单位阵 |
| `E_identity_cosine` | >0.996 | 方向贴近 I |
| `E_offdiag_norm_over_A` | <0.09 | 坐标混合弱 |

异常组 Gemma-3-1B：`rel_A_minus_I≈0.911`，`identity_cosine≈0.421` — **仿射仍优于“无变换”但远离 I**。

**证据**：Task6 CSV 列 `E_rel_A_minus_I_over_I`, `E_identity_cosine`。

## SVD：E_delta vs A-I（主分析组）

| 归一化指标 | E_delta | A-I | 比值（均值） |
|------------|---------|-----|--------------|
| rank_95 / h | 0.769 | 0.426 | A-I 更低 |
| effective_rank / h | 0.418 | 0.203 | ≈0.48× |
| energy@5%h 中位数 | 0.397 | **0.628** | A-I 更集中 |

**证据**：列 `E_delta_E_delta_*`、`A_delta_A_delta_*`；聚合见 `computed_stats.json` → `task6.main_*`。

### 分析流水线：为什么说 A-I 的秩更低

这里的“秩更低”指谱意义上的有效秩 / 能量集中度（`rank95/h`, `eff_rank/h`, `energy@5%h`），不是严格代数秩。

| 步骤 | 做什么 | 产出 / 判断 |
|------|--------|-------------|
| 1. 定义主组 | 31 对 BI 中排除 `Gemma-3-1B` 与 `Gemma-4-*` | 得到主组 n=26，避免异常低 R² 混入 |
| 2. 拟合仿射 | full-vocab id 对齐，拟合 `Y≈XA+b` | 主组 `E_R2` 均值 **0.991**、中位 **0.997**，说明 `A-I` 是有解释力的全局参数 |
| 3. 构造两个谱对象 | naive delta：`G_delta=(Y-X)^T(Y-X)`；仿射偏移：`G_A=(A-I)^T(A-I)` | 两者都是 `h×h` Gram，可用同一组谱指标比较 |
| 4. 同口径比较 | 比 `rank95/h`、`eff_rank/h`、`energy@5%h` | 主组 **26/26** 对均满足：`A-I` rank95 更低、effective rank 更低、energy@5%h 更高 |
| 5. 解释差距来源 | 分解 `Y_c-X_c=X_c(A-I)+R` | `E_delta` 是仿射预测项、residual、mean shift 的混合；它不等价于 `A-I` |
| 6. 写作结论 | 低秩发现归于 `A-I`，高 R² 作为解释条件 | “高 R² 的全局仿射更新，其参数偏移 `A-I` 谱集中；raw delta 因混入 residual 更分散” |

逐对一致性：

| 判断 | 主组成立数 |
|------|------------|
| `A_I_rank95_over_h < E_delta_rank95_over_h` | **26/26** |
| `A_I_eff_rank_over_h < E_delta_eff_rank_over_h` | **26/26** |
| `A_I_energy_5pct_h > E_delta_energy_5pct_h` | **26/26** |

这条流水线的关键是先把 `A-I` 当作全局仿射算子的参数来比较，而不是把 `E_delta` 当作同一个对象。高 `E_R2` 的作用是说明这个参数有解释意义；`rank95/h` 与 `energy@5%h` 才是低秩/集中性的直接证据。

### 系列内 eff_rank/h 线性相关（Pearson r）

| 系列 | r | n |
|------|---|---|
| Qwen3.5 | 0.980 | 4 |
| Qwen2.5 | 0.957 | 7 |
| Gemma-2 | 0.907 | 3 |
| Gemma-3 | 0.905 | 3（含 4B/12B/27B，**不含 1B**） |
| Qwen3 | 0.843 | 5 |

**解释假设**：IT 引起的 **直接 embedding 差** 越高秩，最优仿射相对 I 的偏移越可被少数主方向压缩 — 同族内近似线性。

## 异常 pair 快照

| model_a | E_R2 | rank95_E/h | rank95_A-I/h | energy@5%h E / A-I |
|---------|------|------------|--------------|---------------------|
| Gemma-3-1B | 0.375 | 0.865 | 0.808 | 0.19 / 0.13 |
| Gemma-4-E2B | 0.668 | 0.893 | 0.542 | 0.33 / 0.53 |
| Gemma-4-31B | 0.776 | 0.903 | 0.383 | 0.21 / 0.73 |

**反直觉点（数据事实）**：部分 Gemma-4 的 **E_delta rank95/h 很高**，但 **A-I energy@5%h 仍较高** — 说明“差分矩阵高秩”与“偏移谱集中”可并存；不宜只用 rank95 单指标。

## 与 Task1 GCorr 的关系

- 全体 31 对上 `gcorr_E_cos` 与 `E_R2` 的 Pearson 虽为 r≈0.93，但主要来自主组/异常组分离；主组内部两者都接近 1，存在 ceiling effect，不宜把它写成连续预测关系。
- `gcorr_E_euc` 与 `E_R2` 的全体相关较弱（r≈0.81），也只作诊断参考；Gemma-4 上 euc 先崩溃而 cos 仍中等时尤其明显。
- 见 `key_metrics.csv` 逐行对照。
