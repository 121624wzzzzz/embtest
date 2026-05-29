# 论文 B — Affine / Low-Rank Update

> 先读 [`INSIGHTS.md`](INSIGHTS.md)。BI 叙事侧见 [`../../bi_analysis/README.md`](../../bi_analysis/README.md)（**non-excluded 30 对** = main 26 + extended 4）。旧的分任务细节仍保留在 [`analysis/`](analysis/)，机器可读结果在 [`tables/`](tables/)。

## 一句话

Base→Instruct 的词表侧变化不能只用普通矩阵 R2 来讲；真正有信息的是小 update `D=Y_c-X_c` 中有多少能量来自全局 affine component `P=X_c(A-I)`。在 input embedding 侧这个结构有边界；在 unembedding/lm_head 侧，尤其 untied 模型中，它显著更强。

## 主线结果

| 现象 | 结果 |
|------|------|
| 普通矩阵接近 | E/U 的 identity R2 已经普遍很高，不能单独作为 affine 有效性的强证据 |
| E 侧 update-scale | `P/D` median **0.120**；affine-only 低 W-rank 窗口内胜出 |
| U 侧 update-scale | `P/D` median **0.315**；同预算 affine/hybrid 更稳定 |
| tied 分层 | tied 17 对 E/U 等价，`P/D` median **0.296** |
| untied 分层 | E: `P/D` median **0.049**，U: **0.462**；E rW1 wins **0/9**，U rW1 wins **9/9** |
| hybrid | U 侧 `rW=2/4/8` 同预算 hybrid **26/26** 胜 pure W |

## 必读文件

| 文件 | 内容 |
|------|------|
| [`INSIGHTS.md`](INSIGHTS.md) | 最终结论、数字和故事线 |
| [`tables/final/model_level_e_u_affine_lora_summary.csv`](tables/final/model_level_e_u_affine_lora_summary.csv) | 逐模型 E/U 总表 |
| [`tables/final/model_level_e_u_by_tied_summary.csv`](tables/final/model_level_e_u_by_tied_summary.csv) | tied/untied 汇总 |
| [`tables/final/model_level_e_u_by_family_size_summary.csv`](tables/final/model_level_e_u_by_family_size_summary.csv) | 按模型族/尺寸汇总 |
| [`analysis/tasks/task6_pred_delta_probe.md`](analysis/tasks/task6_pred_delta_probe.md) | residual-aware、W-rank budget、hybrid、U 侧补算 |
| [`analysis/theory/affine_decomposition_proof.md`](analysis/theory/affine_decomposition_proof.md) | 分解与理论推导 |

## 目录说明

| 路径 | 用途 |
|------|------|
| [`analysis/archive/laws_old.md`](analysis/archive/laws_old.md) | 早期跨 Task 规律总结，保留作背景 |
| [`analysis/tasks/`](analysis/tasks/) | 当前主实验流水线 |
| [`analysis/theory/`](analysis/theory/) | 当前理论证明 |
| [`analysis/archive/`](analysis/archive/) | 被后续结果替代的旧理论备忘 |
| [`tables/`](tables/) | 派生结果表；见 [`tables/README.md`](tables/README.md) |
