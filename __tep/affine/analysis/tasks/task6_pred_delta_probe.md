# Task6 D/P/R 与 W-rank：BI-clean 30

## 范围

输入为 Task6 的35对 full-vocab 结果，排除 `Gemma-3-1B` 和四对 `Gemma-4`，最终30对。E/U decomposition 表、common-spectrum 表和 W-rank 表均验证包含30个唯一 `model_a`。

## 分解

```text
X_c = X - mean(X)
Y_c = Y - mean(Y)
D = Y_c - X_c
A = argmin ||Y_c - X_c A||²
P = X_c(A-I)
R = Y_c - X_c A
D = P + R
```

最小二乘正交性给出 `<P,R>_F≈0`，因此 `||D||²≈||P||²+||R||²`。主指标 `P/D=||P||²/||D||²` 衡量 affine component 在真实 centered update 中的占比。

## 30对结果

| 分组 | n | E P/D median | U P/D median |
|---|---:|---:|---:|
| 全部 | 30 | 0.1080 | 0.3098 |
| tied | 17 | 0.2960 | 0.2960 |
| untied | 13 | 0.0491 | 0.3188 |

## W-rank budget

统一档位为 `1,2,4,8,16,32,64,128`，每档30对。U 侧在 rW=1/2/4/8 的 affine 胜出数分别为27/26/24/22；E 侧为16/16/15/15。

结果说明 U 侧 affine component 更稳定，但优势随 W-rank 墦大而减弱。完整数字见 `../../tables/{e,u}/*w_rank_budget_summary.csv`。

## Hybrid 状态

旧 hybrid 产物未覆盖全部30对，已删除。只有在 GPU 空闲时使用 `evaluate_hybrid_affine_w_budget.py --all-clean` 完整重跑 E/U 两侧，并验证每侧30个唯一模型后，才能恢复 hybrid 结论。
