# Centered Affine Update 分解

本文的实证范围固定为 BI-clean 30。

## 1. 最小二乘分解

令 `X,Y∈R^{n×d}`，中心化后为 `X_c,Y_c`。定义：

```text
D = Y_c - X_c
A = argmin_A ||Y_c-X_cA||²_F
P = X_c(A-I)
R = Y_c-X_cA
```

则 `D=P+R`。正规方程给出 `X_c^T R=0`，而 `P` 属于 `X_c` 的列空间，因此：

```text
<P,R>_F = 0
||D||²_F = ||P||²_F + ||R||²_F
```

所以：

```text
P/D = ||P||²_F / ||D||²_F
R/D = ||R||²_F / ||D||²_F
P/D + R/D = 1
```

`P/D` 是 update-scale 上的 affine 可解释比例，比接近1的普通 full-matrix R2 更有辨识力。

## 2. Raw update 与 mean shift

令 `Δ_raw=Y-X`、`δμ=mean(Y)-mean(X)`，则：

```text
||Δ_raw||²_F = ||D||²_F + n||δμ||²_2
```

centered affine component 与行均值平移是正交的两部分，报告 raw-scale 指标时必须同时计入 mean shift。

## 3. BI-clean 30 实证

- 全部30对：E P/D median 0.1080，U P/D median 0.3098。
- untied 13对：E 0.0491，U 0.3188。
- tied 17对：E/U 均为0.2960，因为两侧共享同一矩阵。
- `A-I` mean rank95/h 为0.4528，直接 E delta 为0.7837。

这些结果支持“U 侧全局 affine component 更强且更集中”，但不支持把完整 update 描述为普遍极低秩。

## 4. 适用边界

- 结论仅针对非异常30对，不把5个排除 pair 混入聚合。
- W-rank 优势是 budget-dependent，不是对所有 rank 成立。
- Hybrid 当前没有完整30对数据，不作为证据。
