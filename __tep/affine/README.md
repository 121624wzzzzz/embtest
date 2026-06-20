# Affine / Low-Rank Update

本模块只使用 **BI-clean 30**（17 tied + 13 untied）。排除规则见 [`../../docs/分析口径与特殊案例.md`](../../docs/分析口径与特殊案例.md)。

核心对象为 centered update：

```text
D = Y_c - X_c
P = X_c(A-I)
R = D-P
P/D = ||P||_F^2 / ||D||_F^2
```

优先阅读 [`INSIGHTS.md`](INSIGHTS.md) 和 [`tables/README.md`](tables/README.md)。当前 W-rank 比较统一使用 `1,2,4,8,16,32,64,128` 八个档位，每档均有30对。旧 hybrid 表因未覆盖全部30对已删除。
