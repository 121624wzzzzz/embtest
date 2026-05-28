# 特殊案例排错：Gemma-4 与 DeepSeek-V4

本页把两个容易误读 E/U 几何的案例放在一起：

- **Gemma-4**：checkpoint 存盘 E 行范数存在 near-unit-sphere gauge，导致 raw Euclidean GCorr 被径向标尺差放大。
- **DeepSeek-V4**：裸 `head.weight` 不等价于传统完整 `lm_head`；输出端有 `hc_head + RMSNorm + head.weight`，使 U 的裸谱几何不能直接按传统 lm_head 解读。

## 1. Gemma-4：row-norm gauge

主线 Task1 中，Gemma-4 Base-Instruct 出现「cosine GCorr 仍高，但 raw Euclidean GCorr 明显偏低」的现象。行范数审计显示：

| 模型 | E mean | var 量级 | 说明 |
|------|--------|----------|------|
| Gemma-4 Base | ≈ 1.000000 | 10^-9 到 10^-7 | 存盘 E 行几乎在单位球面 |
| Gemma-4 Instruct | 1.12 到 1.52 | 10^-2 左右 | SFT 后径向标尺明显漂移 |

这说明 Gemma-4 Base 的 E 不是普通随机行范数，而是近似固定半径的 checkpoint gauge。raw Euclidean 同时比较方向和半径；当 Base 近单位球、Instruct 半径变大时，raw Euclidean 会把径向差也计入不相似度，从而低估方向结构的一致性。

**排错结论**：

- 对 Gemma-4 BI，优先使用 cosine 或 row-normalized Euclidean。
- raw Euclidean 若保留，应明确标注受 checkpoint row-norm gauge 影响。
- forward 中的 `sqrt(hidden_size)` 是 runtime activation scale，不解释 checkpoint E 行范数为何恰好接近 1。

## 2. DeepSeek-V4：HC head 使裸 U 不等价于传统 lm_head

untied 模型里，通常观察到 **U 未中心化 rank1 > E**：裸 lm_head 行向量往往有更强公共 mean 方向。但 DeepSeek-V4-Flash / Pro 是例外：

| 模型 | 矩阵 | mu/mean(row) | rank1 | rank1(c) |
|------|------|--------------|-------|----------|
| DeepSeek-V4-Flash | E | 0.237 | 0.055 | 0.021 |
| DeepSeek-V4-Flash | U | 0.110 | 0.028 | 0.019 |
| DeepSeek-V4-Pro | E | 0.180 | 0.035 | 0.019 |
| DeepSeek-V4-Pro | U | 0.108 | 0.027 | 0.019 |

这里的反常不是 U 残差特别各向同性，而是 **E 有更强 mean-shift 公共轴**。中心化后 `rank1(c)` 近似相同，说明 E/U 残差谱并未出现明显低秩塌缩。

### 2.1 Forward 路径

传统模型的输出通常可简化为：

```text
logits = U · RMSNorm(h)
```

DeepSeek-V4 的最后输出路径更接近：

```text
HC states -> hc_head -> RMSNorm -> head.weight -> logits
```

设最后一层输出是多路 HC state：

```text
H = [h1, h2, h3, h4],  hi ∈ R^d
```

`hc_head` 先根据当前 H 计算动态 mixing 权重：

```text
alpha = f(H)
```

再合并：

```text
y = Σ_i alpha_i(H) h_i
z = RMSNorm(y)
logits = U z
```

所以 V4 的完整输出映射应看作：

```text
logits = U · RMSNorm(hc_head(H))
```

而不是传统意义上的：

```text
logits = U · RMSNorm(h)
```

因为 `alpha_i` 依赖输入状态 H，`hc_head` 不能被吸收到一个固定矩阵中。裸 `head.weight` 只是完整输出头的一部分。

### 2.2 Backward 梯度流

训练时梯度从 loss 回传：

```text
loss
  <- logits
  <- head.weight
  <- RMSNorm
  <- hc_head
  <- h1, h2, h3, h4
  <- previous HC blocks
  <- embed.weight
```

对每一路 state，`hc_head` 带来两类梯度：

1. **直接混合梯度**：

```text
alpha_i · ∂L/∂y
```

被 `hc_head` 更重视的 stream 会接收更强的输出监督。

2. **gating / mixing 梯度**：

```text
∂L/∂alpha_i · ∂alpha_i/∂H
```

模型不仅学习每条 state 存什么内容，还学习当前 token / 上下文下如何合并这些 state。

因此 `hc_head` 同时是：

- 输出表示合并器；
- 梯度路由器；
- 多路 residual stream 的出口。

### 2.3 对 E/U 谱几何的含义

V4 主 forward 中，`embed.weight` 直接进入 HC state：

```text
input_ids -> embed.weight -> HC states
```

而 `head.weight` 面对的是经过动态 `hc_head` 合并与 RMSNorm 后的 `z`。因此：

- E 更可能承担初始化 HC state 的全局偏置 / 公共轴角色；
- 裸 U 不需要像传统 lm_head 那样携带同等强度的公共 mean 方向；
- 对 V4，裸 U 的谱几何不能直接代表完整 output geometry。

这解释了为什么 DeepSeek-V4 两个模型出现 **E 未中心化 rank1 > U**，而 DeepSeek-V3/R1 等普通 untied 模型通常是 **U rank1 > E**。

**排错结论**：

- V4 的例外更像架构导致的 E/U 职责变化，而不是统计噪声。
- 若分析 V4 的真实 output geometry，应看 `hc_head + RMSNorm + head.weight` 或局部 `∂logits/∂H`，不能只看裸 `head.weight`。
- 对主线 GCorr，若只比较 checkpoint E，V4 的 U 例外不应反向污染 E 的结论；若比较 U，则需额外说明 V4 U 不是传统完整 lm_head。
- 当前审计只能分析存盘的裸 E / 裸 U，暂时无法可靠刻画 `hc_head + RMSNorm + head.weight` 的有效输出几何。因此，对本目录以 **embedding 存盘几何** 为主的分析来说，DSV4 系列的裸 U 谱结果解释意义较低，应主要作为架构例外记录，而不作为普通 untied 模型规律的证据。

