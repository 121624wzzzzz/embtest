# 项目分析备忘

本文档用于保留对当前 embedding / unembedding 几何项目的阶段性分析结论。它偏向研究判断和结果解读；具体指标定义、脚本入口和自动生成结果分别见 `docs/methods_and_metrics.md`、`README.md` 与 `results/`。

## 项目主线

当前项目围绕不同大语言模型的 embedding matrix `E` 与 unembedding / lm_head matrix `U` 展开，核心问题是：不同模型、同系列模型、Base-Instruct 模型之间的词向量空间是否存在稳定几何关系。

已有任务大致分为：

- Task1：Base-Instruct pair 的 GCorr 几何相关性。
- Task2：同一模型系列内部的 GCorr。
- Task3：跨模型系列、跨规模桶的 GCorr。
- Task5：对 Task1-4 pair 并集做仿射关系分析，拟合 `Y ~= X A + b`。
- Base-Instruct full-vocab 诊断：对 Base-Instruct pair 使用完整词表 id 对齐，进一步分析仿射矩阵 `A`、`A-I` 和 `E_instruct - E_base` 的 SVD 低秩结构。

## Task5 仿射关系的理解

Task5 的核心实验是给定两个矩阵 `X` 和 `Y`，拟合：

```text
Y ~= X A + b
```

其中 `A` 是线性变换矩阵，`b` 是平移项。主要用 `R2` 和 `rel_err` 衡量拟合效果。`R2` 越接近 1，说明两个空间之间越接近一个全局仿射关系。

对 Base-Instruct 这类同 tokenizer、同 hidden dimension 的 pair，更适合使用完整词表 id 对齐，而不是采样 token 行。当前专门脚本 `scripts/run_base_instruct_full_vocab_affine.py` 已经按这个方式重算，并输出到：

- `results/task5_affine_relations/summary_pair_base_instruct_full_vocab.csv`
- `results/task5_affine_relations/base_instruct_full_vocab_affine_report.md`

## Base-Instruct full-vocab 主要结果

完整词表分析显示，大部分 Base-Instruct pair 的 embedding 空间之间存在很强的仿射关系：

```text
all Base-Instruct: R2_E mean ~= 0.9364
Qwen only:         R2_E mean ~= 0.9964
Llama only:        R2_E mean ~= 0.9880
Gemma only:        R2_E mean ~= 0.8305
```

Qwen 和 Llama 的 Base-Instruct 变化基本可以被一个全局仿射变换很好解释。Gemma 组均值明显偏低，主要来自 Gemma-3-1B 和 Gemma 4 系列的异常表现。

## Gemma 异常判断

Gemma-3-1B 和 Gemma 4 暂时不作为主要规律来源。它们的低 `R2` 和高误差更可能来自模型架构与训练机制差异，而不是脚本错误或采样问题。

当前判断：

- `Gemma-3-1B` 的 full-vocab affine `R2_E` 约为 `0.3751`，明显脱离主流 Base-Instruct 规律。
- Gemma 4 多个模型的 `R2_E` 约在 `0.6682` 到 `0.7759`，也显著低于 Qwen / Llama。
- 这些异常 pair 在 `A` 诊断上也表现为更大 `A-I` 偏移、更低 identity cosine 或更强非正交误差。

更具体地说：

- Gemma 4 的异常应优先解释为架构原因。Hugging Face / Transformers 相关讨论明确提到 Gemma 4 的 Per-Layer Embeddings（PLE）机制，包含 `embed_tokens_per_layer`、逐层 token lookup、context projection、RMSNorm 和逐层输入组合；这意味着只分析常规输入 embedding `E` 与 lm_head / unembedding `U`，并不能完整覆盖模型实际使用的嵌入机制。需要注意的是，PLE 主要明确对应 E2B/E4B；26B-A4B 还涉及 MoE 路由，31B 是 dense 但仍属于 Gemma 4 的新长上下文 / 多模态架构族。因此“Gemma 4 异常正常”这个判断是合理的，但最好按 E2B/E4B、26B-A4B、31B 分开解释。参考：Hugging Face Gemma4 文档与 Transformers issue `Gemma4: PLE implementation is underdocumented and config is misleading`。
- `Gemma-3-1B` 更可能是小尺寸模型的训练配比问题：预训练容量较小，而后训练 / instruction tuning 占比或强度可能相对过高，导致 embedding 空间被后训练显著改写。公开材料目前没有直接把 `Gemma-3-1B` 定性为“过度后训练”，但 Gemma 3 技术报告明确说明其使用 distillation 和 novel post-training recipe；社区也有关于 `Gemma-3-1B-IT` 继续微调后输出异常、hallucination / gibberish 的讨论。再结合 overtrained language models 更难 fine-tune 的相关研究，这个解释是合理工作假设，而不是已被官方确认的事实。这个假设与它极低的 full-vocab affine `R2`、较大的 `A-I` 偏移和低 identity cosine 一致。

因此，后续总结主规律时建议明确区分：

- 主分析组：排除 `Gemma-3-1B` 和 Gemma 4。
- 异常组：Gemma-3-1B 与 Gemma 4，作为架构和训练差异导致的反例或单独讨论对象。

### Gemma-3-1B 单独检查

进一步检查后，`Gemma-3-1B` 的异常不像是 token id 对齐错误或少量特殊 token 污染，而更像是 base 到 instruct 的 embedding 表被整体大幅改写。

本地配置与提取层面的排查：

- `Gemma-3-1B` 和 `Gemma-3-1B-Instruct` 都是 `Gemma3ForCausalLM`，`hidden_size = 1152`，`vocab_size = 262144`，`tie_word_embeddings = true`。
- 两者提取的矩阵来源相同，都是 `model.embed_tokens.weight`，`E` 和 `U` 实际 tied。
- base / instruct 的 `tokenizer.model` checksum 一致，说明 SentencePiece token-id 主体不是错位的。
- `tokenizer.json` 和 `tokenizer_config.json` 不同，主要符合 instruct 模型需要 chat template / 额外 EOS 行为的预期；这不足以解释全词表 embedding 行向量同时大幅偏离。
- 跳过前 128 或 1024 个 token 后，`Gemma-3-1B` 的 row cosine 仍然约为 `0.279`，说明异常不是由特殊 token 或 reserved token 主导。

与同系列大模型和小模型对照：

```text
pair                         global cosine  mean row cosine  rel_delta/base
Gemma-3-1B -> 1B-Instruct       0.2798          0.2794          1.1460
Gemma-3-4B -> 4B-Instruct       0.9760          0.9759          0.2236
Gemma-3-12B -> 12B-Instruct     0.9817          0.9817          0.1903
Gemma-3-27B -> 27B-Instruct     0.9840          0.9839          0.1785
Gemma-2-2B -> 2B-Instruct       0.9995          0.9995          0.0326
Qwen2.5-0.5B -> Instruct        0.9939          0.9934          0.1116
Llama-3.2-1B -> Instruct        0.9817          0.9810          0.1946
```

`Gemma-3-1B` 的 row cosine 分布也非常特殊：

```text
all rows:       mean ~= 0.2794, p01 ~= 0.1225, median ~= 0.2802, p99 ~= 0.4237
skip first 128: mean ~= 0.2794
skip first 1024:mean ~= 0.2792
```

相比之下，`Gemma-3-4B` 即使跳过同样 token 区间，mean row cosine 仍然约为 `0.976`。因此，`Gemma-3-1B` 不是“正常 Gemma-3 family 里的一个稍差点”，而是几何上明显处在另一种状态。

更合理的解释链是：

- `Gemma-3-1B` 是 Gemma 3 中唯一的 1B text-only 小模型，context 也只有 32K；4B/12B/27B 是 multimodal conditional generation 模型，且有 128K context。
- Gemma 3 技术报告说明所有模型都使用 distillation，post-training 又包含来自大 IT teacher 的 distillation 和 RL finetuning，用于提升 math、chat、reasoning、instruction-following、多语等能力。
- 对 1B 来说，embedding 参数占比很高：技术报告表中 1B 约有 302M embedding parameters 和 698M non-embedding parameters。小模型容量有限，post-training 若要塞入 chat / reasoning / multilingual 等能力，更可能显著移动 embedding 表本身。
- 这与当前观测一致：`E_instruct - E_base` 的 Frobenius 范数远大于同系列大模型，base/instruct 行向量方向几乎整体重排，最优 affine `R2` 也只有约 `0.3751`。

“1B 是特殊小容量分支”可以更具体地理解为：

- 架构路径不同：本地 `config.json` 显示 `Gemma-3-1B` / `Gemma-3-1B-Instruct` 使用 `Gemma3ForCausalLM`，`model_type = gemma3_text`；而 `Gemma-3-4B/12B/27B` 使用 `Gemma3ForConditionalGeneration`，外层 `model_type = gemma3`，带 `vision_config` 和 `text_config`。也就是说 1B 是 text-only CausalLM 分支，4B 以上是 VLM conditional generation 分支。
- 上下文长度不同：1B 的 `max_position_embeddings` / model card 口径是 32K；4B/12B/27B 是 128K。虽然这不直接决定 embedding 表，但说明 1B 的训练和架构目标与大模型不完全一致。
- 训练 token budget 不同：model card / technical report 中 1B 为 2T tokens，4B 为 4T，12B 为 12T，27B 为 14T。1B 的预训练语料预算更小，但 IT 阶段仍要对齐 chat、reasoning、instruction-following、多语等目标。
- embedding 参数占比异常高：1B 约 302M embedding parameters / 698M non-embedding parameters，embedding 约占总参数 30%；4B 约 675M / 3209M，约 17%；12B 约 1012M / 10759M，约 8.6%；27B 约 1416M / 25600M，约 5.2%。在 1B 中，embedding 表本身是模型容量的重要组成部分，因此后训练若需要改变模型行为，直接改写 embedding 表的“收益/代价比”可能更高。
- 教师蒸馏压力不同：Gemma 3 IT 使用来自更大 IT teacher 的 distillation 和 RL finetuning。对大模型来说，新增能力可以更多分散到深层 transformer 参数；对 1B 来说，非 embedding 容量有限，模型可能需要更大幅度地重排输入词向量，使后续层更容易表达 teacher 要求的行为。
- 本地几何结果正好符合这个图景：4B/12B/27B 的 base/instruct row cosine 仍在 `0.976-0.984`，说明 post-training 后 embedding 仍基本同向；1B 只有 `0.279`，说明不是同一坐标系里的小扰动，而是 embedding 表整体被重写。

外部资料和社区讨论提供的是间接支持，而不是直接定论：

- Google Developers Blog 对 Gemma 3 的说明明确写到，Gemma 3 的 pre-training / post-training 结合了 distillation、reinforcement learning 和 model merging；post-training 包括从更大 instruct model 蒸馏到 Gemma 3 预训练 checkpoint、RLHF、RLMF 和 RLEF。这支持“IT 版本不是轻微 SFT，而是较强 post-training”的判断。
- Gemma 3 model card 写到 1B 只使用 2T training tokens，context 为 32K；4B/12B/27B 使用更大 token budget，且 4B 以上还有 multimodal 架构。这支持“1B 是单独的小容量分支，不能直接按大模型规律外推”的判断。
- Hugging Face `google/gemma-3-1b-it` discussion 中有用户报告继续 instruction fine-tuning 后输出 gibberish；Google 组织成员回复时将原因指向 tokenizer / chat template、inference 参数、`fp16` vs `bf16` 数值稳定性和数据格式。这说明社区确实遇到 1B-IT 继续适配时的输出异常，但官方解释偏工程使用条件，不是 embedding 几何层面的因果。
- Hugging Face `google/gemma-3-1b-pt` discussion 中有用户报告 LoRA fine-tuned 1B full precision 正常，但 GPTQ/AWQ/BitsAndBytes 量化后出现 gibberish / blank output；Google 组织成员建议 QAT、支持 LoRA 的量化工具和部分层保持高精度。这说明 1B fine-tuned 后对量化 / 数值扰动可能更敏感。
- `google/gemma-3-1b-it` 还有关于 `chat_template.json` 和 `gemma3_text` config 的讨论，说明 1B-IT 在 serving/tooling 上确实有一些容易踩坑的格式与配置差异。不过这些问题影响生成和部署，不足以解释本地 full-vocab embedding row cosine 只有约 `0.279`。

因此当前判断应写成：

> `Gemma-3-1B` 的异常最像“小容量模型在强 distillation / post-training 下发生了整体 embedding 空间重写”。这和“预训练与后训练配比不均衡 / 相对过度后训练”的解释一致，但公开资料尚未直接确认这个因果。其他可能原因包括 1B text-only 架构分支、32K context 配置、chat template / EOS 行为差异、以及小模型为了吸收大 teacher 能力而对 embedding 层施加更强更新；不过 tokenizer id 错位和特殊 token 污染基本可以排除。

## A 矩阵与单位阵的关系

最初的直觉是：Base 到 Instruct 的变化可能接近“单位阵附近的小旋转 / 小扰动”。当前结果部分支持这个直觉，但需要更精确表述。

对 Qwen / Llama，`A` 通常非常接近单位方向：

- `identity_cosine` 接近 1。
- `rel_A_minus_I_over_I` 较小。
- `offdiag_norm_over_A` 较小，说明坐标混合有限。
- `R2` 很高，说明全局仿射关系足够解释大部分变化。

但 `A` 并不严格是正交旋转。`A^T A` 与 `I` 仍有可测误差，所以更准确的说法是：

> Base-Instruct 的 embedding 空间变化通常接近单位阵附近的低幅度全局仿射扰动，而不是严格的正交旋转。

## SVD 低秩分析

当前重点比较两类差异矩阵：

```text
E_delta = E_instruct - E_base
A_delta = A - I
```

对二者都做 SVD 能量分析，关注：

- `rank_50 / rank_80 / rank_90 / rank_95 / rank_99`
- `effective_rank`
- `energy_at_k`
- `energy_at_1pct_h / 5pct_h / 10pct_h`

这里的能量指 squared singular value energy。`rank_95` 表示解释 95% 能量所需的最小秩，`effective_rank` 是 entropy-based 有效秩。

## 秩与 hidden dimension 的关系

绝对 rank 明显随 hidden dimension 增大而增大，因此不能只看 rank 原值。更合理的比较方式是看：

```text
rank / hidden_dim
effective_rank / hidden_dim
energy@相对 hidden dimension
```

全体 31 组中：

```text
E_delta rank95 mean ~= 2613, 约 0.789 * hidden_dim
A-I     rank95 mean ~= 1493, 约 0.454 * hidden_dim

E_delta rank99 mean ~= 3171, 约 0.947 * hidden_dim
A-I     rank99 mean ~= 2187, 约 0.676 * hidden_dim

E_delta effective_rank mean ~= 1624, 约 0.443 * hidden_dim
A-I     effective_rank mean ~= 885, 约 0.231 * hidden_dim
```

排除 `Gemma-3-1B` 和 Gemma 4 后：

```text
E_delta rank95 / hidden_dim mean ~= 0.769
A-I     rank95 / hidden_dim mean ~= 0.426

E_delta rank99 / hidden_dim mean ~= 0.942
A-I     rank99 / hidden_dim mean ~= 0.656

E_delta effective_rank / hidden_dim mean ~= 0.418
A-I     effective_rank / hidden_dim mean ~= 0.203
```

结论：

> hidden dimension 决定绝对 rank 的规模；归一化后，`A-I` 的秩和有效秩系统性低于 `E_instruct - E_base`。直接 embedding 差异更高秩、更分散，而最优仿射变换相对单位阵的偏移集中在更低维的子空间。

## 归一化后的线性关系

把 rank 或 effective rank 除以 hidden dimension 后，`A-I` 与 `E_delta` 的关系不是一个全局稳定线性律，但在同一模型族内更接近线性。

主分析组中：

```text
(A-I effective_rank / h) ~= -0.012 + 0.515 * (E_delta effective_rank / h)
R2 ~= 0.553
```

同一模型族内，`effective_rank / h` 的线性关系更强：

```text
Qwen2.5: R2 ~= 0.915
Qwen3:   R2 ~= 0.710
Qwen3.5: R2 ~= 0.961
Llama:   R2 ~= 0.999
Gemma-2: R2 ~= 0.823
```

因此可以写成：

> 同一模型族内，`A-I` 的归一化有效秩与 `E_instruct - E_base` 的归一化有效秩有较明显线性关系；但跨模型族混合后，family 差异会显著扰动这个关系。

`rank95 / rank99` 不如 `effective_rank` 稳定，原因是高阈值 rank 容易接近 hidden dimension 上限，出现饱和效应。

## energy@相对 hidden dimension

为了避免固定 `k` 对不同 hidden dimension 不公平，当前脚本新增了：

```text
energy_at_1pct_h
energy_at_5pct_h
energy_at_10pct_h
```

含义是：top `1% / 5% / 10%` hidden dimensions 的奇异方向解释了多少能量。

排除 `Gemma-3-1B` 和 Gemma 4 后，主分析组中位数为：

```text
energy@1%h:
E_delta median ~= 0.144
A-I     median ~= 0.396

energy@5%h:
E_delta median ~= 0.397
A-I     median ~= 0.628

energy@10%h:
E_delta median ~= 0.492
A-I     median ~= 0.715
```

这个结果很关键：

> 在相同的相对维度预算下，`A-I` 的能量明显比 `E_delta` 更集中。比如只使用 top `5%` hidden dimensions，`A-I` 中位数已经解释约 `62.8%` 能量，而 `E_delta` 只有约 `39.7%`。

这比固定 `energy@100 / 500 / 1000` 更适合跨 hidden dimension 比较。

## 当前最适合写进结论的指标

建议优先级：

1. `energy_at_1pct_h / 5pct_h / 10pct_h`：最公平地比较不同 hidden dimension 下的谱集中程度。
2. `effective_rank / hidden_dim`：最适合概括为单个“有效维度”。
3. `rank80 / rank90 / rank95`：适合作为累计能量阈值的辅助证据。
4. `rank99`：保留，但不要作为主结论，因为容易接近满秩上限。
5. 固定 `energy@100 / 500 / 1000`：可用于补充，但不同 hidden dimension 间公平性较弱。

## 阶段性研究结论

当前最稳的研究表述是：

> 对大多数非异常 Base-Instruct pair，embedding 空间之间存在高质量全局仿射关系。该仿射矩阵 `A` 通常接近单位阵方向，但不是严格正交旋转。与直接 embedding 差异 `E_instruct - E_base` 相比，`A-I` 的谱能量更集中、有效秩更低，说明 instruct tuning 引入的全局空间变换可以被更少的主方向概括。

更短版本：

> Base-Instruct 的 embedding 差异本身偏高秩，但其最优全局仿射变换相对单位阵的偏移是更低秩、更集中的。

## 后续可补充

- 画 `energy threshold -> rank / hidden_dim` 的完整曲线，而不是只看 50/80/90/95/99。
- 画 `top p% hidden dim -> cumulative energy` 曲线，比较 `E_delta` 和 `A-I`。
- 对每个模型族分别拟合 `effective_rank / h` 的线性关系，避免跨 family 混合造成解释偏差。
- 对 Gemma 异常组单独写一节架构解释，避免污染主规律。
- 如需保存可视化，建议在 `results/task5_affine_relations/` 下新增图表或轻量 CSV，而把主观解释继续沉淀在本文档。
