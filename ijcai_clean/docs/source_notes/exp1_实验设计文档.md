# 实验1：全局几何相似性验证 (Global Geometry) - 实验设计文档

## 1. 实验概述

### 1.1 研究目标

本实验旨在通过计算不同模型嵌入空间的**全局几何相似性**，验证和比较不同语言模型的嵌入表示在几何结构上的相似程度。实验通过随机采样 token 对，计算其 Embedding (E) 和 Unembedding (U) 矩阵的距离矩阵相关系数（GCorr），从而量化模型间的几何相似性。

### 1.2 核心问题

- **指令微调的影响**：Base 模型与 Instruct 模型在嵌入几何上是否存在显著差异？
- **模型系列内部一致性**：同一系列不同规模的模型是否具有相似的嵌入几何结构？
- **跨系列架构差异**：不同架构（Qwen、Llama、Gemma等）在相同规模下的几何差异如何？
- **MoE vs Dense**：MoE 架构与 Dense 架构在嵌入几何上的差异特征是什么？

### 1.3 实验版本

- **v4 (当前版本)**：RTX 5090 极致优化版，双卡并行，动态 batch size
- **v3**：优化版，单卡为主
- **v2**：稳定版，基础功能验证
- **v1**：初始版，初步实验

---

## 2. 实验方法

### 2.1 核心算法：全局几何相关性 (GCorr)

**定义**：对于两个模型的嵌入矩阵 $E_A$ 和 $E_B$（或 $U_A$ 和 $U_B$），全局几何相关性定义为：

$$\text{GCorr}(E_A, E_B) = \text{Pearson}(\text{dist}_A, \text{dist}_B)$$

其中：
- $\text{dist}_A$ 和 $\text{dist}_B$ 分别是两个模型对相同 token 对计算的距离向量
- $\text{Pearson}$ 表示 Pearson 相关系数

### 2.2 距离度量

实验计算三种距离度量：

1. **余弦相似度 (Cosine Similarity)**
   $$\cos(x_i, x_j) = \frac{x_i \cdot x_j}{||x_i|| \cdot ||x_j||}$$

2. **欧氏距离 (Euclidean Distance)**
   $$d_{\text{euc}}(x_i, x_j) = ||x_i - x_j||_2 = \sqrt{\sum_{k}(x_{ik} - x_{jk})^2}$$

3. **平方欧氏距离 (Squared Euclidean Distance)**
   $$d_{\text{euc2}}(x_i, x_j) = ||x_i - x_j||_2^2 = \sum_{k}(x_{ik} - x_{jk})^2$$

### 2.3 计算流程

1. **Token 采样**：
   - 从词汇表中随机采样 $n_{\text{tokens}}$ 个有效 token（排除特殊 token）
   - 对于跨 tokenizer 对比，使用共同 token 集合

2. **距离矩阵采样**：
   - 从 $n_{\text{tokens}}$ 个 token 中随机采样 $n_{\text{pairs}}$ 个 token 对 $(i, j)$，其中 $i < j$
   - 使用拒绝采样策略在 GPU 上生成索引

3. **距离计算**：
   - 对每个 token 对，计算两个模型的距离值
   - 使用流式计算，避免构造完整距离矩阵

4. **相关性计算**：
   - 计算两个模型距离向量的 Pearson 相关系数
   - 使用流式统计量（sum, sum², sum(xy)）避免存储完整向量

5. **Bootstrap 重采样**：
   - 重复上述过程 $n_{\text{bootstrap}}$ 次，每次使用不同的随机种子
   - 计算均值、标准差、95% 置信区间等统计量

### 2.4 性能优化 (v4)

针对 RTX 5090 双卡环境的优化：

1. **纯 GPU 随机采样**：移除 CPU 索引计算，使用 GPU 拒绝采样生成 pair 索引
2. **动态 Batch Size**：根据 hidden_dim 自动调整 batch_size（50K-500K）
3. **TF32 加速**：启用 Tensor Core TF32 模式
4. **索引排序**：按 i_indices 排序，提升显存访问局部性（L2 Cache 命中率）
5. **双卡并行**：将 (模型对, bootstrap) 任务分配到双卡并行计算
6. **流式计算**：避免构造大张量，使用逐批累加统计量

---

## 3. 实验设计

### 3.1 实验 1.1：Base vs Instruct 配对对比

**目标**：验证指令微调对嵌入几何的影响

**设计**：比较同尺寸、同系列的 Base 与 Instruct 模型

**配对列表**（共 19 对）：

| 系列 | 配对数量 | 具体配对 |
|------|---------|----------|
| **Qwen3** | 5对 | 0.6B-Base↔0.6B, 1.7B-Base↔1.7B, 4B-Base↔4B, 8B-Base↔8B, 14B-Base↔14B |
| **Qwen2.5** | 7对 | 0.5B↔0.5B-Instruct, 1.5B↔1.5B-Instruct, 3B↔3B-Instruct, 7B↔7B-Instruct, 14B↔14B-Instruct, 32B↔32B-Instruct, 72B-Base↔72B-Instruct |
| **Llama** | 4对 | 3.2-1B↔3.2-1B-Instruct, 3.2-3B↔3.2-3B-Instruct, 3.1-8B↔3.1-8B-Instruct, 3.1-70B-Base↔3.1-70B-Instruct |
| **Gemma2** | 3对 | 2B↔2B-Instruct, 9B↔9B-Instruct, 27B↔27B-Instruct |

**预期结果**：Base 和 Instruct 模型应该具有较高的几何相似性（GCorr > 0.99），但可能存在细微差异。

---

### 3.2 实验 1.2：系列内部全模型互比

**目标**：验证同一模型系列内部不同规模模型的几何一致性

**设计**：在同一模型系列内部进行所有模型两两对比（排除实验 1.1 中已做的 Base-Instruct 配对）

**⚠️ 注意**：由于显存限制，Qwen2.5 和 Llama 系列按大小分组运行，组间交叉对比通过补充实验完成（见 3.5 节）。

**分组设置**：

| 系列 | 分组 | 模型列表 | 对比组数 |
|------|------|----------|---------|
| **Qwen3** | 完整 | 0.6B-Base, 0.6B, 1.7B-Base, 1.7B, 4B-Base, 4B, 8B-Base, 8B, 14B-Base, 14B, 32B (11模型) | 55组 |
| **Qwen2.5-small** | 分组 | 0.5B, 0.5B-Instruct, 1.5B, 1.5B-Instruct, 3B, 3B-Instruct, 7B, 7B-Instruct, 14B, 14B-Instruct (10模型) | 45组（排除5对Base-Instruct） |
| **Qwen2.5-large** | 分组 | 32B, 32B-Instruct, 72B-Base, 72B-Instruct (4模型) | 6组（排除1对Base-Instruct） |
| **Llama-small** | 分组 | 3.2-1B, 3.2-1B-Instruct, 3.2-3B, 3.2-3B-Instruct, 3.1-8B, 3.1-8B-Instruct (6模型) | 15组（排除3对Base-Instruct） |
| **Llama-large** | 分组 | 3.1-70B-Base, 3.1-70B-Instruct (2模型) | 1组（排除1对Base-Instruct） |
| **Gemma2** | 完整 | 2B, 2B-Instruct, 9B, 9B-Instruct, 27B, 27B-Instruct (6模型) | 15组（排除3对Base-Instruct） |

**组间交叉对比**（通过补充实验完成，见 3.5 节）：
- ✅ Qwen2.5: 0.5B~14B ↔ 32B~72B 的交叉对比（40组）
- ✅ Llama: 1B~8B ↔ 70B 的交叉对比（12组）

**预期结果**：同一系列内部模型应该具有较高的几何相似性，且相似性可能随规模差异增大而降低。

---

### 3.3 实验 1.3：跨系列同规模对比

**目标**：探索不同架构在相同规模下的几何差异

**设计**：在相近参数量的模型之间进行跨系列对比

**规模分组**（共 7 个规模组）：

| 规模组 | 模型列表 | 对比组数 |
|--------|----------|---------|
| **~0.5-0.6B** | Qwen3-0.6B-Base, Qwen3-0.6B, Qwen2.5-0.5B, Qwen2.5-0.5B-Instruct | 6组 |
| **~1-2B** | Qwen3-1.7B-Base, Qwen3-1.7B, Qwen2.5-1.5B, Qwen2.5-1.5B-Instruct, Llama-3.2-1B, Llama-3.2-1B-Instruct, Gemma-2-2B, Gemma-2-2B-Instruct | 28组 |
| **~3-4B** | Qwen3-4B-Base, Qwen3-4B, Qwen2.5-3B, Qwen2.5-3B-Instruct, Llama-3.2-3B, Llama-3.2-3B-Instruct | 15组 |
| **~7-9B** | Qwen3-8B-Base, Qwen3-8B, Qwen2.5-7B, Qwen2.5-7B-Instruct, Llama-3.1-8B, Llama-3.1-8B-Instruct, Mistral-7B-v0.3, Yi-1.5-9B, Gemma-2-9B, Gemma-2-9B-Instruct | 45组 |
| **~14B** | Qwen3-14B-Base, Qwen3-14B, Qwen2.5-14B, Qwen2.5-14B-Instruct | 6组 |
| **~27-32B** | Qwen3-32B, Qwen2.5-32B, Qwen2.5-32B-Instruct, Gemma-2-27B, Gemma-2-27B-Instruct | 10组 |
| **~70-72B** | Llama-3.1-70B-Base, Llama-3.1-70B-Instruct, Qwen2.5-72B-Base, Qwen2.5-72B-Instruct | 6组 |

**特殊处理**：跨 tokenizer 对比时，使用共同 token 集合进行采样。

**预期结果**：不同架构在相同规模下可能存在较大的几何差异，但某些架构可能具有相似的几何结构。

---

### 3.4 实验 1.4：MoE 模型对比

**目标**：比较 MoE (Mixture of Experts) 架构与 Dense 架构模型的嵌入几何差异

**模型列表**：

| 类型 | 模型列表 |
|------|----------|
| **MoE 模型** | Qwen3-30B-A3B, DeepSeek-V2-Lite-Chat |
| **Dense 模型** | Qwen3-8B, Qwen3-14B, Qwen2.5-7B, Qwen2.5-14B |

**对比组数**：2 × 4 + 1 = 9 组（MoE-MoE + MoE-Dense）

**预期结果**：MoE 模型与 Dense 模型在嵌入几何上可能存在显著差异，反映架构对表示空间的影响。

---

### 3.5 补充实验：跨大小分组对比 ✅ 已完成

**目标**：补充实验 1.2 中缺失的 small ↔ large 组间对比，完善实验的完整性和权威性

**设计**：由于显存限制，采用每次只加载 2 个模型的策略，避免内存爆炸

**对比组**：
- **Qwen2.5**: small(10) × large(4) = 40 组
  - small组：0.5B, 0.5B-Instruct, 1.5B, 1.5B-Instruct, 3B, 3B-Instruct, 7B, 7B-Instruct, 14B, 14B-Instruct
  - large组：32B, 32B-Instruct, 72B-Base, 72B-Instruct
- **Llama**: small(6) × large(2) = 12 组
  - small组：3.2-1B, 3.2-1B-Instruct, 3.2-3B, 3.2-3B-Instruct, 3.1-8B, 3.1-8B-Instruct
  - large组：3.1-70B-Base, 3.1-70B-Instruct

**总计**：52 组对比 ✅ **已完成**

**技术特点**：
- 内存优化：每次只加载 2 个模型，计算完立即释放
- 双卡并行：Bootstrap 任务分配到双卡并行计算
- 断点续传：支持从断点恢复，避免重复计算
- 实时保存：每完成一组立即保存，防止数据丢失

**完成时间**：2026-01-29

**结果文件**：`results/exp1_cross_size_20260129_015053.csv`

---

## 4. 实验参数配置

### 4.1 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_tokens` | 20,000 | 采样 token 数量 |
| `n_bootstrap` | 100 | Bootstrap 重采样次数 |
| `n_pairs` | 5,000,000 | 距离矩阵采样对数 |
| `random_seed` | 42 | 随机种子 |

### 4.2 参数选择理由

- **n_tokens = 20,000**：平衡计算效率和统计代表性，覆盖大部分词汇表
- **n_bootstrap = 100**：提供足够的统计稳定性，计算 95% 置信区间
- **n_pairs = 5,000,000**：对于 20K tokens，理论上最多有 200M 对，采样 5M 对（2.5%）已足够
- **random_seed = 42**：确保结果可复现

---

## 5. 计算指标

### 5.1 主要指标

对于每个模型对，计算以下指标：

1. **GCorr_E_cos**: Embedding 矩阵余弦相似度的相关系数
2. **GCorr_E_euc**: Embedding 矩阵欧氏距离的相关系数
3. **GCorr_E_euc2**: Embedding 矩阵平方欧氏距离的相关系数
4. **GCorr_U_cos**: Unembedding 矩阵余弦相似度的相关系数
5. **GCorr_U_euc**: Unembedding 矩阵欧氏距离的相关系数
6. **GCorr_U_euc2**: Unembedding 矩阵平方欧氏距离的相关系数

### 5.2 统计量

每个指标计算以下统计量：

- **mean**: Bootstrap 均值
- **std**: Bootstrap 标准差
- **se**: 标准误差 (std / sqrt(n_bootstrap))
- **ci95_low**: 95% 置信区间下界（2.5% 分位数）
- **ci95_high**: 95% 置信区间上界（97.5% 分位数）
- **median**: Bootstrap 中位数

### 5.3 元数据字段

每个结果还包含以下元数据：

- **模型信息**: model_a, model_b, family_a, family_b
- **关系信息**: same_family, same_tokenizer
- **模型配置**: hidden_dim_a, hidden_dim_b, vocab_size_a, vocab_size_b
- **权重绑定**: is_tied_a, is_tied_b, actual_tied_a, actual_tied_b
- **实验参数**: n_tokens, n_bootstrap, n_pairs

---

## 6. 技术实现细节

### 6.1 模型加载

- **优先使用 safetensors**：直接从 safetensors 文件加载权重，避免完整模型加载
- **回退机制**：如果 safetensors 失败，回退到 transformers 库加载
- **权重检测**：自动检测 Embedding (E) 和 Unembedding (U) 权重
- **权重绑定检测**：检查 E 和 U 是否共享权重（tied embeddings）

### 6.2 Token 采样策略

- **有效 Token 识别**：排除特殊 token（BOS, EOS, PAD, UNK 等）
- **共同 Token 匹配**：跨 tokenizer 对比时，通过 token 字符串匹配找到共同 token
- **随机采样**：使用 numpy.random.default_rng 确保可复现性

### 6.3 GPU 计算优化

- **流式计算**：避免构造完整距离矩阵，使用分批计算
- **拒绝采样**：在 GPU 上生成随机索引对，过滤 $i < j$ 条件
- **索引排序**：按 i_indices 排序，提升显存访问局部性
- **动态 Batch Size**：根据 hidden_dim 自动调整，充分利用显存
- **TF32 加速**：启用 Tensor Core TF32 模式（RTX 30xx/40xx/50xx）

### 6.4 并行计算

- **双卡并行**：将 (模型对, bootstrap) 任务分配到双卡
- **线程锁**：使用 GPU_LOCKS 保护 GPU 访问，避免显存碎片化
- **任务分配**：轮询分配任务到不同 GPU，负载均衡

---

## 7. 数据输出格式

### 7.1 CSV 文件格式

结果保存为 CSV 文件，包含以下列：

```
model_a, model_b, family_a, family_b, same_family, same_tokenizer,
n_tokens, n_bootstrap, n_pairs,
hidden_dim_a, hidden_dim_b, vocab_size_a, vocab_size_b,
is_tied_a, is_tied_b, actual_tied_a, actual_tied_b,
gcorr_E_cos_mean, gcorr_E_cos_std, gcorr_E_cos_se, gcorr_E_cos_ci95_low, gcorr_E_cos_ci95_high,
gcorr_U_cos_mean, gcorr_U_cos_std, gcorr_U_cos_se, gcorr_U_cos_ci95_low, gcorr_U_cos_ci95_high,
gcorr_E_euc_mean, gcorr_E_euc_std, gcorr_E_euc_se, gcorr_E_euc_ci95_low, gcorr_E_euc_ci95_high,
gcorr_U_euc_mean, gcorr_U_euc_std, gcorr_U_euc_se, gcorr_U_euc_ci95_low, gcorr_U_euc_ci95_high,
gcorr_E_euc2_mean, gcorr_E_euc2_std, gcorr_E_euc2_se, gcorr_E_euc2_ci95_low, gcorr_E_euc2_ci95_high,
gcorr_U_euc2_mean, gcorr_U_euc2_std, gcorr_U_euc2_se, gcorr_U_euc2_ci95_low, gcorr_U_euc2_ci95_high
```

### 7.2 元数据 JSON 文件

每次运行还会生成元数据 JSON 文件，包含：

```json
{
  "timestamp": "20260127_051454",
  "version": "v4",
  "config": {
    "n_tokens": 20000,
    "n_bootstrap": 100,
    "n_pairs": 5000000,
    "random_seed": 42
  },
  "optimizations": [
    "纯 GPU 随机采样（拒绝采样策略）",
    "torch.compile JIT 编译",
    "TF32 Tensor Core 加速",
    "大 Batch Size (1M)",
    "移除频繁 empty_cache 调用",
    "Index Sorting 显存访问局部性优化"
  ],
  "total_comparisons": 245,
  "torch_compile_enabled": false
}
```

---

## 8. 运行方式

### 8.1 基本命令

```bash
# 运行全部实验
python run_exp1_v4.py --mode all

# 仅运行 Base vs Instruct 对比
python run_exp1_v4.py --mode base_instruct

# 运行特定系列内部对比
python run_exp1_v4.py --mode intra_family --family Qwen3

# 运行特定规模跨系列对比
python run_exp1_v4.py --mode cross_size --size "~7-9B"

# 运行 MoE 模型对比
python run_exp1_v4.py --mode moe
```

### 8.2 参数调整

```bash
# 自定义参数
python run_exp1_v4.py \
  --mode all \
  --n_tokens 20000 \
  --n_bootstrap 100 \
  --n_pairs 5000000 \
  --seed 42
```

### 8.3 补充实验

```bash
# 运行跨大小分组对比
python run_cross_size_groups.py
```

---

## 9. 实验结果统计

### 9.1 v4 版本统计

- **总对比组数**: 245 组（主实验）+ 52 组（补充实验）= **297 组**
- **总耗时**: ~6.5 小时（主实验）+ ~2 小时（补充实验）= **~8.5 小时**
- **模型覆盖**: ~45 个模型
- **模型系列**: Qwen3, Qwen2.5, Llama, Gemma2, Mistral, Yi, DeepSeek

### 9.2 实验分组统计

| 实验 | 对比组数 | 预计耗时 |
|------|---------|---------|
| 1.1 Base vs Instruct | 19 | ~34分钟 |
| 1.2 Qwen3 内部 | 50 | ~66分钟 |
| 1.2 Qwen2.5-small | 40 | ~57分钟 |
| 1.2 Qwen2.5-large | 4 | ~13分钟 |
| 1.2 Llama-small | 12 | ~16分钟 |
| 1.2 Gemma2 | 12 | ~19分钟 |
| 1.3 跨系列同规模 | 116 | ~2.8小时 |
| 1.4 MoE对比 | 15 | ~26分钟 |
| 补充：跨大小分组 ✅ | 52 | ~2小时 |

---

## 10. 数据文件位置

### 10.1 主要结果文件

- **v4 合并结果**: `results/exp1_global_v4_merged.csv` ⭐ **主实验245组**
- **v4 补充结果**: `results/exp1_cross_size_20260129_015053.csv` ⭐ **补充实验52组**
- **元数据文件**: `results/metadata_v4_*.json`

**注意**：完整数据集需要合并主实验结果和补充实验结果，总计 297 组对比。

**数据合并方法**：

**方法1：使用合并脚本（推荐）**
```bash
# 运行合并脚本
python merge_results.py

# 输出文件: results/exp1_global_v4_complete.csv
```

**方法2：手动合并（Python）**
```python
import pandas as pd

# 加载主实验结果
main_df = pd.read_csv('results/exp1_global_v4_merged.csv')

# 加载补充实验结果
supplement_df = pd.read_csv('results/exp1_cross_size_20260129_015053.csv')

# 合并（自动去重）
full_df = pd.concat([main_df, supplement_df], ignore_index=True)
full_df = full_df.drop_duplicates(subset=['model_a', 'model_b'], keep='first')

# 保存完整结果
full_df.to_csv('results/exp1_global_v4_complete.csv', index=False)
print(f"完整数据集: {len(full_df)} 组对比")
```

### 10.2 分批结果文件

实验按组分别运行，每个组生成独立的结果文件：

- `exp1_global_v4_20260126_233713.csv` - Base vs Instruct
- `exp1_global_v4_20260127_004340.csv` - Qwen3 内部
- `exp1_global_v4_20260127_014018.csv` - Qwen2.5-small
- `exp1_global_v4_20260127_015327.csv` - Qwen2.5-large
- `exp1_global_v4_20260127_020954.csv` - Llama-small
- `exp1_global_v4_20260127_022824.csv` - Gemma2
- `exp1_global_v4_20260127_051454.csv` - 跨系列同规模
- `exp1_global_v4_20260127_054008.csv` - MoE对比

---

## 11. 后续分析方向

1. **可视化分析**：
   - 绘制 GCorr 热力图，展示不同模型对之间的相似性
   - 按模型系列、规模、架构类型分组可视化
   - 分析 Base vs Instruct 的差异模式

2. **统计分析**：
   - 计算不同实验组的平均 GCorr 值
   - 分析规模、架构对几何相似性的影响
   - 比较 E 和 U 矩阵的相似性差异

3. **深入探索**：
   - 分析高相似性和低相似性模型对的特征
   - 探索几何相似性与模型性能的关系
   - 研究 MoE 架构的特殊性

---

## 12. 参考文献与相关概念

- **嵌入几何**：研究语言模型嵌入空间的几何结构
- **全局相似性**：通过距离矩阵相关性量化模型间的几何相似性
- **Bootstrap 重采样**：通过重复采样评估统计量的稳定性
- **Pearson 相关系数**：衡量两个变量线性相关程度的指标

---

**文档版本**: v1.1  
**最后更新**: 2026-01-29  
**实验版本**: v4 (RTX 5090 极致优化版) + 补充实验（跨大小分组对比）✅
