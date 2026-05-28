# Aff/LoRA 同解释量实验（non-excluded **30 对**）

**口径**：Task6 residual-aware 分解 $D=Y_c-X_c$，$P=X_c(A-I)$，$R=D-P$。  
在**同 centered 解释量**下比较 **Aff/LoRA**（hidden-space，$A-I$）与 **W-form / Vocab LoRA**（$n\times d$ 词表矩阵）；W rank=$k$ 时 aff rank 取参数量匹配 $\lfloor k(n+h)/(2h)\rfloor$。

详述：[`notes/02_affine_effective_update_insight.md`](../notes/02_affine_effective_update_insight.md) §11。  
**数据目录**：[`README.md`](README.md)。

| 文件 | 规模 |
|------|------|
| [`affine_lora_budget_summary.csv`](affine_lora_budget_summary.csv) | **30 行**（26 main + 4 extended） |
| [`affine_lora_by_tied_summary.csv`](affine_lora_by_tied_summary.csv) | tied / untied 汇总 |
| [`w_rank_budget_extended_{e,u}.csv`](w_rank_budget_extended_e.csv) | extended 4 对 sweep 中间表 |

**重建**：`python3 bi_analysis/scripts/build_affine_lora_budget_30.py`

---

## 参数量与秩换算

[
P_W = h + r_W(n+h), \qquad P_{\mathrm{aff}} = h + 2h\,r_{\mathrm{aff}}
\qquad\Rightarrow\qquad
r_{\mathrm{aff}} \approx r_W \cdot \frac{n}{2d}
]

`U_aff_rank_budget_r1` = W rank=1 时参数量匹配的 aff rank（与 $(n+d)/2d$ 一致）。

---

## 分组汇总（U 侧，30 对）

| 分组 | n | aff/W @r1 中位 | wins @r1 | aff@W1 中位 | (n+d)/2d 中位 |
|------|---:|---:|---:|---:|---:|
| **untied U** | **13** | **2.18×** | **12/13** | **15** | **15.3** |
| untied E | 13 | 0.34× | 1/13 | 15 | 15.3 |
| tied（E=U） | 17 | 3.00× | 15/17 | 37 | 37.6 |

untied U 唯一 r1 例外：**DeepSeek-V3**（aff/W≈0.75；P/D 仍高于 E）。  
extended 4 对 U：Qwen3-30B / Qwen3.5-35B aff/W **>5×**；DeepSeek-V3.1 **1.15×**。

---

## untied 13 对（U 侧，按 n/d 降序）

| 模型 | tier | n/d | aff@W1 | aff/W@r1 | P/D |
|------|:---:|---:|---:|---:|---:|
| Qwen3.5-35B-A3B-Base | extended | 121.2 | 61 | 5.25 | 0.207 |
| Qwen3.5-9B-Base | main | 60.6 | 30 | 5.04 | 0.210 |
| Qwen3-30B-A3B-Base | extended | 74.2 | 37 | 5.17 | 0.311 |
| Qwen2.5-7B | main | 42.4 | 21 | 1.16 | 0.462 |
| Qwen3-8B-Base | main | 37.1 | 19 | 4.45 | 0.308 |
| Llama-3.1-8B | main | 31.3 | 16 | 2.82 | 0.411 |
| Qwen2.5-14B | main | 29.7 | 15 | 1.18 | 0.486 |
| Qwen2.5-32B | main | 29.7 | 15 | 1.09 | 0.515 |
| Qwen3-14B-Base | main | 29.7 | 15 | 4.30 | 0.319 |
| Qwen2.5-72B-Base | main | 18.6 | 9 | 1.08 | 0.481 |
| DeepSeek-V3-Base | extended | 18.0 | 9 | 0.75 | 0.122 |
| DeepSeek-V3.1-Base | extended | 18.0 | 9 | 1.15 | 0.112 |
| Llama-3.1-70B-Base | main | 15.7 | 8 | 2.18 | 0.478 |

→ Vocab LoRA **rank=1** 时，Aff/LoRA 参匹配秩通常 **~8–61**（untied U 中位 **~15**）；大词表 MoE 可达 **~60+**。

---

## Hybrid 定位（补充）

Hybrid（affine P + W-form R）对 U 侧主叙事增量有限；详见 insight §11.3。30 对 hybrid 判据见 CSV 列 `*_hybrid_stable_small_budget_both`（extended 4 对未单独跑 hybrid sweep，由 W-rank 四档 centered gain 推断）。

---

## 关键列（`affine_lora_budget_summary.csv`）

| 列 | 含义 |
|----|------|
| `analysis_tier` | `main` / `extended` |
| `*_P_over_D_full_affine_gain` | $P/D$ |
| `*_aff_vs_W_ratio_r{k}` | aff gain / W gain @ W rank $k$ |
| `*_aff_rank_budget_r{k}` | 参数量匹配的 aff rank @ W rank $k$ |

E/U 对 untied 分列；tied 模型两列相同。
