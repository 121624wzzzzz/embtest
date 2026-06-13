# 论文 A 洞察摘要（GCorr / AGD 诊断）

> 从 [`main_zh_neurips_full.tex`](main_zh_neurips_full.tex) 抽象叙事，数字对齐 `ijcai_clean/results/task{1-4}_*/`。当前全仓口径见 [`../../docs/分析口径与特殊案例.md`](../../docs/分析口径与特殊案例.md)。  
> 仿射/SVD 见 [`../affine/INSIGHTS.md`](../affine/INSIGHTS.md)。

---

## 1. 论文在说什么（与数据无关的动机）

- **问题**：小模型多用 tied（E=U 共享），大模型/MoE 多用 untied；tied 的表征代价缺乏系统解释。
- **方法链**：大规模 **GCorr 几何审计** → 发现 E/U **非对称** → 动机化 **AGD**（输入侧轻量解绑）→ MiniMind / GPT-2 **预训练消融**（§4，**本仓库无 pretrain CSV**）。
- **GCorr 定义**：对 token 对采样，比较两模型上 cos / euc 距离向量的 **Pearson 相关**（非逐向量相等）。

---

## 2. 数据规模（Task1–4）

| 任务 | 内容 | pair 数 |
|------|------|--------|
| Task1 | Base→Instruct | 35 |
| Task2 | 系列内 | 110 |
| Task3 | 跨系列×规模 | 176 |
| Task4 | MoE 跨族 | 21 |
| **合计** | GCorr | **342** |

- **94** 个唯一模型；规模 0.5B–397B（config 口径）。
- **tied 口径**：CSV 列 `actual_tied_*`（权重实测，非 config）。

---

## 3. 三条核心洞见（tex §3.2 ↔ 数据）

### 洞见 1：tied 共享矩阵更像 **U**，不像 **E**

**叙事**：tied 模型 vs 同系列 untied 时，共享矩阵几何上贴近 untied 的 **输出反嵌入**。

| 指标 | Task2 tied↔untied（n=37） | 来源 |
|------|---------------------------|------|
| $U_{\cos}$ / $U_{\euc}$ | **0.8114** / **0.8169** | task2 summary |
| $E_{\cos}$ / $E_{\euc}$ | **0.1395** / **0.5553** | 同上 |
| 逐对 $U>E$（cos 且 euc） | **37/37** | 同上 |

**Base→Instruct 补充**：
- **Untied 9/9**：$E_{\euc}\ge 0.999$（输入欧氏几何几乎不变）。
- **Tied 22 对**：$E_{\euc}$ 均值 **0.8925** — ⚠️ 被 **Gemma-3-1B + Gemma-4 五对** 拉低；排除后 17 对均值 **≈0.983**。

**机制解释（tex + 预训练图）**：输出侧全词表稠密梯度 vs 输入侧稀疏梯度；tied 时两路梯度合并到同一 $W$。

---

### 洞见 2：即便 untied，**U 比 E 更相似**

| 子集 | n | $U_{\cos}$ | $E_{\cos}$ | 来源 |
|------|---|------------|------------|------|
| 同系列 untied↔untied | 48 | **0.8907** | 0.6095 | Task2 |
| 跨规模 untied↔untied（>4B 桶） | 26 | **0.6690** | 0.3095 | Task3 |
| $U_{\cos}>E_{\cos}$ 逐对 | 26 组 | **13/13**（每桶） | — | Task3 |

**边界**：Task2 同 hidden cos **0.873** vs 异 hidden **0.376** — 跨模型结论需同 hidden 或接受混杂。

---

### 洞见 3：输入侧 **euc > cos** → 偏置/尺度路径

| 子集 | $E_{\euc}$ | $E_{\cos}$ | 备注 |
|------|------------|------------|------|
| Task2 UU 48 对 | **0.9446** | 0.6095 | **48/48** $E_{\euc}>E_{\cos}$ |
| Task3 UU 26 对（组均值） | 0.6103 | 0.3095 | 17/26 逐对 euc>cos |

**叙事**：低 cos ≠ 语义崩溃；更可能是全局偏置/尺度使 cosine 退化，euc 仍保留 token–token 结构 → 动机化 AGD **输入侧偏置校正**。

**与仿射论文的桥**：Task6 主组 full-vocab $E\_R^2\approx 0.991$ 支持「全局仿射偏移」叙事（见 affine INSIGHTS）。

---

## 4. Base→Instruct 主规律（Task1，tex 部分提及）

| 分组 | n | GCorr cos 均值 | 备注 |
|------|---|----------------|------|
| 主分析组 | 26 | **0.995** | 排除 Gemma-3-1B + Gemma-4 |
| 全体 | 35 | 0.970 | |
| 异常 | 5 | 0.835 | 须分报 |

详见 [`analysis/gemma_anomalies.md`](analysis/gemma_anomalies.md)。

---

## 5. 边界与负例（tex 宜写清）

| 现象 | 数据 | 含义 |
|------|------|------|
| Task3 全体 cos 均值 | ~**0.48** | 跨系列不能外推 BI 结论 |
| Task3 负 $E_{\euc}$ | **23** 对 | 跨族几何可「反向」 |
| Task2 异 hidden | cos **0.376** | hidden 不匹配主导低 GCorr |
| Gemma-3-1B | cos **0.777** | 唯一 <0.80 |
| Gemma-4 | cos 尚可、euc/R² 崩 | cos 与 euc 须并排 |

---

## 6. tex 与数据不一致（改稿清单）

| 优先级 | tex 现状 | 应用数据 |
|--------|----------|----------|
| **P0** | 仿射 **189** 组 | 当前 Task5 `summary_pair.csv` 为 **340** 行；GCorr Task1-4 当前为 **342** 对，二者需分开报口径 |
| **P1** | tied BI $E_{\euc}=0.8925$ 作普遍退化 | 并列主组 **0.983** + Gemma 五对分报 |
| **P1** | 无 Gemma 异常段 | 附录：5 对 BI 异常 |
| **P0** | §4 预训练表全部数值 | 仓库**无** pretrain CSV，暂不可复核 |
| **P1** | 未写 actual_tied 口径 | 方法节补一句 |

---

## 7. 数据有、tex 未写（可进附录或 B 文）

- Task6：BI full-vocab $E\_R^2$、$A-I$ 低秩 / energy@5%h（→ affine 模块）
- GCorr cos ↔ $E\_R^2$，$r\approx 0.93$（仅 BI）
- Task5 全体 $R^2_E\approx 0.43$ vs BI **0.94**（防误读仿射）

---

## 8. 阅读顺序

1. 本文件（洞察）  
2. [`analysis/base_instruct_gcorr.md`](analysis/base_instruct_gcorr.md) — Task1 细节  
3. [`analysis/gemma_anomalies.md`](analysis/gemma_anomalies.md) — 异常  
4. [`main_zh_neurips_full.tex`](main_zh_neurips_full.tex) — 正文（未改正文）
