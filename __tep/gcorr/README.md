# 论文 A — GCorr

> **洞察摘要**：[`INSIGHTS.md`](INSIGHTS.md)（tex 叙事 + 数据数字，优先读）  
> LaTeX：[`main_zh_neurips_full.tex`](main_zh_neurips_full.tex) · 仿射论文：[`../affine/`](../affine/)

## 五条结论

1. BI 主组（n=26）GCorr cos **0.995**
2. Task2 异 hidden cos **0.376** vs 同 hidden **0.873**
3. Task3 全体 cos ~**0.48**；**23** 对负 euc
4. TU 37 对：**37/37** $U_{\cos}>E_{\cos}$
5. Gemma 异常须分报 → [`analysis/gemma_anomalies.md`](analysis/gemma_anomalies.md)

## 文件

| 路径 | 用途 |
|------|------|
| `INSIGHTS.md` | tex↔数据抽象洞察、改稿清单 |
| `analysis/*_gcorr.md` | Task1–4 分任务细节 |
| `tables/gcorr_task*.csv` | 聚合表 |
| `main_zh_neurips_full.tex` | NeurIPS 稿（正文未改） |

源 CSV：`ijcai_clean/results/task{1,2,3,4}_*/summary*.csv`
