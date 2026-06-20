# __tep：BI-clean 30 论文分析区

`__tep/` 统一使用 **BI-clean 30**：`configs/base_instruct_pairs.yaml` 注册的 35 对 Base→Instruct pair，排除 `Gemma-3-1B` 与四对 `Gemma-4` 异常后得到 30 对（17 tied + 13 untied）。本目录不再维护更早的小规模 BI 口径。

## 数据关系

- 原始实验结果：`../ijcai_clean/results/`
- 标准化矩阵：`../extracts/`
- 当前全局统计：`data/computed_stats.json`、`data/key_metrics.csv`
- GCorr 派生分析：`gcorr/`
- Affine / low-rank 派生分析：`affine/`

## 当前核心结论

| 结论 | BI-clean 30 结果 |
|---|---:|
| Task1 E-cos GCorr mean | 0.9955 |
| Task6 E affine R2 mean | 0.9923 |
| E 侧 P/D median | 0.1080 |
| U 侧 P/D median | 0.3098 |
| untied E / U P/D median | 0.0491 / 0.3188 |
| W-rank=1 时 affine 胜出 | E 16/30；U 27/30 |

Hybrid 的旧结果没有覆盖全部30对，已从当前数据和结论中移除。脚本仍保留，待 GPU 空闲时必须使用 `--all-clean` 对30对完整重跑后才能恢复该分析。

## 复核

```bash
python3 __tep/scripts/verify_metrics.py
```
