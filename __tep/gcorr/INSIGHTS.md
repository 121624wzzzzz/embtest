# GCorr：BI-clean 30 与当前 Task1–4

## 数据范围

| Task | 结果行 | 唯一模型 |
|---|---:|---:|
| Task1 Base→Instruct | 35 | 70 |
| Task2 series | 110 | 60 |
| Task3 cross-scale | 176 | 34 |
| Task4 MoE cross-family | 21 | 7 |

四个任务合计342条结果、340个唯一 pair。BI 主结论只使用 Task1 中排除5个异常后的30对。

## BI-clean

- E cosine GCorr mean：0.9955。
- E Euclidean GCorr mean：0.9902。
- Task6 E affine R2 mean：0.9923。
- 异常5对继续保留在 `cross_model_geometry` 原始结果中，但不混入本目录主聚合。

## 跨模型主要发现

- Task2 同 hidden dimension 的 E cosine mean 为0.8731，异 hidden 为0.3759，hidden mismatch 是主要混杂因素。
- Task3 E cosine mean 为0.4808，并有23对负 Euclidean GCorr，BI 规律不能直接外推到跨 family。
- tied/untied 和 E/U 必须分层报告；不要用单一全体均值替代结构性分组。

## 数据入口

- `tables/gcorr_task1_base_instruct_metrics.csv`：BI-clean 30。
- `tables/gcorr_task2_series_aggregate.csv`：当前 Task2 聚合。
- `tables/gcorr_task3_scale_group_aggregate.csv`：当前 Task3 聚合。
- `../../data/key_metrics.csv`：BI-clean Task1+Task6 联表。
