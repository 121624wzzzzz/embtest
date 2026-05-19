# docs 文档索引

当前 `ijcai_clean` 文档只保留一个方法学权威文档和少量历史结果归档。项目入口、运行命令和目录结构见上一级 `README.md`；研究结论见 `../analysis.md`。

## 当前主线

- `methods_and_metrics.md`：Task1-6 的方法、实验设置、输出产物、指标定义、token 对齐、GCorr、仿射 / SVD、tied / untied 口径。新增实验方法或指标时优先更新这里。
- `../analysis.md`：阶段性研究判断和结果解读，包括 Base-Instruct full-vocab 仿射结论、Gemma 异常解释、`A-I` 与 `E_instruct - E_base` 的 SVD 低秩分析。

## 历史归档

- `historical/results_summary.md`：迁移到当前 Task1-6 / `extracts/` 主线前的旧 exp1/exp2 摘要，仅用于追溯。
- `historical/legacy_results_summary/`：旧轻量汇总 CSV（`exp1_global_summary.csv`、`exp1_model_tag_audit.csv`、`affine_*_summary.csv`、`untied_comparison_summary.csv`）。
- 仓库根 `archive/ijcai_cleanup_2026-05-10/`：归档的旧 V4 实验代码、source notes 与结果快照。
