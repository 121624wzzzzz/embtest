# docs 文档索引

本目录只保留当前主线文档和少量仍有用的审计说明。历史口径、cleanup 记录和旧实验摘要已移到 `historical/`。

## 当前主线

- `methods_and_metrics.md`：当前 Task1-5、Base-Instruct full-vocab 仿射 / `A-I` / SVD 诊断的方法、实验设置、输出产物和指标口径。新增实验方法或指标时优先更新这里。
- `current_state.md`：当前仓库结构、配置、入口脚本、结果目录和运行环境速查。
- `model_tag_audit.md`：tied / untied 标签审计口径，说明 `is_tied` 和 `actual_tied` 的区别；写分组结果或论文表格时可参考。

## 研究结论

- `../analysis.md`：项目阶段性研究判断和结果解读，包括 Base-Instruct full-vocab 仿射结论、Gemma 异常解释、`A-I` 与 `E_instruct - E_base` 的 SVD 低秩分析。

## 历史归档

- `historical/results_summary.md`：旧 exp1/exp2 结果摘要，基于已归档的 V4 入口，不代表当前 Task1-5 主线。
- `historical/legacy_results_summary/`：被 Task1-5 替代的旧轻量汇总 CSV（`exp1_global_summary.csv`、`exp1_model_tag_audit.csv`、`affine_*_summary.csv`、`untied_comparison_summary.csv`）。
- `historical/cleanup_inventory.md`：2026-05-10 cleanup 清单，记录新旧文件关系和归档决策。
- 仓库根 `archive/ijcai_cleanup_2026-05-10/`：归档的旧 V4 实验代码与结果快照。
