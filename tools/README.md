# Tools

本目录只放当前维护的工具入口，避免再按单脚本拆多层目录。

## 脚本

- `get_model_useful.py`：读取 `configs/models.yaml`，从 ModelScope 下载模型所需分片，抽取 embedding / lm_head，并备份到 `extracts/`。
- `audit.py`：统一审计入口。
  - `python3 tools/audit.py downloads`：核对 `configs/models.yaml`、`downloaded_models/` 与抽取产物是否完整，检查矩阵大小。
  - `python3 tools/audit.py tied`：核对 tied/untied 判断，并检查冗余 shard、重复目录和临时目录。
  - `python3 tools/audit.py all`：依次执行全部审计。
- `cleanup_redundant.py`：按内置冗余清单 dry-run 或实际删除缓存中的冗余权重。
- `paths.py`：工具脚本共享的仓库根定位 helper，不直接运行。

## 常用命令

```bash
python3 tools/get_model_useful.py
python3 tools/audit.py all
python3 tools/cleanup_redundant.py
python3 tools/cleanup_redundant.py --apply
```
