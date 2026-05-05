"""解析仓库根目录：支持工具脚本位于 tools/{download,audit,cleanup}/ 下。"""

from __future__ import annotations

from pathlib import Path


def repository_root(caller: str | Path) -> Path:
    p = Path(caller).resolve()
    if p.parent.name in ("download", "audit", "cleanup") and p.parent.parent.name == "tools":
        candidate = p.parents[2]
        if (candidate / "models.yaml").is_file():
            return candidate
    for d in [p.parent, *p.parents]:
        if (d / "models.yaml").is_file():
            return d
    return p.parent
