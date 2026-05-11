"""解析仓库根目录：支持工具脚本位于 tools/ 下。"""

from __future__ import annotations

from pathlib import Path


def repository_root(caller: str | Path) -> Path:
    p = Path(caller).resolve()
    if p.parent.name == "tools":
        candidate = p.parents[1]
        if (candidate / "configs" / "models.yaml").is_file() or (candidate / "models.yaml").is_file():
            return candidate
    for d in [p.parent, *p.parents]:
        if (d / "configs" / "models.yaml").is_file() or (d / "models.yaml").is_file():
            return d
    return p.parent
