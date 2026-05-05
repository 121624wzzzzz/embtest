"""兼容入口：见 tools/cleanup/cleanup_redundant.py。"""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "tools" / "cleanup" / "cleanup_redundant.py"),
        run_name="__main__",
    )
