"""兼容入口：下载与抽取逻辑见 tools/download/get_model_useful.py。"""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "tools" / "download" / "get_model_useful.py"),
        run_name="__main__",
    )
