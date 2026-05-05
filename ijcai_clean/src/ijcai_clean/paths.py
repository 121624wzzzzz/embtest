from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """ijcai_clean/src/ijcai_clean/paths.py -> 仓库根目录 get_useful。"""
    return Path(__file__).resolve().parents[3]


def ijcai_clean_root() -> Path:
    """ijcai_clean 子项目根目录。"""
    return Path(__file__).resolve().parents[2]


def default_extracts_dir(root: Path | None = None) -> Path:
    r = root or repo_root()
    return r / "extracts"


def default_models_yaml(root: Path | None = None) -> Path:
    r = root or repo_root()
    return r / "models.yaml"


def default_cache_dir(root: Path | None = None) -> Path:
    r = root or repo_root()
    return r / "downloaded_models"


def rel_to_repo(path: Path, root: Path | None = None) -> str:
    """尽量返回相对仓库根的路径字符串。"""
    r = root or repo_root()
    try:
        return str(path.resolve().relative_to(r.resolve()))
    except ValueError:
        return str(path.resolve())
