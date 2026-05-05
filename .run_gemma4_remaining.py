"""一次性脚本：续跑剩余 Gemma-4 模型的下载与抽取。

通过 setsid + nohup 在 IDE 之外运行，避免后台 shell 被回收导致下载中断。
跑完即可删除该文件。
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.download.get_model_useful import download_emb_only_with_retry  # noqa: E402

TARGETS = [
    "Gemma-4-26B-A4B",
    "Gemma-4-E2B-Instruct",
    "Gemma-4-E4B-Instruct",
]


def main() -> int:
    cfg = yaml.safe_load((REPO_ROOT / "models.yaml").read_text(encoding="utf-8"))
    config = cfg.get("config") or {}
    cache_dir = REPO_ROOT / config.get("cache_dir", "./downloaded_models")
    extracts_dir = REPO_ROOT / config.get("extracts_dir", "./extracts")
    max_retries = int(config.get("max_retries", 3))
    verify = bool(config.get("verify_extracts", True))
    cleanup = bool(config.get("cleanup_originals", False))
    backup_root = str(extracts_dir) if bool(config.get("backup_extracts", True)) else None
    repo_ids = cfg["model_repo_ids"]

    print(f"Root: {REPO_ROOT}", flush=True)
    print(f"Cache: {cache_dir}", flush=True)
    print(f"Extracts: {extracts_dir}", flush=True)

    for name in TARGETS:
        print("=" * 80, flush=True)
        print(f"TARGET {name} -> {repo_ids[name]}", flush=True)
        ok, msg = download_emb_only_with_retry(
            name,
            repo_ids[name],
            str(cache_dir),
            max_retries,
            backup_root=backup_root,
            verify=verify,
            cleanup_originals=cleanup,
        )
        print(msg, flush=True)
        if not ok:
            return 1

    print("=" * 80, flush=True)
    print("DONE remaining Gemma-4 downloads", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
