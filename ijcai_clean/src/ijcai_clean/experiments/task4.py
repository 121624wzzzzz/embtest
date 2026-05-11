"""任务四：MoE-only 跨族 GCorr。

复用 task3 的 cross-scale 生成器（pair 构造、同系列剔除、写 generated_pairs.yaml），
只改写 metadata 中的任务名为 ``task4_moe_cross_family``，避免重复实现。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from ijcai_clean.experiments.task3 import run_task3_cross_scale_groups

if TYPE_CHECKING:
    import torch


_TASK_NAME = "task4_moe_cross_family"


def _patch_metadata_task_name(meta_json: Path, task_name: str) -> None:
    if not meta_json.is_file():
        return
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    meta["task"] = task_name
    meta_json.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def run_task4_moe_cross_family(
    *,
    repo_root: Path,
    scale_groups_file: Path,
    series_file: Path,
    extracts_dir: Path,
    models_yaml: Path,
    out_dir: Path,
    n_tokens: int,
    n_pairs: int,
    n_bootstrap: int,
    random_seed: int,
    devices: Sequence["torch.device"],
    complete_mode: str = "validate",
    validation_n_tokens: int = 1024,
    validation_n_pairs: int = 10000,
) -> None:
    run_task3_cross_scale_groups(
        repo_root=repo_root,
        scale_groups_file=scale_groups_file,
        series_file=series_file,
        extracts_dir=extracts_dir,
        models_yaml=models_yaml,
        out_dir=out_dir,
        n_tokens=n_tokens,
        n_pairs=n_pairs,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
        devices=devices,
        complete_mode=complete_mode,
        validation_n_tokens=validation_n_tokens,
        validation_n_pairs=validation_n_pairs,
    )
    _patch_metadata_task_name(out_dir / "metadata.json", _TASK_NAME)
