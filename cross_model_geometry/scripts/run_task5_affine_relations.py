#!/usr/bin/env python3
"""
任务五：仿射 R² 关系（E↔E / U↔U / intra E→U），复用 task1..4 的 pair 集合。

用法:
  export PYTHONPATH=cross_model_geometry/src
  python cross_model_geometry/scripts/run_task5_affine_relations.py --devices auto

或在仓库根目录:
  python cross_model_geometry/scripts/run_task5_affine_relations.py --devices auto
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _cli import add_device_args, bootstrap_repo, resolve_devices

_REPO_ROOT = bootstrap_repo(__file__)

from cross_model_geometry.experiments.task5_affine import (  # noqa: E402
    DEFAULT_MAX_FIT_ROWS,
    DEFAULT_MIN_COMMON_TOKENS,
    run_task5_affine_relations,
)
from cross_model_geometry.paths import default_extracts_dir, default_models_yaml  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Task5 Affine Relations (R² of E-E / U-U / intra E->U)")
    p.add_argument(
        "--sources",
        type=Path,
        default=_REPO_ROOT / "configs" / "affine_pairs.yaml",
        help="affine_pairs.yaml；列出 task1..4 generated_pairs.yaml 来源",
    )
    p.add_argument(
        "--extracts",
        type=Path,
        default=default_extracts_dir(_REPO_ROOT),
        help="extracts 目录",
    )
    p.add_argument(
        "--models-yaml",
        type=Path,
        default=default_models_yaml(_REPO_ROOT),
        help="configs/models.yaml",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "cross_model_geometry" / "results" / "task5_affine_subsampled",
        help="输出目录",
    )
    p.add_argument(
        "--max-fit-rows",
        type=int,
        default=DEFAULT_MAX_FIT_ROWS,
        help="单次 lstsq 设计矩阵行数上限（与 legacy 一致默认 24000）",
    )
    p.add_argument(
        "--min-common-tokens",
        type=int,
        default=DEFAULT_MIN_COMMON_TOKENS,
        help="公共 token 数下限，低于则跳过该对（默认 5000，与 legacy 一致）",
    )
    p.add_argument(
        "--cache-max-models",
        type=int,
        default=8,
        help="模型 E/U 矩阵 LRU 缓存数（默认 8，足以覆盖大多数 pair 局部性）",
    )
    add_device_args(p)
    args = p.parse_args()

    run_task5_affine_relations(
        repo_root=_REPO_ROOT,
        sources_file=args.sources,
        extracts_dir=args.extracts,
        models_yaml=args.models_yaml,
        out_dir=args.out,
        devices=resolve_devices(args.devices, args.exclude_gpus),
        max_fit_rows=args.max_fit_rows,
        min_common_tokens=args.min_common_tokens,
        cache_max_models=args.cache_max_models,
    )


if __name__ == "__main__":
    main()
