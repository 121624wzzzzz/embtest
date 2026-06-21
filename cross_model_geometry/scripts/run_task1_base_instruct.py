#!/usr/bin/env python3
"""
任务一：Base–Instruct GCorr（基于 extracts/ 与 cross_model_geometry 包）。

用法:
  export PYTHONPATH=cross_model_geometry/src
  python cross_model_geometry/scripts/run_task1_base_instruct.py --devices auto

或在仓库根目录:
  python cross_model_geometry/scripts/run_task1_base_instruct.py --devices auto
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _cli import add_gcorr_args, bootstrap_repo, resolve_devices

_REPO_ROOT = bootstrap_repo(__file__)

from cross_model_geometry.experiments.task1 import run_task1_base_instruct  # noqa: E402
from cross_model_geometry.paths import default_extracts_dir, default_models_yaml  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Task1 Base-Instruct GCorr (extracts/)")
    p.add_argument(
        "--pairs",
        type=Path,
        default=_REPO_ROOT / "configs" / "base_instruct_pairs.yaml",
        help="pair 配置 YAML",
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
        default=_REPO_ROOT / "cross_model_geometry" / "results" / "task1_base_instruct",
        help="输出目录",
    )
    add_gcorr_args(p)
    args = p.parse_args()

    run_task1_base_instruct(
        repo_root=_REPO_ROOT,
        pairs_file=args.pairs,
        extracts_dir=args.extracts,
        models_yaml=args.models_yaml,
        out_dir=args.out,
        n_tokens=args.n_tokens,
        n_pairs=args.n_pairs,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.seed,
        devices=resolve_devices(args.devices, args.exclude_gpus),
        complete_mode=args.complete_mode,
        validation_n_tokens=args.validation_n_tokens,
        validation_n_pairs=args.validation_n_pairs,
    )


if __name__ == "__main__":
    main()
