#!/usr/bin/env python3
"""
任务四：MoE-only 跨族 GCorr（基于 extracts/ 与 cross_model_geometry 包）。

用法:
  export PYTHONPATH=cross_model_geometry/src
  python cross_model_geometry/scripts/run_task4_moe_cross_family.py --devices auto

或在仓库根目录:
  python cross_model_geometry/scripts/run_task4_moe_cross_family.py --devices auto
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _cli import add_gcorr_args, bootstrap_repo, resolve_devices

_REPO_ROOT = bootstrap_repo(__file__)

from cross_model_geometry.experiments.task3 import run_task3_cross_scale_groups  # noqa: E402
from cross_model_geometry.paths import default_extracts_dir, default_models_yaml  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Task4 MoE-only Cross-Family GCorr (extracts/)")
    p.add_argument(
        "--scale-groups",
        type=Path,
        default=_REPO_ROOT / "configs" / "moe_cross_family.yaml",
        help="MoE 跨族桶配置 YAML",
    )
    p.add_argument(
        "--series",
        type=Path,
        default=_REPO_ROOT / "configs" / "model_series.yaml",
        help="系列配置 YAML，用于排除同系列 pair",
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
        default=_REPO_ROOT / "cross_model_geometry" / "results" / "task4_moe_cross_family",
        help="输出目录",
    )
    add_gcorr_args(p)
    args = p.parse_args()

    run_task3_cross_scale_groups(
        repo_root=_REPO_ROOT,
        scale_groups_file=args.scale_groups,
        series_file=args.series,
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
        task_name="task4_moe_cross_family",
    )


if __name__ == "__main__":
    main()
