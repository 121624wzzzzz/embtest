#!/usr/bin/env python3
"""
任务三：跨系列三档规模 GCorr（基于 extracts/ 与 ijcai_clean 包）。

用法:
  export PYTHONPATH=ijcai_clean/src
  python ijcai_clean/scripts/run_task3_cross_scale_groups.py --devices auto

或在仓库根目录:
  python ijcai_clean/scripts/run_task3_cross_scale_groups.py --devices auto
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _cli import add_gcorr_args, bootstrap_repo, resolve_devices

_REPO_ROOT = bootstrap_repo(__file__)

from ijcai_clean.experiments.task3 import run_task3_cross_scale_groups  # noqa: E402
from ijcai_clean.paths import default_extracts_dir, default_models_yaml  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Task3 Cross-Scale GCorr (extracts/)")
    p.add_argument(
        "--scale-groups",
        type=Path,
        default=_REPO_ROOT / "configs" / "cross_scale_groups.yaml",
        help="三档规模桶配置 YAML",
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
        default=_REPO_ROOT / "ijcai_clean" / "results" / "task3_cross_scale_groups",
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
    )


if __name__ == "__main__":
    main()
