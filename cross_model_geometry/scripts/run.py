#!/usr/bin/env python3
"""cross_model_geometry 数据驱动 CLI 派发入口。

用法:
  python cross_model_geometry/scripts/run.py {task1|task2|task3|task4|task5|task6|list} [task 自身参数...]
  python cross_model_geometry/scripts/run.py task1 --devices auto
  python cross_model_geometry/scripts/run.py task5 --max-fit-rows 20000
  python cross_model_geometry/scripts/run.py list
  python cross_model_geometry/scripts/run.py task1 --help

各 task 旧入口 ``scripts/run_task*_*.py`` 仍可用，作为向后兼容的薄包装。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _cli import add_device_args, add_gcorr_args, bootstrap_repo, resolve_devices

_REPO_ROOT = bootstrap_repo(__file__)

from cross_model_geometry.paths import default_extracts_dir, default_models_yaml  # noqa: E402


def _task1(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--pairs",
        type=Path,
        default=_REPO_ROOT / "configs" / "base_instruct_pairs.yaml",
    )
    p.add_argument("--extracts", type=Path, default=default_extracts_dir(_REPO_ROOT))
    p.add_argument("--models-yaml", type=Path, default=default_models_yaml(_REPO_ROOT))
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "cross_model_geometry" / "results" / "task1_base_instruct",
    )
    add_gcorr_args(p)


def _run_task1(args: argparse.Namespace) -> None:
    from cross_model_geometry.experiments.task1 import run_task1_base_instruct

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


def _task2(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--series",
        type=Path,
        default=_REPO_ROOT / "configs" / "model_series.yaml",
    )
    p.add_argument("--extracts", type=Path, default=default_extracts_dir(_REPO_ROOT))
    p.add_argument("--models-yaml", type=Path, default=default_models_yaml(_REPO_ROOT))
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "cross_model_geometry" / "results" / "task2_model_series",
    )
    add_gcorr_args(p)


def _run_task2(args: argparse.Namespace) -> None:
    from cross_model_geometry.experiments.task2 import run_task2_model_series

    run_task2_model_series(
        repo_root=_REPO_ROOT,
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


def _task3(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--scale-groups",
        type=Path,
        default=_REPO_ROOT / "configs" / "cross_scale_groups.yaml",
    )
    p.add_argument(
        "--series",
        type=Path,
        default=_REPO_ROOT / "configs" / "model_series.yaml",
    )
    p.add_argument("--extracts", type=Path, default=default_extracts_dir(_REPO_ROOT))
    p.add_argument("--models-yaml", type=Path, default=default_models_yaml(_REPO_ROOT))
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "cross_model_geometry" / "results" / "task3_cross_scale_groups",
    )
    add_gcorr_args(p)


def _run_task3(args: argparse.Namespace, task_name: str = "task3_cross_scale_groups") -> None:
    from cross_model_geometry.experiments.task3 import run_task3_cross_scale_groups

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
        task_name=task_name,
    )


def _task4(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--scale-groups",
        type=Path,
        default=_REPO_ROOT / "configs" / "moe_cross_family.yaml",
    )
    p.add_argument(
        "--series",
        type=Path,
        default=_REPO_ROOT / "configs" / "model_series.yaml",
    )
    p.add_argument("--extracts", type=Path, default=default_extracts_dir(_REPO_ROOT))
    p.add_argument("--models-yaml", type=Path, default=default_models_yaml(_REPO_ROOT))
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "cross_model_geometry" / "results" / "task4_moe_cross_family",
    )
    add_gcorr_args(p)


def _run_task4(args: argparse.Namespace) -> None:
    _run_task3(args, task_name="task4_moe_cross_family")


def _task5(p: argparse.ArgumentParser) -> None:
    from cross_model_geometry.experiments.task5_affine import (
        DEFAULT_MAX_FIT_ROWS,
        DEFAULT_MIN_COMMON_TOKENS,
    )

    p.add_argument(
        "--sources",
        type=Path,
        default=_REPO_ROOT / "configs" / "affine_pairs.yaml",
    )
    p.add_argument("--extracts", type=Path, default=default_extracts_dir(_REPO_ROOT))
    p.add_argument("--models-yaml", type=Path, default=default_models_yaml(_REPO_ROOT))
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "cross_model_geometry" / "results" / "task5_affine_subsampled",
    )
    p.add_argument("--max-fit-rows", type=int, default=DEFAULT_MAX_FIT_ROWS)
    p.add_argument("--min-common-tokens", type=int, default=DEFAULT_MIN_COMMON_TOKENS)
    p.add_argument("--cache-max-models", type=int, default=8)
    add_device_args(p)


def _run_task5(args: argparse.Namespace) -> None:
    from cross_model_geometry.experiments.task5_affine import run_task5_affine_relations

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


def _task6(p: argparse.ArgumentParser) -> None:
    """Task6 当前不需要 CLI 参数，调用前会读取 task5 子采样结果。"""


def _run_task6(_args: argparse.Namespace) -> None:
    from runpy import run_path

    run_path(str(Path(__file__).with_name("run_task6_base_instruct_full_vocab_affine.py")), run_name="__main__")


TASKS: dict[str, dict[str, object]] = {
    "task1": {
        "help": "Base-Instruct GCorr",
        "add_args": _task1,
        "run": _run_task1,
    },
    "task2": {
        "help": "Model-Series GCorr",
        "add_args": _task2,
        "run": _run_task2,
    },
    "task3": {
        "help": "Cross-Scale GCorr",
        "add_args": _task3,
        "run": _run_task3,
    },
    "task4": {
        "help": "MoE Cross-Family GCorr (复用 Task3 runner)",
        "add_args": _task4,
        "run": _run_task4,
    },
    "task5": {
        "help": "Affine Relations (子采样 lstsq)",
        "add_args": _task5,
        "run": _run_task5,
    },
    "task6": {
        "help": "Base-Instruct Full-Vocab Affine / A-I / SVD",
        "add_args": _task6,
        "run": _run_task6,
    },
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="cross_model_geometry 任务派发入口；用 `run.py list` 查看可用任务。",
    )
    subparsers = parser.add_subparsers(dest="task", metavar="TASK")
    subparsers.required = True
    for name, spec in TASKS.items():
        sub = subparsers.add_parser(name, help=str(spec["help"]))
        spec["add_args"](sub)  # type: ignore[operator]
    subparsers.add_parser("list", help="列出可用任务")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.task == "list":
        for name, spec in TASKS.items():
            print(f"  {name:6s}  {spec['help']}")
        return
    TASKS[args.task]["run"](args)  # type: ignore[operator]


if __name__ == "__main__":
    main(sys.argv[1:])
