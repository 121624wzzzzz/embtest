#!/usr/bin/env python3
"""
任务一：Base–Instruct GCorr（基于 extracts/ 与 ijcai_clean 包）。

用法:
  export PYTHONPATH=ijcai_clean/src
  python ijcai_clean/scripts/run_task1_base_instruct.py --devices auto

或在仓库根目录:
  python ijcai_clean/scripts/run_task1_base_instruct.py --devices auto
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 必须最先执行：保证 ijcai_clean 包在 import torch 等之前已加入 path
_SCRIPT_DIR = Path(__file__).resolve().parent


def _find_repo_root(start: Path) -> Path:
    """向上查找含 models.yaml 的目录（兼容不同克隆/挂载深度）。"""
    for d in [start, *start.parents]:
        if (d / "models.yaml").is_file():
            return d
    raise RuntimeError(
        "找不到仓库根目录（未在上级路径发现 models.yaml）。"
        "请在 get_useful 仓库根目录下运行，或设置环境变量 REPO_ROOT。"
    )


_REPO_ROOT = Path(os.environ["REPO_ROOT"]).resolve() if os.environ.get("REPO_ROOT") else _find_repo_root(_SCRIPT_DIR)
_SRC = (_REPO_ROOT / "ijcai_clean" / "src").resolve()
if not _SRC.is_dir():
    raise RuntimeError(f"找不到分析包目录: {_SRC}")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch

from ijcai_clean.experiments.task1 import run_task1_base_instruct  # noqa: E402
from ijcai_clean.paths import default_extracts_dir, default_models_yaml  # noqa: E402


def parse_devices(spec: str) -> list[torch.device]:
    spec = spec.strip().lower()
    if spec == "auto":
        n = torch.cuda.device_count()
        if n == 0:
            return [torch.device("cpu")]
        return [torch.device(f"cuda:{i}") for i in range(n)]
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return [torch.device(p) for p in parts]


def main() -> None:
    p = argparse.ArgumentParser(description="Task1 Base-Instruct GCorr (extracts/)")
    p.add_argument(
        "--pairs",
        type=Path,
        default=_REPO_ROOT / "ijcai_clean" / "configs" / "base_instruct_pairs.yaml",
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
        help="根目录 models.yaml",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "ijcai_clean" / "results" / "task1_base_instruct",
        help="输出目录",
    )
    p.add_argument("--n-tokens", type=int, default=20_000)
    p.add_argument("--n-pairs", type=int, default=5_000_000)
    p.add_argument("--n-bootstrap", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--devices", type=str, default="auto", help='如 auto 或 "cuda:0,cuda:1"')
    p.add_argument(
        "--complete-mode",
        choices=("validate", "csv-only"),
        default="validate",
        help=(
            "当 bootstrap 已完整时的行为：validate=仍加载矩阵并做小规模 GCorr 校验；"
            "csv-only=只从 CSV 快速重建 summary"
        ),
    )
    p.add_argument(
        "--validation-n-tokens",
        type=int,
        default=1024,
        help="complete-mode=validate 时每组 pair 采样 token 数",
    )
    p.add_argument(
        "--validation-n-pairs",
        type=int,
        default=10_000,
        help="complete-mode=validate 时每组 pair 采样 token-pair 数",
    )
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
        devices=parse_devices(args.devices),
        complete_mode=args.complete_mode,
        validation_n_tokens=args.validation_n_tokens,
        validation_n_pairs=args.validation_n_pairs,
    )


if __name__ == "__main__":
    main()
