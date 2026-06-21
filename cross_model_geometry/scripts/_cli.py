from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


def find_repo_root(start: Path) -> Path:
    """Find the repository root from a script path."""
    for directory in [start, *start.parents]:
        if (directory / "configs" / "models.yaml").is_file() or (
            directory / "models.yaml"
        ).is_file():
            return directory
    raise RuntimeError(
        "找不到仓库根目录（未在上级路径发现 configs/models.yaml）。"
        "请在 get_useful 仓库根目录下运行，或设置环境变量 REPO_ROOT。"
    )


def bootstrap_repo(script_file: str) -> Path:
    """Resolve repo root and put cross_model_geometry/src on sys.path before package imports."""
    script_dir = Path(script_file).resolve().parent
    repo_root = (
        Path(os.environ["REPO_ROOT"]).resolve()
        if os.environ.get("REPO_ROOT")
        else find_repo_root(script_dir)
    )
    src = (repo_root / "cross_model_geometry" / "src").resolve()
    if not src.is_dir():
        raise RuntimeError(f"找不到分析包目录: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return repo_root


def parse_devices(spec: str) -> list[torch.device]:
    spec = spec.strip().lower()
    if spec == "auto":
        n = torch.cuda.device_count()
        if n == 0:
            return [torch.device("cpu")]
        return [torch.device(f"cuda:{i}") for i in range(n)]
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return [torch.device(p) for p in parts]


def parse_exclude_gpus(spec: str) -> set[int]:
    out: set[int] = set()
    for raw in (spec or "").split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token.startswith("cuda:"):
            token = token[5:]
        try:
            out.add(int(token))
        except ValueError:
            continue
    return out


def filter_devices(devices: list[torch.device], excluded: set[int]) -> list[torch.device]:
    if not excluded:
        return devices
    kept: list[torch.device] = []
    for device in devices:
        if device.type == "cuda" and device.index in excluded:
            continue
        kept.append(device)
    if not kept:
        raise ValueError(
            "exclude_gpus filtered out all devices; please loosen --exclude-gpus or --devices"
        )
    return kept


def resolve_devices(devices_spec: str, exclude_gpus_spec: str) -> list[torch.device]:
    return filter_devices(parse_devices(devices_spec), parse_exclude_gpus(exclude_gpus_spec))


def add_device_args(parser) -> None:
    parser.add_argument("--devices", type=str, default="auto", help='如 auto 或 "cuda:0,cuda:1"')
    parser.add_argument(
        "--exclude-gpus",
        type=str,
        default="",
        help='排除某些卡，例如 "0" 或 "cuda:0,cuda:7"，常用于躲开常驻 VLLM 等大型外部进程',
    )


def add_gcorr_args(parser) -> None:
    parser.add_argument("--n-tokens", type=int, default=20_000)
    parser.add_argument("--n-pairs", type=int, default=5_000_000)
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    add_device_args(parser)
    parser.add_argument(
        "--complete-mode",
        choices=("validate", "csv-only"),
        default="validate",
        help=(
            "当 bootstrap 已完整时的行为：validate=仍加载矩阵并做小规模 GCorr 校验；"
            "csv-only=只从 CSV 快速重建 summary"
        ),
    )
    parser.add_argument(
        "--validation-n-tokens",
        type=int,
        default=1024,
        help="complete-mode=validate 时每组 pair 采样 token 数",
    )
    parser.add_argument(
        "--validation-n-pairs",
        type=int,
        default=10_000,
        help="complete-mode=validate 时每组 pair 采样 token-pair 数",
    )
