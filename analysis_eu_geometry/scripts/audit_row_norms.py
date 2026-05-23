#!/usr/bin/env python3
"""全库 E/U checkpoint 几何审计 CLI（行范数、μ-ratio、谱分析）。

读取 extracts/ 中的 embedding 与 lm_head，按模型范围输出到 results/row_norms/。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from row_norm_audit_common import (
    bootstrap_repo,
    merge_all_models_row_norms,
    run_base_instruct_audit,
    run_mu_ratio_audit,
    run_other_models_audit,
    run_spectral_audit,
)

_REPO = bootstrap_repo()

from ijcai_clean.paths import default_extracts_dir, default_models_yaml  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="E/U 行范数审计")
    p.add_argument(
        "--scope",
        choices=("all", "base_instruct", "other_models"),
        default="all",
        help="模型范围：all=BI+other+合并（默认），base_instruct=70，other_models=24",
    )
    p.add_argument(
        "--pairs",
        type=Path,
        default=_REPO / "configs" / "base_instruct_pairs.yaml",
    )
    p.add_argument("--models-yaml", type=Path, default=default_models_yaml(_REPO))
    p.add_argument("--extracts", type=Path, default=default_extracts_dir(_REPO))
    p.add_argument(
        "--merge-only",
        action="store_true",
        help="只合并已有 CSV 为 94 模型总表，不重算",
    )
    p.add_argument(
        "--mu-ratio-only",
        action="store_true",
        help="只算 ||μ|| 与 ||μ||/mean(row norm)，跳过后续 row-norm 全量统计",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="--mu-ratio-only 时的 CPU 并行进程数（默认 min(16, cpu_count)）",
    )
    p.add_argument(
        "--svd-only",
        action="store_true",
        help="GPU Gram top-1 谱分析（σ1、σ1_c、cos(v1,μ) 等）",
    )
    p.add_argument(
        "--device",
        default="cuda:0",
        help="--svd-only 使用的 GPU（默认 cuda:0）",
    )
    args = p.parse_args()

    if args.svd_only:
        out_csv = run_spectral_audit(
            _REPO,
            models_yaml=args.models_yaml,
            pairs_file=args.pairs,
            extracts_dir=args.extracts,
            device=args.device,
        )
        print(f"\nSpectral table -> {out_csv}", flush=True)
        return

    if args.mu_ratio_only:
        out_csv = run_mu_ratio_audit(
            _REPO,
            models_yaml=args.models_yaml,
            pairs_file=args.pairs,
            extracts_dir=args.extracts,
            workers=args.workers,
        )
        print(f"\nMu-ratio table -> {out_csv}", flush=True)
        return

    if not args.merge_only:
        if args.scope in ("all", "base_instruct"):
            run_base_instruct_audit(
                _REPO,
                pairs_file=args.pairs,
                extracts_dir=args.extracts,
            )
        if args.scope in ("all", "other_models"):
            run_other_models_audit(
                _REPO,
                models_yaml=args.models_yaml,
                pairs_file=args.pairs,
                extracts_dir=args.extracts,
            )

    if args.scope == "all" or args.merge_only:
        out_csv = merge_all_models_row_norms(_REPO, models_yaml=args.models_yaml)
        print(f"\nMerged 94-model table -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
