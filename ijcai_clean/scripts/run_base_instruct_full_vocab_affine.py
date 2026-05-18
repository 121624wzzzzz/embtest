#!/usr/bin/env python3
"""Task6: Base-Instruct full-vocabulary 仿射 / A 诊断 / SVD 低秩能量。

Source: ``results/task5_affine_subsampled/summary_pair.csv`` 中 ``source_tasks ==
task1_base_instruct`` 的 pair。对这些 pair 按完整词表 id 行直接对齐，使用
流式中心化 normal equations 拟合 ``Y ~= X A + b``，不采样、不排除 token，
随后输出 full-vocab 仿射 R²、`A-I` 诊断、E_delta 与 A-I 的 SVD 能量。

用法:
  python ijcai_clean/scripts/run_base_instruct_full_vocab_affine.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from _cli import bootstrap_repo

_REPO_ROOT = bootstrap_repo(__file__)

from ijcai_clean.experiments.full_vocab_affine import (  # noqa: E402
    BATCH_ROWS,
    csv_fields,
    load_base_instruct_pairs,
    run_full_vocab_pair,
    write_markdown_report,
)


def main() -> None:
    subsampled_dir = _REPO_ROOT / "ijcai_clean" / "results" / "task5_affine_subsampled"
    out_dir = _REPO_ROOT / "ijcai_clean" / "results" / "task6_base_instruct_full_vocab"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_csv = subsampled_dir / "summary_pair.csv"
    out_csv = out_dir / "summary_pair_base_instruct_full_vocab.csv"
    out_md = out_dir / "base_instruct_full_vocab_affine_report.md"
    meta_json = out_dir / "base_instruct_full_vocab_metadata.json"
    extracts = _REPO_ROOT / "extracts"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    pairs = load_base_instruct_pairs(source_csv)

    rows = []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields())
        writer.writeheader()
        for idx, pair in enumerate(pairs, 1):
            model_a = pair["model_a"]
            model_b = pair["model_b"]
            print(f"[{idx}/{len(pairs)}] {model_a} -> {model_b}", flush=True)
            row = run_full_vocab_pair(
                model_a=model_a,
                model_b=model_b,
                extracts_dir=extracts,
                device=device,
                dtype=dtype,
                batch_rows=BATCH_ROWS,
            )
            writer.writerow(row)
            f.flush()
            rows.append(row)
            print(
                f"  R2_E={row['R2_E']:.6f} R2_U={row['R2_U']:.6f} "
                f"rel_A-I/I={row['E_rel_A_minus_I_over_I']:.4f} "
                f"E_rank95={row['E_delta_E_delta_rank_95']} "
                f"A-I_rank95={row['A_delta_A_delta_rank_95']} "
                f"n={row['vocab_size_a']} d={row['hidden_dim_a']} "
                f"elapsed={row['elapsed_sec']:.1f}s",
                flush=True,
            )

    write_markdown_report(rows, out_md)
    meta_json.write_text(
        json.dumps(
            {
                "task": "task6_base_instruct_full_vocab",
                "source_csv": str(source_csv.relative_to(_REPO_ROOT)),
                "out_csv": str(out_csv.relative_to(_REPO_ROOT)),
                "out_md": str(out_md.relative_to(_REPO_ROOT)),
                "n_pairs": len(rows),
                "device": str(device),
                "batch_rows": BATCH_ROWS,
                "dtype": str(dtype),
                "method": "full id-aligned vocabulary, centered normal equations, no row sampling, A diagnostics and SVD energy included",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"DONE {out_csv}", flush=True)


if __name__ == "__main__":
    main()
