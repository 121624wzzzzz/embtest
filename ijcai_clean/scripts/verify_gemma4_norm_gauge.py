#!/usr/bin/env python3
"""核验 Gemma-4 BI：原始 embed_tokens vs 逐行 L2 归一化后的 GCorr。

输出到 results/_analysis/，不修改 task1 已存 CSV。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _cli import bootstrap_repo

_REPO = bootstrap_repo(__file__)
sys.path.insert(0, str(_REPO / "ijcai_clean" / "src"))

from ijcai_clean.alignment import TokenSampler, build_pair_token_info
from ijcai_clean.data import find_model_cache_dir, load_E_U_matrices, load_info_json
from ijcai_clean.metrics import compute_single_pair_bootstrap
from ijcai_clean.paths import default_extracts_dir, default_models_yaml

DEFAULT_PAIRS = [
    ("Gemma-4-26B-A4B", "Gemma-4-26B-A4B-Instruct"),
    ("Gemma-4-31B", "Gemma-4-31B-Instruct"),
    ("Gemma-4-E2B", "Gemma-4-E2B-Instruct"),
    ("Gemma-4-E4B", "Gemma-4-E4B-Instruct"),
    ("Qwen3-8B-Base", "Qwen3-8B"),
]


def _load_bundle(extracts_dir: Path, name: str):
    info = load_info_json(extracts_dir, name)
    E, U, info = load_E_U_matrices(extracts_dir, name, info=info)
    dims = info["standardized_dims"]["embed"]
    tied = bool(np.allclose(E, U, rtol=1e-5, atol=1e-5))
    return {
        "E": E,
        "U": U,
        "info": info,
        "hidden_dim": int(dims[1]),
        "vocab_size": int(dims[0]),
        "actual_tied": tied,
    }


def _row_unit(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.clip(n, 1e-12, None)


def _norm_stats(M: np.ndarray, sample: int = 30000) -> dict:
    rng = np.random.default_rng(0)
    n = M.shape[0]
    idx = rng.choice(n, min(sample, n), replace=False)
    norms = np.linalg.norm(M[idx], axis=1)
    return {
        "mean": float(norms.mean()),
        "std": float(norms.std()),
        "median": float(np.median(norms)),
        "frac_near_1": float((np.abs(norms - 1.0) < 0.01).mean()),
    }


def _apply_variant(Ea: np.ndarray, Eb: np.ndarray, variant: str):
    if variant == "raw":
        return Ea.copy(), Eb.copy()
    if variant == "row_unit":
        return _row_unit(Ea), _row_unit(Eb)
    if variant == "match_inst_norm_to_base":
        # 仅把 instruct 每行缩放到与 base 同行范数（方向不变）
        na = np.linalg.norm(Ea, axis=1, keepdims=True)
        nb = np.linalg.norm(Eb, axis=1, keepdims=True)
        Eb2 = Eb * (na / np.clip(nb, 1e-12, None))
        return Ea.copy(), Eb2
    raise ValueError(variant)


def main() -> None:
    p = argparse.ArgumentParser(description="Gemma-4 行范数标尺 GCorr 核验")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-tokens", type=int, default=20000)
    p.add_argument("--n-pairs", type=int, default=5_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "ijcai_clean" / "results" / "_analysis",
    )
    args = p.parse_args()

    extracts_dir = default_extracts_dir(_REPO)
    models_yaml = default_models_yaml(_REPO)
    device = torch.device(args.device)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs_file = _REPO / "configs" / "base_instruct_pairs.yaml"
    cfg = yaml.safe_load(pairs_file.read_text(encoding="utf-8"))
    all_pairs = [tuple(x) for x in cfg["pairs"]]
    pairs = [pr for pr in DEFAULT_PAIRS if pr in all_pairs or (pr[1], pr[0]) in all_pairs]
    if not pairs:
        pairs = DEFAULT_PAIRS

    task1_summary = _REPO / "ijcai_clean" / "results" / "task1_base_instruct" / "summary.csv"
    saved = {}
    if task1_summary.is_file():
        for r in csv.DictReader(task1_summary.open(encoding="utf-8")):
            saved[(r["model_a"], r["model_b"])] = r

    sampler = TokenSampler()
    variants = ["raw", "row_unit", "match_inst_norm_to_base"]
    rows_out = []
    norm_rows = []

    def get_tok(name: str):
        d = find_model_cache_dir(_REPO, name, models_yaml)
        if d is None:
            return None
        return AutoTokenizer.from_pretrained(str(d), trust_remote_code=True)

    print(f"device={device} n_tokens={args.n_tokens} n_pairs={args.n_pairs} seed={args.seed}")
    print(f"pairs={len(pairs)} variants={variants}")

    for model_a, model_b in pairs:
        print(f"\n=== {model_a} -> {model_b} ===", flush=True)
        ia = load_info_json(extracts_dir, model_a)
        ib = load_info_json(extracts_dir, model_b)
        info = build_pair_token_info(
            vocab_a=int(ia["standardized_dims"]["embed"][0]),
            vocab_b=int(ib["standardized_dims"]["embed"][0]),
            tokenizer_a=get_tok(model_a),
            tokenizer_b=get_tok(model_b),
            n_tokens=args.n_tokens,
            sampler=sampler,
        )
        if info is None:
            print("  skip: common tokens < 1000", flush=True)
            continue

        data_a = _load_bundle(extracts_dir, model_a)
        data_b = _load_bundle(extracts_dir, model_b)

        for label, M in [("base", data_a["E"]), ("instruct", data_b["E"])]:
            st = _norm_stats(M)
            norm_rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "side": label,
                    **{f"norm_{k}": v for k, v in st.items()},
                }
            )
            print(
                f"  {label} norm: mean={st['mean']:.4f} med={st['median']:.4f} "
                f"frac|norm-1|<0.01={st['frac_near_1']:.2%}",
                flush=True,
            )

        if info["same_tokenizer"]:
            sample_ids = sampler.sample_tokens(info["ids_a"], info["n_sample"], args.seed)
            sa = sb = sample_ids
        else:
            rng = np.random.default_rng(args.seed)
            indices = rng.choice(len(info["ids_a"]), size=info["n_sample"], replace=False)
            sa = info["ids_a"][indices]
            sb = info["ids_b"][indices]

        Ea_raw = data_a["E"][sa]
        Eb_raw = data_b["E"][sb]
        Ua_raw = data_a["U"][sa]
        Ub_raw = data_b["U"][sb]

        ref = saved.get((model_a, model_b), {})

        for variant in variants:
            Ea, Eb = _apply_variant(Ea_raw, Eb_raw, variant)
            if data_a["actual_tied"]:
                Ua, Ub = Ea, Eb
            else:
                Ua, Ub = _apply_variant(Ua_raw, Ub_raw, variant)

            res = compute_single_pair_bootstrap(
                Ea,
                Eb,
                Ua,
                Ub,
                args.n_pairs,
                args.seed,
                device,
            )
            row = {
                "model_a": model_a,
                "model_b": model_b,
                "variant": variant,
                "n_tokens": info["n_sample"],
                "n_pairs": args.n_pairs,
                "seed": args.seed,
                "gcorr_E_cos": res["gcorr_E_cos"],
                "gcorr_E_euc": res["gcorr_E_euc"],
                "gcorr_E_euc2": res["gcorr_E_euc2"],
                "gcorr_U_euc": res["gcorr_U_euc"],
                "task1_saved_euc": float(ref["gcorr_E_euc_mean"]) if ref else "",
                "task1_saved_cos": float(ref["gcorr_E_cos_mean"]) if ref else "",
            }
            rows_out.append(row)
            print(
                f"  [{variant:28s}] E_euc={res['gcorr_E_euc']:.4f} E_cos={res['gcorr_E_cos']:.4f} "
                f"(task1 saved euc={row['task1_saved_euc']})",
                flush=True,
            )

    gcorr_csv = out_dir / "gemma4_norm_gauge_gcorr_verify.csv"
    norm_csv = out_dir / "gemma4_norm_gauge_row_norms.csv"
    meta_json = out_dir / "gemma4_norm_gauge_metadata.json"

    if rows_out:
        with gcorr_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
    if norm_rows:
        with norm_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(norm_rows[0].keys()))
            w.writeheader()
            w.writerows(norm_rows)

    meta_json.write_text(
        json.dumps(
            {
                "device": str(device),
                "n_tokens": args.n_tokens,
                "n_pairs": args.n_pairs,
                "seed": args.seed,
                "variants": variants,
                "pairs": [list(p) for p in pairs],
                "outputs": {
                    "gcorr_csv": str(gcorr_csv.relative_to(_REPO)),
                    "norm_csv": str(norm_csv.relative_to(_REPO)),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {gcorr_csv}", flush=True)
    print(f"Wrote {norm_csv}", flush=True)


if __name__ == "__main__":
    main()
