#!/usr/bin/env python3
"""Gemma 全系列 embedding 标尺审计：存盘行范数 + BI 与 Task1 gcorr 对照。

输出到 ijcai_clean/results/_analysis/，不修改 task1 结果。
"""
from __future__ import annotations

import argparse
import csv
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

# 复用 verify_gemma4_norm_gauge 的变体逻辑
from verify_gemma4_norm_gauge import _apply_variant, _load_bundle, _norm_stats


def _generation(name: str) -> str:
    if name.startswith("Gemma-4"):
        return "gemma4"
    if name.startswith("Gemma-3"):
        return "gemma3"
    if name.startswith("Gemma-2"):
        return "gemma2"
    return "gemma_other"


def _is_base(name: str) -> bool:
    return "Instruct" not in name and not name.endswith("-it")


def _full_norm_stats(M: np.ndarray) -> dict:
    norms = np.linalg.norm(M, axis=1)
    return {
        "mean": float(norms.mean()),
        "std": float(norms.std()),
        "median": float(np.median(norms)),
        "frac_near_1": float((np.abs(norms - 1.0) < 0.01).mean()),
        "frac_near_sqrt_d": None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Gemma 系列 embedding 标尺审计")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-tokens", type=int, default=20000)
    p.add_argument("--n-pairs", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-gcorr", action="store_true", help="仅行范数统计，不重算 GCorr")
    p.add_argument("--skip-norms", action="store_true", help="跳过单模型行范数（已有 CSV 时）")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "ijcai_clean" / "results" / "_analysis",
    )
    args = p.parse_args()

    extracts_dir = default_extracts_dir(_REPO)
    models_yaml = default_models_yaml(_REPO)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    gemma_models = sorted(
        p.stem.replace(".info", "")
        for p in extracts_dir.glob("Gemma*.info.json")
    )

    model_csv = out_dir / "gemma_series_row_norms.csv"
    model_rows = []
    if args.skip_norms and model_csv.is_file():
        model_rows = list(csv.DictReader(model_csv.open(encoding="utf-8")))
        print(f"Loaded {len(model_rows)} models from {model_csv}", flush=True)
    # --- 1) 单模型行范数 ---
    for name in ([] if args.skip_norms and model_rows else gemma_models):
        info = load_info_json(extracts_dir, name)
        E, _, _ = load_E_U_matrices(extracts_dir, name, info=info)
        h = int(info["standardized_dims"]["embed"][1])
        st = _full_norm_stats(E)
        sqrt_h = h**0.5
        norms = np.linalg.norm(E, axis=1)
        st["frac_near_sqrt_d"] = float((np.abs(norms - sqrt_h) < 0.05 * sqrt_h).mean())
        st["hidden_size"] = h
        st["sqrt_hidden"] = sqrt_h
        st["runtime_norm_median_if_scaled"] = float(np.median(norms) * sqrt_h)
        model_rows.append(
            {
                "model": name,
                "generation": _generation(name),
                "role": "base" if _is_base(name) else "instruct",
                **st,
            }
        )
        print(
            f"{name:28} h={h:5} med_norm={st['median']:.4f} "
            f"near1={st['frac_near_1']:.1%} med*sqrt(h)={st['runtime_norm_median_if_scaled']:.2f}",
            flush=True,
        )

    if not args.skip_norms:
        with model_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(model_rows[0].keys()))
            w.writeheader()
            w.writerows(model_rows)

    # --- 2) BI 对：范数错位 + Task1 ---
    pairs_file = _REPO / "configs" / "base_instruct_pairs.yaml"
    cfg = yaml.safe_load(pairs_file.read_text(encoding="utf-8"))
    gemma_pairs = [
        tuple(x)
        for x in cfg["pairs"]
        if x[0].startswith("Gemma-")
    ]

    task1 = _REPO / "ijcai_clean" / "results" / "task1_base_instruct" / "summary.csv"
    saved = {}
    if task1.is_file():
        for r in csv.DictReader(task1.open(encoding="utf-8")):
            saved[(r["model_a"], r["model_b"])] = r

    pair_rows = []
    sampler = TokenSampler()

    def get_tok(name: str):
        d = find_model_cache_dir(_REPO, name, models_yaml)
        if d is None:
            return None
        return AutoTokenizer.from_pretrained(str(d), trust_remote_code=True)

    device = torch.device(args.device)

    for model_a, model_b in gemma_pairs:
        print(f"\n=== BI {model_a} | {model_b} ===", flush=True)
        ia = load_info_json(extracts_dir, model_a)
        ib = load_info_json(extracts_dir, model_b)
        Ea, _, _ = load_E_U_matrices(extracts_dir, model_a, info=ia)
        Eb, _, _ = load_E_U_matrices(extracts_dir, model_b, info=ib)
        ha = int(ia["standardized_dims"]["embed"][1])
        hb = int(ib["standardized_dims"]["embed"][1])

        na = np.linalg.norm(Ea, axis=1)
        nb = np.linalg.norm(Eb, axis=1)
        base_st = _norm_stats(Ea)
        inst_st = _norm_stats(Eb)

        info = build_pair_token_info(
            vocab_a=int(ia["standardized_dims"]["embed"][0]),
            vocab_b=int(ib["standardized_dims"]["embed"][0]),
            tokenizer_a=get_tok(model_a),
            tokenizer_b=get_tok(model_b),
            n_tokens=args.n_tokens,
            sampler=sampler,
        )
        aligned_norm_gap = None
        if info is not None:
            if info["same_tokenizer"]:
                ids = sampler.sample_tokens(info["ids_a"], info["n_sample"], args.seed)
                ia_idx, ib_idx = ids, ids
            else:
                rng = np.random.default_rng(args.seed)
                idx = rng.choice(len(info["ids_a"]), size=info["n_sample"], replace=False)
                ia_idx = info["ids_a"][idx]
                ib_idx = info["ids_b"][idx]
            aligned_norm_gap = float(
                np.mean(np.abs(na[ia_idx] - nb[ib_idx]))
            )

        ref = saved.get((model_a, model_b), {})
        t1_cos = float(ref["gcorr_E_cos_mean"]) if ref else None
        t1_euc = float(ref["gcorr_E_euc_mean"]) if ref else None
        t1_gap = (t1_cos - t1_euc) if (t1_cos is not None and t1_euc is not None) else None

        row = {
            "model_a": model_a,
            "model_b": model_b,
            "generation": _generation(model_a),
            "hidden_a": ha,
            "hidden_b": hb,
            "base_norm_median": base_st["median"],
            "inst_norm_median": inst_st["median"],
            "inst_over_base_norm_ratio": inst_st["median"] / max(base_st["median"], 1e-12),
            "base_frac_near_1": base_st["frac_near_1"],
            "inst_frac_near_1": inst_st["frac_near_1"],
            "aligned_mean_abs_norm_diff": aligned_norm_gap,
            "task1_gcorr_E_cos": t1_cos,
            "task1_gcorr_E_euc": t1_euc,
            "task1_cos_minus_euc": t1_gap,
            "verify_gcorr_E_cos_raw": None,
            "verify_gcorr_E_euc_raw": None,
            "verify_gcorr_E_cos_row_unit": None,
            "verify_gcorr_E_euc_row_unit": None,
        }

        if not args.skip_gcorr and info is not None:
            data_a = _load_bundle(extracts_dir, model_a)
            data_b = _load_bundle(extracts_dir, model_b)
            if info["same_tokenizer"]:
                sample_ids = sampler.sample_tokens(info["ids_a"], info["n_sample"], args.seed)
                sa = sb = sample_ids
            else:
                rng = np.random.default_rng(args.seed)
                indices = rng.choice(len(info["ids_a"]), size=info["n_sample"], replace=False)
                sa = info["ids_a"][indices]
                sb = info["ids_b"][indices]
            Ea_s = data_a["E"][sa]
            Eb_s = data_b["E"][sb]
            for variant in ("raw", "row_unit"):
                Eav, Ebv = _apply_variant(Ea_s, Eb_s, variant)
                res = compute_single_pair_bootstrap(
                    Eav, Ebv, Eav, Ebv, args.n_pairs, args.seed, device
                )
                row[f"verify_gcorr_E_cos_{variant}"] = res["gcorr_E_cos"]
                row[f"verify_gcorr_E_euc_{variant}"] = res["gcorr_E_euc"]
                print(
                    f"  {variant}: cos={res['gcorr_E_cos']:.4f} "
                    f"euc={res['gcorr_E_euc']:.4f}",
                    flush=True,
                )

        pair_rows.append(row)

    pair_csv = out_dir / "gemma_series_bi_norm_gcorr.csv"
    with pair_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)

    # --- 3) Transformers 实现摘要 ---
    impl_rows = []
    import transformers

    tf_root = Path(transformers.__file__).parent / "models"
    for gen in ("gemma", "gemma2", "gemma3", "gemma3n"):
        f = tf_root / gen / f"modeling_{gen}.py"
        if not f.is_file():
            continue
        text = f.read_text(encoding="utf-8")
        impl_rows.append(
            {
                "generation": gen,
                "ScaledWordEmbedding": "ScaledWordEmbedding" in text,
                "sqrt_hidden_in_modeling": "hidden_size**0.5" in text,
            }
        )
    impl_csv = out_dir / "gemma_series_transformers_impl.csv"
    if impl_rows:
        with impl_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(impl_rows[0].keys()))
            w.writeheader()
            w.writerows(impl_rows)

    print(f"\nWrote:\n  {model_csv}\n  {pair_csv}", flush=True)
    if impl_rows:
        print(f"  {impl_csv}", flush=True)


if __name__ == "__main__":
    main()
