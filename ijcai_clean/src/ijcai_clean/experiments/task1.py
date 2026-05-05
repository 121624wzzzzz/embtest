from __future__ import annotations

import csv
import gc
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from ijcai_clean.alignment import TokenSampler, build_pair_token_info
from ijcai_clean.data import (
    actual_tied,
    find_model_cache_dir,
    load_E_U_matrices,
    load_E_U_matrix_rows,
    load_info_json,
)
from ijcai_clean.metrics import compute_single_pair_bootstrap
from ijcai_clean.paths import rel_to_repo

BOOTSTRAP_CSV_FIELDS = [
    "model_a",
    "model_b",
    "bootstrap",
    "align_mode",
    "n_tokens",
    "n_pairs",
    "hidden_dim_a",
    "hidden_dim_b",
    "vocab_size_a",
    "vocab_size_b",
    "actual_tied_a",
    "actual_tied_b",
    "gcorr_E_cos",
    "gcorr_E_euc",
    "gcorr_E_euc2",
    "gcorr_U_cos",
    "gcorr_U_euc",
    "gcorr_U_euc2",
]


def load_pairs_yaml(path: Path) -> List[Tuple[str, str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    pairs_raw = data.get("pairs") or []
    out: List[Tuple[str, str]] = []
    for item in pairs_raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((str(item[0]).strip(), str(item[1]).strip()))
    return out


def read_existing_bootstrap_rows(csv_path: Path) -> Set[Tuple[str, str, int]]:
    if not csv_path.is_file():
        return set()
    done: Set[Tuple[str, str, int]] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                b = int(row["bootstrap"])
            except (KeyError, ValueError):
                continue
            done.add((row["model_a"], row["model_b"], b))
    return done


def load_bootstrap_rows_by_pair(boot_csv: Path) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    """按模型对分组并按 bootstrap 序号排序。"""
    by_pair: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    if not boot_csv.is_file():
        return {}
    with boot_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_pair[(row["model_a"], row["model_b"])].append(row)
    for key in by_pair:
        by_pair[key].sort(key=lambda r: int(r["bootstrap"]))
    return by_pair


def _summarize_row(
    model_a: str,
    model_b: str,
    data_a: Dict[str, Any],
    data_b: Dict[str, Any],
    bootstrap_results: List[Dict[str, float]],
    n_tokens: int,
    align_mode: str,
    n_bootstrap: int,
    n_pairs: int,
) -> Dict[str, Any]:
    def stat(key: str) -> Dict[str, float]:
        values = np.array([r[key] for r in bootstrap_results])
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "ci95_low": float(np.percentile(values, 2.5)),
            "ci95_high": float(np.percentile(values, 97.5)),
            "median": float(np.median(values)),
            "se": float(np.std(values) / np.sqrt(len(values))),
        }

    E_cos = stat("gcorr_E_cos")
    E_euc = stat("gcorr_E_euc")
    E_euc2 = stat("gcorr_E_euc2")
    U_cos = stat("gcorr_U_cos")
    U_euc = stat("gcorr_U_euc")
    U_euc2 = stat("gcorr_U_euc2")

    return {
        "model_a": model_a,
        "model_b": model_b,
        "align_mode": align_mode,
        "n_tokens": n_tokens,
        "n_bootstrap": n_bootstrap,
        "n_pairs": n_pairs,
        "hidden_dim_a": data_a["hidden_dim"],
        "hidden_dim_b": data_b["hidden_dim"],
        "vocab_size_a": data_a["vocab_size"],
        "vocab_size_b": data_b["vocab_size"],
        "actual_tied_a": data_a["actual_tied"],
        "actual_tied_b": data_b["actual_tied"],
        "gcorr_E_cos_mean": E_cos["mean"],
        "gcorr_E_cos_std": E_cos["std"],
        "gcorr_E_cos_se": E_cos["se"],
        "gcorr_E_cos_ci95_low": E_cos["ci95_low"],
        "gcorr_E_cos_ci95_high": E_cos["ci95_high"],
        "gcorr_E_cos_median": E_cos["median"],
        "gcorr_E_euc_mean": E_euc["mean"],
        "gcorr_E_euc_std": E_euc["std"],
        "gcorr_E_euc_se": E_euc["se"],
        "gcorr_E_euc_ci95_low": E_euc["ci95_low"],
        "gcorr_E_euc_ci95_high": E_euc["ci95_high"],
        "gcorr_E_euc_median": E_euc["median"],
        "gcorr_E_euc2_mean": E_euc2["mean"],
        "gcorr_E_euc2_std": E_euc2["std"],
        "gcorr_E_euc2_se": E_euc2["se"],
        "gcorr_E_euc2_ci95_low": E_euc2["ci95_low"],
        "gcorr_E_euc2_ci95_high": E_euc2["ci95_high"],
        "gcorr_E_euc2_median": E_euc2["median"],
        "gcorr_U_cos_mean": U_cos["mean"],
        "gcorr_U_cos_std": U_cos["std"],
        "gcorr_U_cos_se": U_cos["se"],
        "gcorr_U_cos_ci95_low": U_cos["ci95_low"],
        "gcorr_U_cos_ci95_high": U_cos["ci95_high"],
        "gcorr_U_cos_median": U_cos["median"],
        "gcorr_U_euc_mean": U_euc["mean"],
        "gcorr_U_euc_std": U_euc["std"],
        "gcorr_U_euc_se": U_euc["se"],
        "gcorr_U_euc_ci95_low": U_euc["ci95_low"],
        "gcorr_U_euc_ci95_high": U_euc["ci95_high"],
        "gcorr_U_euc_median": U_euc["median"],
        "gcorr_U_euc2_mean": U_euc2["mean"],
        "gcorr_U_euc2_std": U_euc2["std"],
        "gcorr_U_euc2_se": U_euc2["se"],
        "gcorr_U_euc2_ci95_low": U_euc2["ci95_low"],
        "gcorr_U_euc2_ci95_high": U_euc2["ci95_high"],
        "gcorr_U_euc2_median": U_euc2["median"],
    }


def _load_model_bundle(extracts_dir: Path, name: str) -> Dict[str, Any]:
    info = load_info_json(extracts_dir, name)
    E, U, info = load_E_U_matrices(extracts_dir, name, info=info)
    emb_shape = info["standardized_dims"]["embed"]
    vocab_size, hidden_dim = int(emb_shape[0]), int(emb_shape[1])
    tie_decl = info.get("tie_word_embeddings")
    return {
        "E": E,
        "U": U,
        "info": info,
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "is_tied": tie_decl,
        "actual_tied": actual_tied(E, U),
    }


def _bool_from_csv(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _model_data_from_bootstrap_row(row: Dict[str, str], side: str) -> Dict[str, Any]:
    suffix = "a" if side == "a" else "b"
    return {
        "hidden_dim": int(row[f"hidden_dim_{suffix}"]),
        "vocab_size": int(row[f"vocab_size_{suffix}"]),
        "actual_tied": _bool_from_csv(row[f"actual_tied_{suffix}"]),
    }


def write_summary_from_bootstrap_csv(
    *,
    boot_csv: Path,
    summary_csv: Path,
    pairs: Sequence[Tuple[str, str]],
    n_bootstrap: int,
    n_pairs: int,
) -> int:
    """只基于 bootstrap_results.csv 重建 summary，不加载 tokenizer 或大矩阵。"""
    by_pair_csv = load_bootstrap_rows_by_pair(boot_csv)
    summary_rows = []
    keys_float = (
        "gcorr_E_cos",
        "gcorr_E_euc",
        "gcorr_E_euc2",
        "gcorr_U_cos",
        "gcorr_U_euc",
        "gcorr_U_euc2",
    )
    for model_a, model_b in pairs:
        sub = by_pair_csv.get((model_a, model_b), [])
        if len(sub) < n_bootstrap:
            print(f"  warn: {model_a}->{model_b} only {len(sub)}/{n_bootstrap} bootstraps")
            continue
        first = sub[0]
        bdicts = [{k: float(r[k]) for k in keys_float} for r in sub]
        summary_rows.append(
            _summarize_row(
                model_a,
                model_b,
                _model_data_from_bootstrap_row(first, "a"),
                _model_data_from_bootstrap_row(first, "b"),
                bdicts,
                int(first["n_tokens"]),
                str(first["align_mode"]),
                n_bootstrap,
                n_pairs,
            )
        )

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)
    return len(summary_rows)


def validate_completed_pairs_compute(
    *,
    extracts_dir: Path,
    pairs: Sequence[Tuple[str, str]],
    validation_n_tokens: int,
    validation_n_pairs: int,
    random_seed: int,
    devices: Sequence[torch.device],
) -> int:
    """
    在已有完整 bootstrap 结果时，仍逐 pair 加载 E/U 并跑一次小规模 GCorr。

    这个路径用于验证“当前代码确实能加载并计算这些矩阵”，但不重复写入
    bootstrap_results.csv，避免覆盖正式实验结果。
    """
    n_ok = 0
    n_devices = max(len(devices), 1)
    for idx, (model_a, model_b) in enumerate(pairs, start=1):
        device = devices[(idx - 1) % n_devices]
        print(
            f"[validate {idx}/{len(pairs)}] {model_a} -> {model_b} "
            f"on {device}",
            flush=True,
        )
        info_a = load_info_json(extracts_dir, model_a)
        info_b = load_info_json(extracts_dir, model_b)
        vocab_a = int(info_a["standardized_dims"]["embed"][0])
        vocab_b = int(info_b["standardized_dims"]["embed"][0])

        n_sample = min(validation_n_tokens, vocab_a, vocab_b)
        if n_sample < 2:
            raise ValueError(f"{model_a}->{model_b}: 可采样 token 数过少: {n_sample}")
        rng = np.random.default_rng(random_seed + idx)
        ids = rng.choice(min(vocab_a, vocab_b), size=n_sample, replace=False)
        E_a, U_a, _ = load_E_U_matrix_rows(extracts_dir, model_a, ids, info=info_a)
        E_b, U_b, _ = load_E_U_matrix_rows(extracts_dir, model_b, ids, info=info_b)

        result = compute_single_pair_bootstrap(
            E_a,
            E_b,
            U_a,
            U_b,
            validation_n_pairs,
            random_seed + idx,
            device,
        )
        print(
            f"  ok: E_euc={result['gcorr_E_euc']:.6f}, "
            f"U_euc={result['gcorr_U_euc']:.6f}",
            flush=True,
        )
        n_ok += 1

        del E_a, E_b, U_a, U_b, result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return n_ok


def try_git_commit(repo: Path) -> Optional[str]:
    try:
        import subprocess

        r = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


def write_metadata(
    *,
    meta_json: Path,
    repo_root: Path,
    pairs_file: Path,
    extracts_dir: Path,
    models_yaml: Path,
    pairs: Sequence[Tuple[str, str]],
    n_tokens: int,
    n_pairs: int,
    n_bootstrap: int,
    random_seed: int,
    devices: Sequence[torch.device],
    resume_mode: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    meta = {
        "pairs_file": rel_to_repo(pairs_file, repo_root),
        "n_pairs_config": len(pairs),
        "n_tokens": n_tokens,
        "n_pairs": n_pairs,
        "n_bootstrap": n_bootstrap,
        "seed": random_seed,
        "devices": [str(d) for d in devices],
        "extracts_dir": rel_to_repo(extracts_dir, repo_root),
        "models_yaml": rel_to_repo(models_yaml, repo_root),
        "resume_mode": resume_mode,
        "git_commit": try_git_commit(repo_root),
    }
    if extra_metadata:
        meta.update(extra_metadata)
    meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_task1_base_instruct(
    *,
    repo_root: Path,
    pairs_file: Path,
    extracts_dir: Path,
    models_yaml: Path,
    out_dir: Path,
    n_tokens: int,
    n_pairs: int,
    n_bootstrap: int,
    random_seed: int,
    devices: Sequence[torch.device],
    complete_mode: str = "validate",
    validation_n_tokens: int = 1024,
    validation_n_pairs: int = 10000,
) -> None:
    pairs = load_pairs_yaml(pairs_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    boot_csv = out_dir / "bootstrap_results.csv"
    summary_csv = out_dir / "summary.csv"
    meta_json = out_dir / "metadata.json"

    done = read_existing_bootstrap_rows(boot_csv)
    expected_done = {(model_a, model_b, b) for model_a, model_b in pairs for b in range(n_bootstrap)}
    if expected_done and expected_done.issubset(done):
        print(f"all bootstraps already complete: {len(done)}/{len(expected_done)}")
        if complete_mode == "csv-only":
            print("complete_mode=csv-only: rebuilding summary from CSV only", flush=True)
            n_validated = None
            resume_mode = "csv_only_complete"
        elif complete_mode == "validate":
            print(
                "complete_mode=validate: loading matrices and running small GCorr "
                f"({validation_n_tokens} tokens, {validation_n_pairs} pairs)",
                flush=True,
            )
            n_validated = validate_completed_pairs_compute(
                extracts_dir=extracts_dir,
                pairs=pairs,
                validation_n_tokens=validation_n_tokens,
                validation_n_pairs=validation_n_pairs,
                random_seed=random_seed,
                devices=devices,
            )
            resume_mode = "validate_complete"
        else:
            raise ValueError(f"未知 complete_mode: {complete_mode}")

        n_summary = write_summary_from_bootstrap_csv(
            boot_csv=boot_csv,
            summary_csv=summary_csv,
            pairs=pairs,
            n_bootstrap=n_bootstrap,
            n_pairs=n_pairs,
        )
        write_metadata(
            meta_json=meta_json,
            repo_root=repo_root,
            pairs_file=pairs_file,
            extracts_dir=extracts_dir,
            models_yaml=models_yaml,
            pairs=pairs,
            n_tokens=n_tokens,
            n_pairs=n_pairs,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
            devices=devices,
            resume_mode=resume_mode,
            extra_metadata={
                "complete_mode": complete_mode,
                "validation_n_tokens": validation_n_tokens,
                "validation_n_pairs": validation_n_pairs,
                "validated_pairs": n_validated,
            },
        )
        print(f"summary rows: {n_summary}")
        print(f"done: {out_dir}")
        return

    sampler = TokenSampler()
    n_gpu = max(len(devices), 1)
    file_lock = threading.Lock()

    def append_bootstrap_row(row: Dict[str, Any]) -> None:
        with file_lock:
            write_header = not boot_csv.exists()
            with boot_csv.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=BOOTSTRAP_CSV_FIELDS, extrasaction="ignore")
                if write_header:
                    w.writeheader()
                w.writerow(row)

    def get_tok(model_name: str):
        d = find_model_cache_dir(repo_root, model_name, models_yaml)
        if d is None:
            return None
        return AutoTokenizer.from_pretrained(str(d), trust_remote_code=True)

    def _vocab_from_info(info: Dict[str, Any]) -> int:
        return int(info["standardized_dims"]["embed"][0])

    for idx, (model_a, model_b) in enumerate(pairs, start=1):
        missing_bootstraps = [b for b in range(n_bootstrap) if (model_a, model_b, b) not in done]
        print(
            f"[{idx}/{len(pairs)}] {model_a} -> {model_b} "
            f"missing={len(missing_bootstraps)}/{n_bootstrap}",
            flush=True,
        )
        if not missing_bootstraps:
            continue

        ia = load_info_json(extracts_dir, model_a)
        ib = load_info_json(extracts_dir, model_b)
        info = build_pair_token_info(
            vocab_a=_vocab_from_info(ia),
            vocab_b=_vocab_from_info(ib),
            tokenizer_a=get_tok(model_a),
            tokenizer_b=get_tok(model_b),
            n_tokens=n_tokens,
            sampler=sampler,
        )
        if info is None:
            print("  skip: common tokens < 1000", flush=True)
            continue

        data_a = _load_model_bundle(extracts_dir, model_a)
        data_b = _load_model_bundle(extracts_dir, model_b)

        def run_one_bootstrap(task: Tuple[int, int]) -> Tuple[int, Dict[str, Any]]:
            b, gpu_slot = task
            device = devices[gpu_slot % n_gpu]
            seed = random_seed + b
            n_sample = info["n_sample"]
            rng = np.random.default_rng(seed)

            if info["same_tokenizer"]:
                sample_ids = sampler.sample_tokens(info["ids_a"], n_sample, seed)
                sa = sb = sample_ids
            else:
                indices = rng.choice(len(info["ids_a"]), size=n_sample, replace=False)
                sa = info["ids_a"][indices]
                sb = info["ids_b"][indices]

            result = compute_single_pair_bootstrap(
                data_a["E"][sa],
                data_b["E"][sb],
                data_a["U"][sa],
                data_b["U"][sb],
                n_pairs,
                seed,
                device,
            )
            row = {
                "model_a": model_a,
                "model_b": model_b,
                "bootstrap": b,
                "align_mode": info["align_mode"],
                "n_tokens": n_sample,
                "n_pairs": n_pairs,
                "hidden_dim_a": data_a["hidden_dim"],
                "hidden_dim_b": data_b["hidden_dim"],
                "vocab_size_a": data_a["vocab_size"],
                "vocab_size_b": data_b["vocab_size"],
                "actual_tied_a": data_a["actual_tied"],
                "actual_tied_b": data_b["actual_tied"],
                "gcorr_E_cos": result["gcorr_E_cos"],
                "gcorr_E_euc": result["gcorr_E_euc"],
                "gcorr_E_euc2": result["gcorr_E_euc2"],
                "gcorr_U_cos": result["gcorr_U_cos"],
                "gcorr_U_euc": result["gcorr_U_euc"],
                "gcorr_U_euc2": result["gcorr_U_euc2"],
            }
            return b, row

        tasks = [(b, i % n_gpu) for i, b in enumerate(missing_bootstraps)]
        print(
            f"  running {len(tasks)} bootstrap(s) on {len(devices)} device(s): "
            f"{', '.join(str(d) for d in devices)}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=min(len(tasks), n_gpu)) as ex:
            futures = [ex.submit(run_one_bootstrap, task) for task in tasks]
            for future in as_completed(futures):
                b, row = future.result()
                append_bootstrap_row(row)
                done.add((model_a, model_b, b))
                print(
                    f"  bootstrap {b + 1}/{n_bootstrap}: "
                    f"E_euc={row['gcorr_E_euc']:.4f}, U_euc={row['gcorr_U_euc']:.4f}",
                    flush=True,
                )

        del data_a, data_b
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n_summary = write_summary_from_bootstrap_csv(
        boot_csv=boot_csv,
        summary_csv=summary_csv,
        pairs=pairs,
        n_bootstrap=n_bootstrap,
        n_pairs=n_pairs,
    )
    print(f"summary rows: {n_summary}", flush=True)

    write_metadata(
        meta_json=meta_json,
        repo_root=repo_root,
        pairs_file=pairs_file,
        extracts_dir=extracts_dir,
        models_yaml=models_yaml,
        pairs=pairs,
        n_tokens=n_tokens,
        n_pairs=n_pairs,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
        devices=devices,
        resume_mode="computed_missing_then_csv_summary",
    )
    print(f"done: {out_dir}")
