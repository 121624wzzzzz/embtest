from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import yaml

from cross_model_geometry.paths import rel_to_repo

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
