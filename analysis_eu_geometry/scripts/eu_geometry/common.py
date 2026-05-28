"""Shared paths, I/O, model catalog, and matrix statistics."""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent.parent


# --- paths ---


def find_repo_root(start: Path) -> Path:
    for directory in [start, *start.parents]:
        if (directory / "configs" / "models.yaml").is_file():
            return directory
    raise RuntimeError(
        "找不到仓库根目录（未在上级路径发现 configs/models.yaml）。"
        "请在 get_useful 仓库根目录下运行，或设置环境变量 REPO_ROOT。"
    )


def bootstrap_repo() -> Path:
    repo_root = (
        Path(os.environ["REPO_ROOT"]).resolve()
        if os.environ.get("REPO_ROOT")
        else find_repo_root(SCRIPT_DIR)
    )
    src = repo_root / "ijcai_clean" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    return repo_root


def analysis_root(repo: Path) -> Path:
    return repo / "analysis_eu_geometry"


def results_dir(repo: Path) -> Path:
    return analysis_root(repo) / "results"


def results_file(repo: Path, name: str) -> Path:
    return results_dir(repo) / name


def layers_dir(repo: Path) -> Path:
    return results_dir(repo) / "layers"


def layers_file(repo: Path, name: str) -> Path:
    return layers_dir(repo) / name


# Result filenames
F_BI_ROW_NORMS = "bi_row_norms.csv"
F_BI_BY_PAIR = "bi_pair_delta.csv"
F_BI_SUMMARY = "BI_ROW_NORMS_SUMMARY.md"
F_OTHER_ROW_NORMS = "other_models_row_norms.csv"
F_LAYER1_ROW_NORMS = "layer1_row_norms.csv"
F_LAYER2_MU_RATIO = "layer2_mu_ratio.csv"
F_LAYER3_SPECTRAL = "layer3_spectral.csv"
F_EU_FEATURES = "all_models_eu_features.csv"
F_EU_FEATURES_SUMMARY = "ALL_MODELS_EU_FEATURES_SUMMARY.md"


# --- matrix stats ---


def norm_stats(M: np.ndarray) -> tuple[float, float, float, float]:
    norms = np.linalg.norm(M, axis=1)
    return float(norms.mean()), float(norms.var()), float(norms.min()), float(norms.max())


def mean_vector_stats(M: np.ndarray) -> tuple[float, float, float]:
    """Return mean row norm, ||μ||, and ||μ|| / mean(row norm)."""
    row_norms = np.linalg.norm(M, axis=1)
    mean_row_norm = float(row_norms.mean())
    mean_vec_norm = float(np.linalg.norm(M.mean(axis=0)))
    ratio = mean_vec_norm / mean_row_norm if mean_row_norm else float("nan")
    return mean_row_norm, mean_vec_norm, ratio


# --- CSV I/O and formatting ---


def write_long_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_long_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def fmt_mean(x: float) -> str:
    return f"{x:.6f}"


def fmt_num(x: float) -> str:
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax < 1e-4:
        return f"{x:.2e}"
    if ax >= 100:
        return f"{x:.2f}"
    return f"{x:.4f}"


def fmt_stat(mean: float, var: float, vmin: float, vmax: float) -> str:
    return f"{fmt_mean(mean)} (var={fmt_num(var)}) [{fmt_num(vmin)}, {fmt_num(vmax)}]"


def delta_pct(base_mean: float, inst_mean: float) -> float:
    return (inst_mean - base_mean) / base_mean * 100.0


def fmt_delta_pct(value: float) -> str:
    return f"{value:+.2f}%"


def build_model_tables(rows: list[dict[str, object]]) -> tuple[dict, dict]:
    by_model: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    meta: dict[str, dict[str, object]] = {}
    for row in rows:
        name = str(row["model"])
        mat = str(row["matrix"])
        by_model[name][mat] = {
            "mean": float(row["mean"]),
            "var": float(row["var"]),
            "min": float(row["min"]),
            "max": float(row["max"]),
        }
        meta[name] = {k: row[k] for k in row if k not in {"matrix", "mean", "var", "min", "max"}}
    return by_model, meta


def model_stat(by_model: dict, name: str, mat: str) -> str:
    d = by_model[name][mat]
    return fmt_stat(d["mean"], d["var"], d["min"], d["max"])


def u_col(by_model: dict, meta: dict, name: str) -> str:
    if meta[name]["tied"]:
        return "(=E)"
    return model_stat(by_model, name, "U")


def write_model_summary_csv(
    path: Path,
    by_model: dict,
    meta: dict,
    extra_columns: list[str],
) -> None:
    fields = ["model", *extra_columns, "tied", "E_mean_var_range", "U_mean_var_range"]
    model_rows = []
    for name in sorted(by_model):
        row = {k: meta[name].get(k, "") for k in extra_columns}
        row.update(
            {
                "model": name,
                "tied": meta[name]["tied"],
                "E_mean_var_range": model_stat(by_model, name, "E"),
                "U_mean_var_range": model_stat(by_model, name, "U")
                if not meta[name]["tied"]
                else "(=E)",
            }
        )
        model_rows.append(row)
    write_long_csv(path, model_rows, fields)


# --- model catalog ---


def load_model_groups(models_yaml: Path) -> dict[str, str]:
    cfg = yaml.safe_load(models_yaml.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for group, names in (cfg.get("model_groups") or {}).items():
        for name in names or []:
            out[str(name)] = str(group)
    return out


def is_instruct(name: str) -> bool:
    return "Instruct" in name or name.endswith("-it")


def bi_models(pairs_file: Path) -> tuple[list[tuple[str, str]], list[str]]:
    from ijcai_clean.experiments.gcorr_io import load_pairs_yaml

    pairs = load_pairs_yaml(pairs_file)
    models = sorted({m for pair in pairs for m in pair})
    return pairs, models


def other_models(models_yaml: Path, pairs_file: Path) -> list[str]:
    from ijcai_clean.experiments.gcorr_io import load_pairs_yaml

    cfg = yaml.safe_load(models_yaml.read_text(encoding="utf-8"))
    all_models = sorted((cfg.get("model_repo_ids") or {}).keys())
    bi = {m for pair in load_pairs_yaml(pairs_file) for m in pair}
    return [m for m in all_models if m not in bi]


def all_models_from_yaml(models_yaml: Path) -> list[str]:
    cfg = yaml.safe_load(models_yaml.read_text(encoding="utf-8"))
    return sorted((cfg.get("model_repo_ids") or {}).keys())


def catalog_context(
    models_yaml: Path,
    pairs_file: Path,
) -> tuple[dict[str, str], set[str], list[str]]:
    model_groups = load_model_groups(models_yaml)
    _, bi_models_list = bi_models(pairs_file)
    return model_groups, set(bi_models_list), all_models_from_yaml(models_yaml)


def row_metadata(
    name: str,
    *,
    bi_set: set[str],
    model_groups: dict[str, str],
) -> dict[str, object]:
    return {
        "subset": "base_instruct" if name in bi_set else "other_models",
        "model_group": model_groups.get(name, ""),
        "role": (
            "instruct"
            if name in bi_set and is_instruct(name)
            else ("base" if name in bi_set else "")
        ),
    }
