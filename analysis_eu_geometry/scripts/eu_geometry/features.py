"""Merge layer-1/2/3 CSVs into one per (model, matrix) feature table."""
from __future__ import annotations

from pathlib import Path

from .common import (
    F_EU_FEATURES,
    F_EU_FEATURES_SUMMARY,
    F_LAYER1_ROW_NORMS,
    F_LAYER2_MU_RATIO,
    F_LAYER3_SPECTRAL,
    as_bool,
    read_long_csv,
    results_dir,
    results_file,
    layers_file,
    write_long_csv,
)

ALL_MODELS_FEATURES_FIELDS = [
    "subset",
    "model",
    "model_group",
    "role",
    "matrix",
    "tied",
    "vocab_size",
    "hidden_dim",
    "mean",
    "var",
    "min",
    "max",
    "mean_vec_norm",
    "mu_over_row_norm",
    "sigma1",
    "sigma1_centered",
    "sigma1_over_mean_row",
    "sigma1_c_over_mean_row",
    "sigma1_over_mean_sqrt_n",
    "sigma1_c_over_mean_sqrt_n",
    "cos_v1_mu",
    "cos_v1_v1_centered",
    "rank1_energy_frac",
    "rank5_energy_frac",
    "rank10_energy_frac",
    "participation_ratio",
    "effective_rank",
    "isotropy_pr_over_d",
    "sigma_ratio",
    "rank1_centered_energy_frac",
    "rank5_centered_energy_frac",
    "participation_ratio_centered",
    "effective_rank_centered",
    "isotropy_pr_over_d_centered",
    "sigma_ratio_centered",
]

_MU_RATIO_COLS = ("mean_vec_norm", "mu_over_row_norm")
_SPECTRAL_COLS = ALL_MODELS_FEATURES_FIELDS[14:]


def _key(row: dict[str, str]) -> tuple[str, str]:
    return str(row["model"]), str(row["matrix"])


def _index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {_key(row): row for row in rows}


def _copy_cols(target: dict[str, object], source: dict[str, str], cols: tuple[str, ...]) -> None:
    for col in cols:
        if col in source and source[col] != "":
            target[col] = source[col]


def merge_all_models_features(repo: Path) -> Path | None:
    """Join layer1 + layer2 + layer3 on (model, matrix)."""
    layer1 = layers_file(repo, F_LAYER1_ROW_NORMS)
    if not layer1.is_file():
        return None

    mu_csv = layers_file(repo, F_LAYER2_MU_RATIO)
    spec_csv = layers_file(repo, F_LAYER3_SPECTRAL)

    row_rows = read_long_csv(layer1)
    mu_by_key = _index(read_long_csv(mu_csv)) if mu_csv.is_file() else {}
    spec_by_key = _index(read_long_csv(spec_csv)) if spec_csv.is_file() else {}

    merged: list[dict[str, object]] = []
    for base in row_rows:
        model, matrix = _key(base)
        tied = as_bool(base.get("tied"))
        row: dict[str, object] = {
            "subset": base.get("subset", ""),
            "model": model,
            "model_group": base.get("model_group", ""),
            "role": base.get("role", ""),
            "matrix": matrix,
            "tied": tied,
            "vocab_size": base.get("vocab_size", ""),
            "hidden_dim": base.get("hidden_dim", ""),
            "mean": base.get("mean", ""),
            "var": base.get("var", ""),
            "min": base.get("min", ""),
            "max": base.get("max", ""),
        }

        mu_src = mu_by_key.get((model, matrix))
        if mu_src is None and tied and matrix == "U":
            mu_src = mu_by_key.get((model, "E"))
        if mu_src is not None:
            _copy_cols(row, mu_src, _MU_RATIO_COLS)

        spec_src = spec_by_key.get((model, matrix))
        if spec_src is None and tied and matrix == "U":
            spec_src = spec_by_key.get((model, "E"))
        if spec_src is not None:
            _copy_cols(row, spec_src, _SPECTRAL_COLS)

        merged.append(row)

    out_csv = results_file(repo, F_EU_FEATURES)
    write_long_csv(out_csv, merged, ALL_MODELS_FEATURES_FIELDS)

    md_lines = [
        "# 全库 E/U 特征总表",
        "",
        "每行 = 一个 `(model, matrix)`：`layers/layer1_row_norms` + `layers/layer2_mu_ratio` + `layers/layer3_spectral`。",
        "tied 模型的 U 行若无 μ/谱条目，则沿用同模型 E 行。",
        "",
        f"原始长表：`{F_EU_FEATURES}`（{len(merged)} 行）",
        "",
        "| Model | M | mean | μ/mean | rank1 | rank1(c) | PR/d | PR/d(c) | tied |",
        "|-------|---|------|--------|-------|----------|------|---------|------|",
    ]
    for row in merged:
        def _fmt(v: object, nd: int = 3) -> str:
            if v == "" or v is None:
                return ""
            try:
                return f"{float(v):.{nd}f}"
            except (TypeError, ValueError):
                return str(v)

        md_lines.append(
            f"| {row['model']} | {row['matrix']} | {_fmt(row.get('mean'), 4)} | "
            f"{_fmt(row.get('mu_over_row_norm'))} | {_fmt(row.get('rank1_energy_frac'))} | "
            f"{_fmt(row.get('rank1_centered_energy_frac'))} | {_fmt(row.get('isotropy_pr_over_d'))} | "
            f"{_fmt(row.get('isotropy_pr_over_d_centered'))} | {row['tied']} |"
        )
    md_lines.append("")
    (results_dir(repo) / F_EU_FEATURES_SUMMARY).write_text("\n".join(md_lines), encoding="utf-8")

    return out_csv
