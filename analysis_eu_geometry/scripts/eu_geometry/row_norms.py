"""Layer 1: row-norm audit (mean, var, min, max) and BI pair summaries."""
from __future__ import annotations

from pathlib import Path

from .common import (
    F_BI_BY_PAIR,
    F_BI_ROW_NORMS,
    F_BI_SUMMARY,
    F_LAYER1_ROW_NORMS,
    F_OTHER_ROW_NORMS,
    as_bool,
    bi_models,
    build_model_tables,
    delta_pct,
    fmt_delta_pct,
    fmt_stat,
    is_instruct,
    load_model_groups,
    norm_stats,
    other_models,
    read_long_csv,
    results_dir,
    results_file,
    layers_file,
    write_long_csv,
)

MERGED_FIELDS = [
    "subset",
    "model",
    "model_group",
    "matrix",
    "mean",
    "var",
    "min",
    "max",
    "role",
    "tied",
    "vocab_size",
    "hidden_dim",
]

ROW_NORM_LONG_FIELDS = [
    "model",
    "matrix",
    "mean",
    "var",
    "min",
    "max",
    "role",
    "tied",
    "vocab_size",
    "hidden_dim",
]

OTHER_ROW_NORM_FIELDS = [
    "model",
    "model_group",
    "matrix",
    "mean",
    "var",
    "min",
    "max",
    "tied",
    "vocab_size",
    "hidden_dim",
]


def audit_models(
    *,
    extracts_dir: Path,
    models: list[str],
    extra_fields_fn=None,
) -> list[dict[str, object]]:
    from ijcai_clean.data import load_E_U_matrices, load_info_json

    rows: list[dict[str, object]] = []
    for name in models:
        info = load_info_json(extracts_dir, name)
        E, U, info = load_E_U_matrices(extracts_dir, name, info=info)
        emb_shape = info["standardized_dims"]["embed"]
        vocab_size, hidden_dim = int(emb_shape[0]), int(emb_shape[1])
        tied = bool(info.get("tie_word_embeddings"))
        base_row = {
            "model": name,
            "tied": tied,
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
        }
        if extra_fields_fn is not None:
            base_row.update(extra_fields_fn(name))

        for matrix_name, M in (("E", E), ("U", U)):
            mean, var, vmin, vmax = norm_stats(M)
            rows.append(
                {
                    **base_row,
                    "matrix": matrix_name,
                    "mean": mean,
                    "var": var,
                    "min": vmin,
                    "max": vmax,
                }
            )
            print(
                f"{name:40} {matrix_name} mean={mean:.6f} var={var:.6e} "
                f"range=[{vmin:.6f}, {vmax:.6f}] tied={tied}",
                flush=True,
            )
    return rows


def bi_pair_deltas(
    by_model: dict[str, dict[str, dict[str, float]]],
    meta: dict[str, dict[str, object]],
    base: str,
    instr: str,
) -> tuple[float, float]:
    dE = delta_pct(by_model[base]["E"]["mean"], by_model[instr]["E"]["mean"])
    if meta[base]["tied"]:
        return dE, dE
    dU = delta_pct(by_model[base]["U"]["mean"], by_model[instr]["U"]["mean"])
    return dE, dU


def write_bi_summaries(
    repo: Path,
    *,
    pairs: list[tuple[str, str]],
    rows: list[dict[str, object]],
) -> None:
    by_model, meta = build_model_tables(rows)
    for name in meta:
        meta[name]["tied"] = as_bool(meta[name]["tied"])
    write_bi_summaries_from_tables(repo=repo, pairs=pairs, by_model=by_model, meta=meta)


def write_bi_summaries_from_tables(
    *,
    repo: Path,
    pairs: list[tuple[str, str]],
    by_model: dict[str, dict[str, dict[str, float]]],
    meta: dict[str, dict[str, object]],
) -> None:
    out_dir = results_dir(repo)

    def stat(name: str, mat: str) -> str:
        d = by_model[name][mat]
        return fmt_stat(d["mean"], d["var"], d["min"], d["max"])

    def u(name: str) -> str:
        return "(=E)" if meta[name]["tied"] else stat(name, "U")

    pair_rows = []
    for base, instr in pairs:
        dE, dU = bi_pair_deltas(by_model, meta, base, instr)
        pair_rows.append(
            {
                "base_model": base,
                "instruct_model": instr,
                "base_tied": meta[base]["tied"],
                "base_E": stat(base, "E"),
                "base_U": u(base),
                "instruct_E": stat(instr, "E"),
                "instruct_U": u(instr),
                "delta_E_pct": f"{dE:.4f}",
                "delta_U_pct": f"{dU:.4f}",
            }
        )
    write_long_csv(
        out_dir / F_BI_BY_PAIR,
        pair_rows,
        [
            "base_model",
            "instruct_model",
            "base_tied",
            "base_E",
            "base_U",
            "instruct_E",
            "instruct_U",
            "delta_E_pct",
            "delta_U_pct",
        ],
    )

    sections = [
        ("Qwen3", lambda b: b.startswith("Qwen3-") and not b.startswith("Qwen3.5")),
        ("Qwen3.5", lambda b: b.startswith("Qwen3.5")),
        ("Qwen2.5", lambda b: b.startswith("Qwen2.5")),
        ("Llama", lambda b: b.startswith("Llama")),
        ("Gemma2", lambda b: b.startswith("Gemma-2")),
        ("Gemma3", lambda b: b.startswith("Gemma-3")),
        ("Gemma4", lambda b: b.startswith("Gemma-4")),
        ("DeepSeek", lambda b: b.startswith("DeepSeek")),
    ]
    md_lines = [
        "# Base-Instruct 行范数汇总",
        "",
        "格式：`均值 (var=方差) [min, max]`。tied 模型 E=U，U 列写 `(=E)`。",
        "",
        f"原始长表：`{F_BI_ROW_NORMS}`",
        f"配对 Δ：`{F_BI_BY_PAIR}`（含 `delta_E_pct` / `delta_U_pct`）",
        "",
    ]
    for sec_name, pred in sections:
        sec_pairs = [(b, i) for b, i in pairs if pred(b)]
        if not sec_pairs:
            continue
        md_lines.extend(
            [
                f"## {sec_name}",
                "",
                "| Pair | Base E | Base U | Instruct E | Instruct U |",
                "|------|--------|--------|------------|------------|",
            ]
        )
        for base, instr in sec_pairs:
            md_lines.append(
                f"| {base} / {instr} | {stat(base, 'E')} | {u(base)} | "
                f"{stat(instr, 'E')} | {u(instr)} |"
            )
        md_lines.append("")

    untied_pairs = [(b, i) for b, i in pairs if not meta[b]["tied"]]
    md_lines.extend(
        [
            "## Untied BI：行范数 mean 漂移（Δ%）",
            "",
            "Δ% = `(Instruct mean − Base mean) / Base mean × 100`。",
            "",
            "| Pair | ΔE | ΔU |",
            "|------|-----|-----|",
        ]
    )
    for base, instr in untied_pairs:
        dE, dU = bi_pair_deltas(by_model, meta, base, instr)
        md_lines.append(
            f"| {base} / {instr} | {fmt_delta_pct(dE)} | {fmt_delta_pct(dU)} |"
        )
    md_lines.append("")

    tied_pairs = [(b, i) for b, i in pairs if meta[b]["tied"]]
    md_lines.extend(
        [
            "## Tied BI：行范数 mean 漂移（Δ%，E=U）",
            "",
            "| Pair | ΔE (=ΔU) |",
            "|------|-----------|",
        ]
    )
    for base, instr in tied_pairs:
        dE, _ = bi_pair_deltas(by_model, meta, base, instr)
        md_lines.append(f"| {base} / {instr} | {fmt_delta_pct(dE)} |")
    md_lines.append("")

    (out_dir / F_BI_SUMMARY).write_text("\n".join(md_lines), encoding="utf-8")


def refresh_base_instruct_summaries(
    repo: Path,
    *,
    pairs_file: Path | None = None,
) -> None:
    pairs_file = pairs_file or (repo / "configs" / "base_instruct_pairs.yaml")
    long_csv = results_file(repo, F_BI_ROW_NORMS)
    pairs, _ = bi_models(pairs_file)
    rows_raw = read_long_csv(long_csv)
    by_model, meta = build_model_tables(rows_raw)
    for name in meta:
        meta[name]["tied"] = as_bool(meta[name]["tied"])
    write_bi_summaries_from_tables(repo=repo, pairs=pairs, by_model=by_model, meta=meta)


def run_base_instruct_audit(
    repo: Path,
    *,
    pairs_file: Path,
    extracts_dir: Path,
) -> Path:
    out_dir = results_dir(repo)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs, models = bi_models(pairs_file)
    rows = audit_models(
        extracts_dir=extracts_dir,
        models=models,
        extra_fields_fn=lambda name: {"role": "instruct" if is_instruct(name) else "base"},
    )
    out_csv = out_dir / F_BI_ROW_NORMS
    write_long_csv(out_csv, rows, ROW_NORM_LONG_FIELDS)
    write_bi_summaries(repo, pairs=pairs, rows=rows)
    print(f"\nWrote {len(rows)} rows ({len(models)} models) -> {out_csv}", flush=True)
    return out_csv


def run_other_models_audit(
    repo: Path,
    *,
    models_yaml: Path,
    pairs_file: Path,
    extracts_dir: Path,
) -> Path:
    out_dir = results_dir(repo)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = other_models(models_yaml, pairs_file)
    model_groups = load_model_groups(models_yaml)
    rows = audit_models(
        extracts_dir=extracts_dir,
        models=models,
        extra_fields_fn=lambda name: {"model_group": model_groups.get(name, "other")},
    )
    out_csv = out_dir / F_OTHER_ROW_NORMS
    write_long_csv(out_csv, rows, OTHER_ROW_NORM_FIELDS)
    print(f"\nWrote {len(rows)} rows ({len(models)} models) -> {out_csv}", flush=True)
    return out_csv


def merge_all_models_row_norms(
    repo: Path,
    *,
    models_yaml: Path | None = None,
    pairs_file: Path | None = None,
) -> Path:
    models_yaml = models_yaml or (repo / "configs" / "models.yaml")
    model_groups = load_model_groups(models_yaml)

    bi_csv = results_file(repo, F_BI_ROW_NORMS)
    other_csv = results_file(repo, F_OTHER_ROW_NORMS)
    out_csv = layers_file(repo, F_LAYER1_ROW_NORMS)

    merged: list[dict[str, object]] = []
    for subset, path in (("base_instruct", bi_csv), ("other_models", other_csv)):
        for row in read_long_csv(path):
            merged.append(
                {
                    "subset": subset,
                    "model": row["model"],
                    "model_group": row.get("model_group") or model_groups.get(row["model"], ""),
                    "matrix": row["matrix"],
                    "mean": row["mean"],
                    "var": row["var"],
                    "min": row["min"],
                    "max": row["max"],
                    "role": row.get("role", ""),
                    "tied": row["tied"],
                    "vocab_size": row["vocab_size"],
                    "hidden_dim": row["hidden_dim"],
                }
            )

    write_long_csv(out_csv, merged, MERGED_FIELDS)
    refresh_base_instruct_summaries(repo, pairs_file=pairs_file)

    from .features import merge_all_models_features

    merge_all_models_features(repo)
    return out_csv
