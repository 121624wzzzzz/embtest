"""Shared helpers for E/U row-norm audit."""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


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
    return repo_root


def analysis_root(repo: Path) -> Path:
    return repo / "analysis_eu_geometry"


def row_norms_dir(repo: Path, *parts: str) -> Path:
    return analysis_root(repo) / "results" / "row_norms" / Path(*parts)


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


def write_long_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_long_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
    *,
    out_dir: Path,
    pairs: list[tuple[str, str]],
    rows: list[dict[str, object]],
) -> None:
    by_model, meta = build_model_tables(rows)
    for name in meta:
        meta[name]["tied"] = _as_bool(meta[name]["tied"])
    write_bi_summaries_from_tables(out_dir=out_dir, pairs=pairs, by_model=by_model, meta=meta)


def write_bi_summaries_from_tables(
    *,
    out_dir: Path,
    pairs: list[tuple[str, str]],
    by_model: dict[str, dict[str, dict[str, float]]],
    meta: dict[str, dict[str, object]],
) -> None:
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
        out_dir / "base_instruct_row_norms_by_pair.csv",
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

    model_rows = [
        {
            "model": name,
            "role": meta[name]["role"],
            "tied": meta[name]["tied"],
            "E_mean_var_range": stat(name, "E"),
            "U_mean_var_range": stat(name, "U") if not meta[name]["tied"] else "(=E)",
        }
        for name in sorted(by_model)
    ]
    write_long_csv(
        out_dir / "base_instruct_row_norms_by_model.csv",
        model_rows,
        ["model", "role", "tied", "E_mean_var_range", "U_mean_var_range"],
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
        "原始长表：`base_instruct_row_norms.csv`",
        "配对 Δ：`base_instruct_row_norms_by_pair.csv`（含 `delta_E_pct` / `delta_U_pct`）",
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
            "untied 模型 E≠U，故 **E、U 分别计算**；tied 对见 CSV（ΔU=ΔE）。",
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

    (out_dir / "BASE_INSTRUCT_ROW_NORMS_SUMMARY.md").write_text("\n".join(md_lines), encoding="utf-8")


def refresh_base_instruct_summaries(
    repo: Path,
    *,
    pairs_file: Path | None = None,
) -> None:
    pairs_file = pairs_file or (repo / "configs" / "base_instruct_pairs.yaml")
    out_dir = row_norms_dir(repo, "base_instruct")
    long_csv = out_dir / "base_instruct_row_norms.csv"
    pairs, _ = bi_models(pairs_file)
    rows_raw = read_long_csv(long_csv)
    by_model, meta = build_model_tables(rows_raw)
    for name in meta:
        meta[name]["tied"] = _as_bool(meta[name]["tied"])
    write_bi_summaries_from_tables(out_dir=out_dir, pairs=pairs, by_model=by_model, meta=meta)


def write_other_summaries(
    *,
    out_dir: Path,
    by_model: dict,
    meta: dict,
    model_groups: dict[str, str],
    models: list[str],
) -> None:
    group_order: list[str] = []
    grouped: dict[str, list[str]] = {}
    for name in models:
        group = model_groups.get(name, "other")
        grouped.setdefault(group, []).append(name)
        if group not in group_order:
            group_order.append(group)

    md_lines = [
        "# 非 Base-Instruct 模型行范数汇总",
        "",
        "来源：`configs/models.yaml` 全体模型减去 `configs/base_instruct_pairs.yaml` 中的 70 个 BI 模型。",
        "",
        "格式：`均值 (var=方差) [min, max]`。tied 模型 E=U，U 列写 `(=E)`。",
        "",
        "原始长表：`other_models_row_norms.csv`",
        "",
    ]
    for group in group_order:
        names = grouped[group]
        if not names:
            continue
        md_lines.extend(
            [
                f"## {group}",
                "",
                "| Model | E | U | tied |",
                "|-------|---|---|------|",
            ]
        )
        for name in names:
            md_lines.append(
                f"| {name} | {model_stat(by_model, name, 'E')} | "
                f"{u_col(by_model, meta, name)} | {meta[name]['tied']} |"
            )
        md_lines.append("")
    (out_dir / "OTHER_MODELS_ROW_NORMS_SUMMARY.md").write_text("\n".join(md_lines), encoding="utf-8")


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


def run_base_instruct_audit(
    repo: Path,
    *,
    pairs_file: Path,
    extracts_dir: Path,
    out_dir: Path | None = None,
) -> Path:
    out_dir = out_dir or row_norms_dir(repo, "base_instruct")
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs, models = bi_models(pairs_file)
    rows = audit_models(
        extracts_dir=extracts_dir,
        models=models,
        extra_fields_fn=lambda name: {"role": "instruct" if is_instruct(name) else "base"},
    )
    out_csv = out_dir / "base_instruct_row_norms.csv"
    write_long_csv(
        out_csv,
        rows,
        [
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
        ],
    )
    write_bi_summaries(out_dir=out_dir, pairs=pairs, rows=rows)
    print(f"\nWrote {len(rows)} rows ({len(models)} models) -> {out_csv}", flush=True)
    return out_csv


def run_other_models_audit(
    repo: Path,
    *,
    models_yaml: Path,
    pairs_file: Path,
    extracts_dir: Path,
    out_dir: Path | None = None,
) -> Path:
    out_dir = out_dir or row_norms_dir(repo, "other_models")
    out_dir.mkdir(parents=True, exist_ok=True)

    models = other_models(models_yaml, pairs_file)
    model_groups = load_model_groups(models_yaml)
    rows = audit_models(
        extracts_dir=extracts_dir,
        models=models,
        extra_fields_fn=lambda name: {"model_group": model_groups.get(name, "other")},
    )
    out_csv = out_dir / "other_models_row_norms.csv"
    write_long_csv(
        out_csv,
        rows,
        [
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
        ],
    )
    by_model, meta = build_model_tables(rows)
    write_model_summary_csv(
        out_dir / "other_models_row_norms_by_model.csv",
        by_model,
        meta,
        extra_columns=["model_group"],
    )
    write_other_summaries(
        out_dir=out_dir,
        by_model=by_model,
        meta=meta,
        model_groups=model_groups,
        models=models,
    )
    print(f"\nWrote {len(rows)} rows ({len(models)} models) -> {out_csv}", flush=True)
    return out_csv


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def write_all_models_summary(
    repo: Path,
    *,
    models_yaml: Path | None = None,
    pairs_file: Path | None = None,
) -> None:
    models_yaml = models_yaml or (repo / "configs" / "models.yaml")
    pairs_file = pairs_file or (repo / "configs" / "base_instruct_pairs.yaml")
    model_groups = load_model_groups(models_yaml)

    bi_rows_raw = read_long_csv(row_norms_dir(repo, "base_instruct", "base_instruct_row_norms.csv"))
    other_rows_raw = read_long_csv(row_norms_dir(repo, "other_models", "other_models_row_norms.csv"))

    bi_by_model, bi_meta = build_model_tables(bi_rows_raw)
    other_by_model, other_meta = build_model_tables(other_rows_raw)
    for meta in (bi_meta, other_meta):
        for name in meta:
            meta[name]["tied"] = _as_bool(meta[name]["tied"])

    pairs, _ = bi_models(pairs_file)
    other_model_names = sorted(other_by_model)

    md_lines = [
        "# 全库 94 模型行范数总表",
        "",
        "由 `base_instruct/`（70 模型）与 `other_models/`（24 模型）合并。",
        "",
        "格式：`均值 (var=方差) [min, max]`。tied 模型 E=U，U 列写 `(=E)`。",
        "",
        "原始长表：`all_models_row_norms.csv`",
        "",
        "## 表 1：Base–Instruct（35 对 / 70 模型）",
        "",
        "| Pair | Base E | Base U | Instruct E | Instruct U |",
        "|------|--------|--------|------------|------------|",
    ]
    for base, instr in pairs:
        md_lines.append(
            f"| {base} / {instr} "
            f"| {model_stat(bi_by_model, base, 'E')} "
            f"| {u_col(bi_by_model, bi_meta, base)} "
            f"| {model_stat(bi_by_model, instr, 'E')} "
            f"| {u_col(bi_by_model, bi_meta, instr)} |"
        )

    md_lines.extend(
        [
            "",
            "## 表 2：其他模型（24 模型）",
            "",
            "| Model | model_group | E | U | tied |",
            "|-------|-------------|---|---|------|",
        ]
    )
    for name in other_model_names:
        group = other_meta[name].get("model_group") or model_groups.get(name, "")
        md_lines.append(
            f"| {name} | {group} "
            f"| {model_stat(other_by_model, name, 'E')} "
            f"| {u_col(other_by_model, other_meta, name)} "
            f"| {other_meta[name]['tied']} |"
        )

    untied_pairs = [(b, i) for b, i in pairs if not bi_meta[b]["tied"]]
    md_lines.extend(
        [
            "",
            "## 表 3：Untied BI 行范数 mean 漂移（Δ%）",
            "",
            "Δ% = `(Instruct mean − Base mean) / Base mean × 100`。untied 模型 E≠U，**分别计算**。",
            "",
            "| Pair | ΔE | ΔU |",
            "|------|-----|-----|",
        ]
    )
    for base, instr in untied_pairs:
        dE, dU = bi_pair_deltas(bi_by_model, bi_meta, base, instr)
        md_lines.append(
            f"| {base} / {instr} | {fmt_delta_pct(dE)} | {fmt_delta_pct(dU)} |"
        )
    md_lines.append("")

    out_dir = row_norms_dir(repo, "all_models")
    (out_dir / "ALL_MODELS_ROW_NORMS_SUMMARY.md").write_text("\n".join(md_lines), encoding="utf-8")


def merge_all_models_row_norms(
    repo: Path,
    *,
    models_yaml: Path | None = None,
    pairs_file: Path | None = None,
) -> Path:
    models_yaml = models_yaml or (repo / "configs" / "models.yaml")
    model_groups = load_model_groups(models_yaml)

    bi_csv = row_norms_dir(repo, "base_instruct", "base_instruct_row_norms.csv")
    other_csv = row_norms_dir(repo, "other_models", "other_models_row_norms.csv")
    out_dir = row_norms_dir(repo, "all_models")
    out_csv = out_dir / "all_models_row_norms.csv"
    out_by_model = out_dir / "all_models_row_norms_by_model.csv"

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
    by_model, meta = build_model_tables(merged)
    write_model_summary_csv(
        out_by_model,
        by_model,
        meta,
        extra_columns=["subset", "model_group", "role"],
    )

    refresh_base_instruct_summaries(repo, pairs_file=pairs_file)
    write_all_models_summary(repo, models_yaml=models_yaml, pairs_file=pairs_file)
    return out_csv


_MU_RATIO_EXTRACTS: Path | None = None
_MU_RATIO_REPO: Path | None = None


def _init_mu_ratio_worker(repo_root: str, extracts_dir: str) -> None:
    global _MU_RATIO_EXTRACTS, _MU_RATIO_REPO
    _MU_RATIO_REPO = Path(repo_root)
    _MU_RATIO_EXTRACTS = Path(extracts_dir)
    src = _MU_RATIO_REPO / "ijcai_clean" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _mu_ratio_one_model(task: tuple[str, dict[str, object]]) -> list[dict[str, object]]:
    from ijcai_clean.data import load_E_U_matrices, load_info_json

    name, extra = task
    assert _MU_RATIO_EXTRACTS is not None
    info = load_info_json(_MU_RATIO_EXTRACTS, name)
    E, U, info = load_E_U_matrices(_MU_RATIO_EXTRACTS, name, info=info)
    emb_shape = info["standardized_dims"]["embed"]
    vocab_size, hidden_dim = int(emb_shape[0]), int(emb_shape[1])
    tied = bool(info.get("tie_word_embeddings"))

    rows: list[dict[str, object]] = []
    matrices = [("E", E)] if tied else [("E", E), ("U", U)]
    for matrix_name, M in matrices:
        mean_row_norm, mean_vec_norm, ratio = mean_vector_stats(M)
        rows.append(
            {
                "model": name,
                "matrix": matrix_name,
                "mean_row_norm": mean_row_norm,
                "mean_vec_norm": mean_vec_norm,
                "mu_over_row_norm": ratio,
                "tied": tied,
                "vocab_size": vocab_size,
                "hidden_dim": hidden_dim,
                **extra,
            }
        )
        print(
            f"{name:40} {matrix_name} mean_row={mean_row_norm:.6f} "
            f"||mu||={mean_vec_norm:.6f} ratio={ratio:.6f} tied={tied}",
            flush=True,
        )
    return rows


def all_models_from_yaml(models_yaml: Path) -> list[str]:
    cfg = yaml.safe_load(models_yaml.read_text(encoding="utf-8"))
    return sorted((cfg.get("model_repo_ids") or {}).keys())


MU_RATIO_FIELDS = [
    "model",
    "subset",
    "model_group",
    "role",
    "matrix",
    "mean_row_norm",
    "mean_vec_norm",
    "mu_over_row_norm",
    "tied",
    "vocab_size",
    "hidden_dim",
]


def run_mu_ratio_audit(
    repo: Path,
    *,
    models_yaml: Path,
    pairs_file: Path,
    extracts_dir: Path,
    workers: int | None = None,
) -> Path:
    model_groups = load_model_groups(models_yaml)
    pairs, bi_models_list = bi_models(pairs_file)
    bi_set = set(bi_models_list)
    models = all_models_from_yaml(models_yaml)

    tasks: list[tuple[str, dict[str, object]]] = []
    for name in models:
        extra: dict[str, object] = {
            "subset": "base_instruct" if name in bi_set else "other_models",
            "model_group": model_groups.get(name, ""),
            "role": (
                "instruct"
                if name in bi_set and is_instruct(name)
                else ("base" if name in bi_set else "")
            ),
        }
        tasks.append((name, extra))

    n_workers = workers or min(16, os.cpu_count() or 4)
    out_dir = row_norms_dir(repo, "mu_ratio")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "all_models_mu_ratio.csv"

    all_rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_mu_ratio_worker,
        initargs=(str(repo.resolve()), str(extracts_dir.resolve())),
    ) as pool:
        futures = {pool.submit(_mu_ratio_one_model, task): task[0] for task in tasks}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                all_rows.extend(fut.result())
            except Exception as exc:
                raise RuntimeError(f"mu_ratio audit failed for {name}") from exc

    all_rows.sort(key=lambda r: (str(r["model"]), str(r["matrix"])))
    write_long_csv(out_csv, all_rows, MU_RATIO_FIELDS)

    md_lines = [
        "# ‖μ‖ / mean(row norm) 汇总",
        "",
        "μ = 矩阵按行平均；mean(row norm) = 各行 L2 范数均值。",
        "ratio = ||μ|| / mean(row norm)，衡量行向量方向相干度。",
        "",
        f"并行 workers={n_workers}。原始表：`all_models_mu_ratio.csv`",
        "",
        "| Model | matrix | mean(row norm) | ||μ|| | ratio | tied |",
        "|-------|--------|----------------|-------|-------|------|",
    ]
    for row in all_rows:
        md_lines.append(
            f"| {row['model']} | {row['matrix']} | "
            f"{row['mean_row_norm']:.6f} | {row['mean_vec_norm']:.6f} | "
            f"{row['mu_over_row_norm']:.6f} | {row['tied']} |"
        )
    md_lines.append("")
    (out_dir / "ALL_MODELS_MU_RATIO_SUMMARY.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\nWrote {len(all_rows)} rows ({len(models)} models) -> {out_csv}", flush=True)
    return out_csv


def _full_spectrum_metrics(
    mat: "torch.Tensor",
    n: int,
    *,
    mu_hat: "torch.Tensor | None" = None,
) -> tuple[dict[str, float], "torch.Tensor"]:
    """All d singular values via Gram eigendecomposition (full economy SVD)."""
    import torch

    g = (mat.T @ mat) / n
    evals, evecs = torch.linalg.eigh(g)
    sigmas_sq = torch.clamp(evals * n, min=0.0)
    sigmas = torch.sqrt(sigmas_sq)
    total = sigmas_sq.sum()
    eps = 1e-12
    p = sigmas_sq / (total + eps)
    dim = int(p.numel())

    rank1_frac = float(p[-1].cpu())
    rank5_frac = float(p[-5:].sum().cpu()) if dim >= 5 else float(total.cpu())
    rank10_frac = float(p[-10:].sum().cpu()) if dim >= 10 else float(total.cpu())
    participation_ratio = float((total * total / (sigmas_sq * sigmas_sq).sum()).cpu())
    effective_rank = float(torch.exp(-(p * torch.log(p + eps)).sum()).cpu())
    sigma_max = float(sigmas[-1].cpu())
    sigma_min = float(sigmas[0].cpu())
    sigma_ratio = sigma_max / (sigma_min + eps)
    isotropy_pr_over_d = participation_ratio / dim if dim else float("nan")

    out: dict[str, float] = {
        "rank1_energy_frac": rank1_frac,
        "rank5_energy_frac": rank5_frac,
        "rank10_energy_frac": rank10_frac,
        "participation_ratio": participation_ratio,
        "effective_rank": effective_rank,
        "isotropy_pr_over_d": isotropy_pr_over_d,
        "sigma_ratio": sigma_ratio,
        "sigma1": sigma_max,
    }
    v1 = evecs[:, -1]
    if mu_hat is not None:
        out["cos_v1_mu"] = float(torch.abs(torch.dot(v1, mu_hat)).cpu())
    return out, v1


def spectral_stats_gpu(M: np.ndarray, device: str = "cuda:0") -> dict[str, float]:
    """Full economy SVD stats via d×d Gram on GPU (all d singular values)."""
    import torch

    X = torch.as_tensor(M, device=device, dtype=torch.float32)
    n = X.shape[0]
    row_norms = torch.linalg.norm(X, dim=1)
    mean_row_norm = float(row_norms.mean().cpu())
    mu = X.mean(dim=0)
    mean_vec_norm = float(torch.linalg.norm(mu).cpu())
    ratio = mean_vec_norm / mean_row_norm if mean_row_norm else float("nan")
    mu_hat = mu / (torch.linalg.norm(mu) + 1e-12)

    unc, v1_u = _full_spectrum_metrics(X, n, mu_hat=mu_hat)
    Xc = X - mu
    cen, v1_c = _full_spectrum_metrics(Xc, n)
    cos_v1_u_c = float(torch.abs(torch.dot(v1_u, v1_c)).cpu())

    sqrt_n = float(n**0.5)
    norm_denom = mean_row_norm * sqrt_n if mean_row_norm else float("nan")
    sigma1 = unc["sigma1"]
    sigma1_c = cen["sigma1"]

    del X, Xc, v1_u, v1_c
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "mean_row_norm": mean_row_norm,
        "mean_vec_norm": mean_vec_norm,
        "mu_over_row_norm": ratio,
        "sigma1": sigma1,
        "sigma1_centered": sigma1_c,
        "sigma1_over_mean_row": sigma1 / mean_row_norm if mean_row_norm else float("nan"),
        "sigma1_c_over_mean_row": sigma1_c / mean_row_norm if mean_row_norm else float("nan"),
        "sigma1_over_mean_sqrt_n": sigma1 / norm_denom,
        "sigma1_c_over_mean_sqrt_n": sigma1_c / norm_denom,
        "cos_v1_mu": unc.get("cos_v1_mu", float("nan")),
        "cos_v1_v1_centered": cos_v1_u_c,
        "rank1_energy_frac": unc["rank1_energy_frac"],
        "rank5_energy_frac": unc["rank5_energy_frac"],
        "rank10_energy_frac": unc["rank10_energy_frac"],
        "participation_ratio": unc["participation_ratio"],
        "effective_rank": unc["effective_rank"],
        "isotropy_pr_over_d": unc["isotropy_pr_over_d"],
        "sigma_ratio": unc["sigma_ratio"],
        "rank1_centered_energy_frac": cen["rank1_energy_frac"],
        "rank5_centered_energy_frac": cen["rank5_energy_frac"],
        "participation_ratio_centered": cen["participation_ratio"],
        "effective_rank_centered": cen["effective_rank"],
        "isotropy_pr_over_d_centered": cen["isotropy_pr_over_d"],
        "sigma_ratio_centered": cen["sigma_ratio"],
    }


SPECTRAL_FIELDS = [
    "model",
    "subset",
    "model_group",
    "role",
    "matrix",
    "mean_row_norm",
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
    "tied",
    "vocab_size",
    "hidden_dim",
]


def run_spectral_audit(
    repo: Path,
    *,
    models_yaml: Path,
    pairs_file: Path,
    extracts_dir: Path,
    device: str = "cuda:0",
) -> Path:
    import torch
    from ijcai_clean.data import load_E_U_matrices, load_info_json

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable, requested device={device}")
    if device.startswith("cuda"):
        torch.cuda.set_device(device)

    model_groups = load_model_groups(models_yaml)
    _, bi_models_list = bi_models(pairs_file)
    bi_set = set(bi_models_list)
    models = all_models_from_yaml(models_yaml)

    out_dir = row_norms_dir(repo, "spectral")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "all_models_spectral.csv"

    all_rows: list[dict[str, object]] = []
    for i, name in enumerate(models, 1):
        extra: dict[str, object] = {
            "subset": "base_instruct" if name in bi_set else "other_models",
            "model_group": model_groups.get(name, ""),
            "role": (
                "instruct"
                if name in bi_set and is_instruct(name)
                else ("base" if name in bi_set else "")
            ),
        }
        info = load_info_json(extracts_dir, name)
        E, U, info = load_E_U_matrices(extracts_dir, name, info=info)
        emb_shape = info["standardized_dims"]["embed"]
        vocab_size, hidden_dim = int(emb_shape[0]), int(emb_shape[1])
        tied = bool(info.get("tie_word_embeddings"))
        matrices = [("E", E)] if tied else [("E", E), ("U", U)]

        for matrix_name, M in matrices:
            stats = spectral_stats_gpu(M, device=device)
            row = {
                "model": name,
                "matrix": matrix_name,
                "tied": tied,
                "vocab_size": vocab_size,
                "hidden_dim": hidden_dim,
                **extra,
                **stats,
            }
            all_rows.append(row)
            print(
                f"[{i}/{len(models)}] {name:40} {matrix_name} "
                f"PR/d={stats['isotropy_pr_over_d']:.3f} "
                f"PR/d(c)={stats['isotropy_pr_over_d_centered']:.3f} "
                f"rank1={stats['rank1_energy_frac']:.3f} "
                f"rank1(c)={stats['rank1_centered_energy_frac']:.3f}",
                flush=True,
            )

    all_rows.sort(key=lambda r: (str(r["model"]), str(r["matrix"])))
    write_long_csv(out_csv, all_rows, SPECTRAL_FIELDS)

    md_lines = [
        "# E/U 谱分析（GPU 全 economy SVD / Gram）",
        "",
        "对 n×d 矩阵做 **完整 d 维** 奇异值谱（via Gram eigendecomposition，非慢速 full n-SVD）。",
        "",
        "**各向同性 / 异性指标**（centered 列带 `_centered` 或 `(c)`）：",
        "- `participation_ratio` (PR)：有效参与方向数，越大越各向同性",
        "- `isotropy_pr_over_d` = PR/d：→1 各向同性，→1/d 极度各向异性",
        "- `effective_rank`：谱熵有效秩",
        "- `rank1_energy_frac`：σ₁² 能量占比，越大越各向异性",
        "- `sigma_ratio`：σ_max/σ_min",
        "",
        f"device={device}。原始表：`all_models_spectral.csv`",
        "",
        "| Model | M | PR/d | PR/d(c) | rank1 | rank1(c) | eff_rank | σ_ratio | tied |",
        "|-------|---|------|---------|-------|----------|----------|---------|------|",
    ]
    for row in all_rows:
        md_lines.append(
            f"| {row['model']} | {row['matrix']} | "
            f"{row['isotropy_pr_over_d']:.3f} | {row['isotropy_pr_over_d_centered']:.3f} | "
            f"{row['rank1_energy_frac']:.3f} | {row['rank1_centered_energy_frac']:.3f} | "
            f"{row['effective_rank_centered']:.1f} | {row['sigma_ratio_centered']:.0f} | "
            f"{row['tied']} |"
        )
    md_lines.append("")
    (out_dir / "ALL_MODELS_SPECTRAL_SUMMARY.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\nWrote {len(all_rows)} rows ({len(models)} models) -> {out_csv}", flush=True)
    return out_csv
