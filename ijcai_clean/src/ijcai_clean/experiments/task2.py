from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

import yaml

from ijcai_clean.paths import rel_to_repo

if TYPE_CHECKING:
    import torch

Pair = Tuple[str, str]
PairRecord = Dict[str, Any]


def _select_representative(item: Any) -> Tuple[str, str | None]:
    """
    model_series.yaml 中的列表项表示同一模型的 base/instruct 候选。
    若二者同时存在，任务二按需求使用 instruct 侧与其他模型比较。
    """
    if isinstance(item, str):
        return item.strip(), None
    if isinstance(item, (list, tuple)) and item:
        selected = str(item[-1]).strip()
        base = str(item[0]).strip() if len(item) > 1 else None
        return selected, base
    raise ValueError(f"无效的 series 条目: {item!r}")


def load_series_yaml(path: Path) -> Dict[str, List[Tuple[str, str | None]]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    series_raw = data.get("series") or {}
    if not isinstance(series_raw, dict):
        raise ValueError(f"{path}: series 必须是 mapping")

    series: Dict[str, List[Tuple[str, str | None]]] = {}
    for series_name, items in series_raw.items():
        if not isinstance(items, list):
            raise ValueError(f"{path}: series.{series_name} 必须是 list")
        selected = [_select_representative(item) for item in items]
        series[str(series_name)] = selected
    return series


def build_task2_pairs(series_file: Path) -> Tuple[List[Pair], List[PairRecord]]:
    series = load_series_yaml(series_file)
    pairs: List[Pair] = []
    records: List[PairRecord] = []

    for series_name, selected_items in series.items():
        reps = [model for model, _base_model in selected_items]
        for left_idx, right_idx in combinations(range(len(reps)), 2):
            model_a = reps[left_idx]
            model_b = reps[right_idx]
            pairs.append((model_a, model_b))
            records.append(
                {
                    "series": series_name,
                    "model_a": model_a,
                    "model_b": model_b,
                    "source_a": selected_items[left_idx][1] or model_a,
                    "source_b": selected_items[right_idx][1] or model_b,
                    "selected_a": model_a,
                    "selected_b": model_b,
                }
            )

    return pairs, records


def write_generated_pairs_yaml(path: Path, pairs: Sequence[Pair]) -> None:
    data = {
        "task": "task2_model_series_generated_pairs",
        "pairs": [[model_a, model_b] for model_a, model_b in pairs],
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_pair_plan_csv(path: Path, records: Sequence[PairRecord]) -> None:
    fieldnames = ["series", "model_a", "model_b", "source_a", "source_b", "selected_a", "selected_b"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def _model_extract_status(extracts_dir: Path, model_name: str) -> Dict[str, Any]:
    info_path = extracts_dir / f"{model_name}.info.json"
    matrix_path = extracts_dir / f"{model_name}.safetensors"
    info_has_embed = False
    info_has_lm_head = False
    reason = ""
    if info_path.is_file():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            dims = info.get("standardized_dims") or {}
            sources = info.get("standardized_sources") or {}
            info_has_embed = bool(dims.get("embed") and sources.get("embed"))
            info_has_lm_head = bool(dims.get("lm_head") and sources.get("lm_head"))
            if not info_has_embed:
                reason = "missing standardized embed"
            elif not info_has_lm_head:
                reason = "missing standardized lm_head"
        except json.JSONDecodeError as exc:
            reason = f"invalid info json: {exc}"
    elif not matrix_path.is_file():
        reason = "missing info and matrix files"
    else:
        reason = "missing info file"
    if info_path.is_file() and not matrix_path.is_file():
        reason = "missing matrix file"
    return {
        "model": model_name,
        "info_exists": info_path.is_file(),
        "matrix_exists": matrix_path.is_file(),
        "info_has_embed": info_has_embed,
        "info_has_lm_head": info_has_lm_head,
        "reason": reason,
        "info_path": str(info_path),
        "matrix_path": str(matrix_path),
    }


def _filter_pairs_with_extracts(
    *,
    pairs: Sequence[Pair],
    records: Sequence[PairRecord],
    extracts_dir: Path,
) -> Tuple[List[Pair], List[PairRecord], List[Dict[str, Any]]]:
    model_names = sorted({name for pair in pairs for name in pair})
    status_by_model = {name: _model_extract_status(extracts_dir, name) for name in model_names}
    skipped_models = [
        status
        for status in status_by_model.values()
        if not status["info_exists"]
        or not status["matrix_exists"]
        or not status["info_has_embed"]
        or not status["info_has_lm_head"]
    ]
    missing = {status["model"] for status in skipped_models}
    if not missing:
        return list(pairs), list(records), []

    filtered_pairs: List[Pair] = []
    filtered_records: List[PairRecord] = []
    for pair, record in zip(pairs, records):
        if pair[0] in missing or pair[1] in missing:
            continue
        filtered_pairs.append(pair)
        filtered_records.append(record)
    return filtered_pairs, filtered_records, skipped_models


def write_skipped_models_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "info_exists",
        "matrix_exists",
        "info_has_embed",
        "info_has_lm_head",
        "reason",
        "info_path",
        "matrix_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _series_pair_counts(records: Sequence[PairRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in records:
        series_name = str(row["series"])
        counts[series_name] = counts.get(series_name, 0) + 1
    return counts


def _augment_metadata(
    *,
    meta_json: Path,
    repo_root: Path,
    series_file: Path,
    generated_pairs_file: Path,
    pair_plan_csv: Path,
    records: Sequence[PairRecord],
    skipped_models_csv: Path,
    skipped_models: Sequence[Dict[str, Any]],
) -> None:
    if not meta_json.is_file():
        return
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    meta.update(
        {
            "task": "task2_model_series",
            "series_file": rel_to_repo(series_file, repo_root),
            "generated_pairs_file": rel_to_repo(generated_pairs_file, repo_root),
            "pair_plan_csv": rel_to_repo(pair_plan_csv, repo_root),
            "skipped_models_csv": rel_to_repo(skipped_models_csv, repo_root),
            "skipped_models": list(skipped_models),
            "selection_rule": "if base/instruct candidates both exist, use the instruct-side model",
            "series_pair_counts": _series_pair_counts(records),
        }
    )
    meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_task2_model_series(
    *,
    repo_root: Path,
    series_file: Path,
    extracts_dir: Path,
    models_yaml: Path,
    out_dir: Path,
    n_tokens: int,
    n_pairs: int,
    n_bootstrap: int,
    random_seed: int,
    devices: Sequence["torch.device"],
    complete_mode: str = "validate",
    validation_n_tokens: int = 1024,
    validation_n_pairs: int = 10000,
) -> None:
    from ijcai_clean.experiments.task1 import run_task1_base_instruct

    pairs, records = build_task2_pairs(series_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs, records, skipped_models = _filter_pairs_with_extracts(
        pairs=pairs,
        records=records,
        extracts_dir=extracts_dir,
    )

    generated_pairs_file = out_dir / "generated_pairs.yaml"
    pair_plan_csv = out_dir / "pair_plan.csv"
    skipped_models_csv = out_dir / "skipped_models.csv"
    write_generated_pairs_yaml(generated_pairs_file, pairs)
    write_pair_plan_csv(pair_plan_csv, records)
    write_skipped_models_csv(skipped_models_csv, skipped_models)

    print(
        f"task2 generated {len(pairs)} pair(s) from {len(load_series_yaml(series_file))} series",
        flush=True,
    )
    if skipped_models:
        print(
            "skip models without complete extracts: "
            + ", ".join(str(row["model"]) for row in skipped_models),
            flush=True,
        )
    print(f"pair plan: {pair_plan_csv}", flush=True)

    run_task1_base_instruct(
        repo_root=repo_root,
        pairs_file=generated_pairs_file,
        extracts_dir=extracts_dir,
        models_yaml=models_yaml,
        out_dir=out_dir,
        n_tokens=n_tokens,
        n_pairs=n_pairs,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
        devices=devices,
        complete_mode=complete_mode,
        validation_n_tokens=validation_n_tokens,
        validation_n_pairs=validation_n_pairs,
    )

    _augment_metadata(
        meta_json=out_dir / "metadata.json",
        repo_root=repo_root,
        series_file=series_file,
        generated_pairs_file=generated_pairs_file,
        pair_plan_csv=pair_plan_csv,
        records=records,
        skipped_models_csv=skipped_models_csv,
        skipped_models=skipped_models,
    )
