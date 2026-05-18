from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

import yaml

from ijcai_clean.experiments.pair_planning import (
    Pair,
    PairRecord,
    filter_pairs_with_extracts,
    select_representative,
    write_skipped_models_csv,
)
from ijcai_clean.paths import rel_to_repo

if TYPE_CHECKING:
    import torch

SeriesIndex = Dict[str, str]


def _series_index_from_yaml(path: Path) -> SeriesIndex:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    series_raw = data.get("series") or {}
    if not isinstance(series_raw, dict):
        raise ValueError(f"{path}: series 必须是 mapping")

    index: SeriesIndex = {}
    for series_name, items in series_raw.items():
        if not isinstance(items, list):
            raise ValueError(f"{path}: series.{series_name} 必须是 list")
        for item in items:
            selected, source = select_representative(item)
            index[selected] = str(series_name)
            if source:
                index[source] = str(series_name)
            if isinstance(item, (list, tuple)):
                for name in item:
                    index[str(name).strip()] = str(series_name)
    return index


def load_scale_groups_yaml(path: Path, series_file: Path) -> Tuple[Dict[str, List[Tuple[str, str | None]]], SeriesIndex]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    groups_raw = data.get("scale_groups") or {}
    if not isinstance(groups_raw, dict):
        raise ValueError(f"{path}: scale_groups 必须是 mapping")

    groups: Dict[str, List[Tuple[str, str | None]]] = {}
    for group_name, items in groups_raw.items():
        if not isinstance(items, list):
            raise ValueError(f"{path}: scale_groups.{group_name} 必须是 list")
        groups[str(group_name)] = [select_representative(item) for item in items]
    return groups, _series_index_from_yaml(series_file)


def build_task3_pairs(scale_groups_file: Path, series_file: Path) -> Tuple[List[Pair], List[PairRecord]]:
    groups, series_index = load_scale_groups_yaml(scale_groups_file, series_file)
    pairs: List[Pair] = []
    records: List[PairRecord] = []
    seen: set[Tuple[str, str]] = set()

    for scale_group, selected_items in groups.items():
        reps = [model for model, _source_model in selected_items]
        for left_idx, right_idx in combinations(range(len(reps)), 2):
            model_a = reps[left_idx]
            model_b = reps[right_idx]
            series_a = series_index.get(model_a, "unknown")
            series_b = series_index.get(model_b, "unknown")
            if series_a == series_b:
                continue

            key = tuple(sorted((model_a, model_b)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((model_a, model_b))
            records.append(
                {
                    "scale_group": scale_group,
                    "series_a": series_a,
                    "series_b": series_b,
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
        "task": "task3_cross_scale_groups_generated_pairs",
        "pairs": [[model_a, model_b] for model_a, model_b in pairs],
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_pair_plan_csv(path: Path, records: Sequence[PairRecord]) -> None:
    fieldnames = [
        "scale_group",
        "series_a",
        "series_b",
        "model_a",
        "model_b",
        "source_a",
        "source_b",
        "selected_a",
        "selected_b",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def _scale_pair_counts(records: Sequence[PairRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in records:
        scale_group = str(row["scale_group"])
        counts[scale_group] = counts.get(scale_group, 0) + 1
    return counts


def _augment_metadata(
    *,
    meta_json: Path,
    repo_root: Path,
    scale_groups_file: Path,
    series_file: Path,
    generated_pairs_file: Path,
    pair_plan_csv: Path,
    records: Sequence[PairRecord],
    skipped_models_csv: Path,
    skipped_models: Sequence[Dict[str, Any]],
    task_name: str = "task3_cross_scale_groups",
) -> None:
    if not meta_json.is_file():
        return
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    meta.update(
        {
            "task": task_name,
            "scale_groups_file": rel_to_repo(scale_groups_file, repo_root),
            "series_file": rel_to_repo(series_file, repo_root),
            "generated_pairs_file": rel_to_repo(generated_pairs_file, repo_root),
            "pair_plan_csv": rel_to_repo(pair_plan_csv, repo_root),
            "skipped_models_csv": rel_to_repo(skipped_models_csv, repo_root),
            "skipped_models": list(skipped_models),
            "selection_rule": "if base/instruct candidates both exist, use the instruct-side model",
            "dedupe_rule": "skip pairs whose selected models belong to the same series",
            "scale_pair_counts": _scale_pair_counts(records),
        }
    )
    meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_task3_cross_scale_groups(
    *,
    repo_root: Path,
    scale_groups_file: Path,
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
    task_name: str = "task3_cross_scale_groups",
) -> None:
    from ijcai_clean.experiments.task1 import run_task1_base_instruct

    pairs, records = build_task3_pairs(scale_groups_file, series_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs, records, skipped_models = filter_pairs_with_extracts(
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

    print(f"task3 generated {len(pairs)} cross-scale pair(s)", flush=True)
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
        scale_groups_file=scale_groups_file,
        series_file=series_file,
        generated_pairs_file=generated_pairs_file,
        pair_plan_csv=pair_plan_csv,
        records=records,
        skipped_models_csv=skipped_models_csv,
        skipped_models=skipped_models,
        task_name=task_name,
    )
