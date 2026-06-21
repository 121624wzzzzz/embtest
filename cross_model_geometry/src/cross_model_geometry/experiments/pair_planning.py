"""Pair 规划公共逻辑（被 Task2 / Task3 / Task4 共用）。

提供：
- ``select_representative(item)``：从 ``model_series.yaml`` 条目中选 instruct 侧
  作为代表，并返回 (selected, base_or_none)。
- ``model_extract_status(extracts_dir, name)``：检查某模型的 ``extracts/``
  状态，缺 info / matrix / standardized_sources 时给出原因。
- ``filter_pairs_with_extracts(pairs, records, extracts_dir)``：剔除任一侧
  缺 extracts 的 pair，返回 (kept_pairs, kept_records, skipped_status)。
- ``write_skipped_models_csv(path, rows)``：把 skipped_models 写成 CSV。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

Pair = Tuple[str, str]
PairRecord = Dict[str, Any]


def select_representative(item: Any) -> Tuple[str, str | None]:
    """``model_series.yaml`` 中的列表项表示同一模型的 base/instruct 候选。

    若二者同时存在，按主线约定使用 instruct 侧作为对比代表，
    返回 (selected, base_name_or_None)。
    """
    if isinstance(item, str):
        return item.strip(), None
    if isinstance(item, (list, tuple)) and item:
        selected = str(item[-1]).strip()
        base = str(item[0]).strip() if len(item) > 1 else None
        return selected, base
    raise ValueError(f"无效的 series 条目: {item!r}")


def model_extract_status(extracts_dir: Path, model_name: str) -> Dict[str, Any]:
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


def filter_pairs_with_extracts(
    *,
    pairs: Sequence[Pair],
    records: Sequence[PairRecord],
    extracts_dir: Path,
) -> Tuple[List[Pair], List[PairRecord], List[Dict[str, Any]]]:
    """剔除任一侧 extracts 不完整的 pair。"""
    model_names = sorted({name for pair in pairs for name in pair})
    status_by_model = {name: model_extract_status(extracts_dir, name) for name in model_names}
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
