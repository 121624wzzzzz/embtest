"""Layer 2: mean-vector ratio audit."""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .common import (
    F_LAYER2_MU_RATIO,
    catalog_context,
    mean_vector_stats,
    layers_dir,
    layers_file,
    row_metadata,
    write_long_csv,
)

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

_MU_RATIO_EXTRACTS: Path | None = None
_MU_RATIO_REPO: Path | None = None


def _init_mu_ratio_worker(repo_root: str, extracts_dir: str) -> None:
    global _MU_RATIO_EXTRACTS, _MU_RATIO_REPO
    _MU_RATIO_REPO = Path(repo_root)
    _MU_RATIO_EXTRACTS = Path(extracts_dir)
    src = _MU_RATIO_REPO / "cross_model_geometry" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _mu_ratio_one_model(task: tuple[str, dict[str, object]]) -> list[dict[str, object]]:
    from cross_model_geometry.data import load_E_U_matrices, load_info_json

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


def run_mu_ratio_audit(
    repo: Path,
    *,
    models_yaml: Path,
    pairs_file: Path,
    extracts_dir: Path,
    workers: int | None = None,
) -> Path:
    model_groups, bi_set, models = catalog_context(models_yaml, pairs_file)

    tasks: list[tuple[str, dict[str, object]]] = [
        (name, row_metadata(name, bi_set=bi_set, model_groups=model_groups))
        for name in models
    ]

    n_workers = workers or min(16, os.cpu_count() or 4)
    layers_dir(repo).mkdir(parents=True, exist_ok=True)
    out_csv = layers_file(repo, F_LAYER2_MU_RATIO)

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
    print(f"\nWrote {len(all_rows)} rows ({len(models)} models) -> {out_csv}", flush=True)

    from .features import merge_all_models_features

    features_csv = merge_all_models_features(repo)
    if features_csv is not None:
        print(f"Merged feature table -> {features_csv}", flush=True)
    return out_csv
