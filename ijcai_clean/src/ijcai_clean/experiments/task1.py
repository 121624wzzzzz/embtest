from __future__ import annotations

import csv
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

from ijcai_clean.alignment import TokenSampler, build_pair_token_info
from ijcai_clean.data import (
    actual_tied,
    find_model_cache_dir,
    load_E_U_matrices,
    load_info_json,
)
from ijcai_clean.experiments.gcorr_io import (
    BOOTSTRAP_CSV_FIELDS,
    load_pairs_yaml,
    read_existing_bootstrap_rows,
    write_metadata,
    write_summary_from_bootstrap_csv,
)
from ijcai_clean.experiments.gcorr_validation import validate_completed_pairs_compute
from ijcai_clean.metrics import compute_single_pair_bootstrap


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

    if torch.cuda.is_available():
        from ijcai_clean.metrics import (
            DEFAULT_TARGET_MEMORY_BYTES,
            can_use_parallel_streams,
        )

        for dev in devices:
            if dev.type != "cuda":
                continue
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
            except Exception:
                free_bytes = total_bytes = 0
            mode = (
                "parallel-streams"
                if can_use_parallel_streams(dev, DEFAULT_TARGET_MEMORY_BYTES)
                else "serial"
            )
            print(
                f"  device {dev}: free={free_bytes / 1024**3:.1f}GB "
                f"total={total_bytes / 1024**3:.1f}GB mode={mode}",
                flush=True,
            )

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

    skipped_pairs: List[Dict[str, str]] = []

    for idx, (model_a, model_b) in enumerate(pairs, start=1):
        missing_bootstraps = [b for b in range(n_bootstrap) if (model_a, model_b, b) not in done]
        print(
            f"[{idx}/{len(pairs)}] {model_a} -> {model_b} "
            f"missing={len(missing_bootstraps)}/{n_bootstrap}",
            flush=True,
        )
        if not missing_bootstraps:
            continue

        try:
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
        except Exception as exc:
            reason = f"alignment failed: {exc}"
            print(f"  skip: {reason}", flush=True)
            skipped_pairs.append({"model_a": model_a, "model_b": model_b, "reason": reason})
            continue

        if info is None:
            reason = "common tokens < 1000"
            print(f"  skip: {reason}", flush=True)
            skipped_pairs.append({"model_a": model_a, "model_b": model_b, "reason": reason})
            continue

        try:
            data_a = _load_model_bundle(extracts_dir, model_a)
            data_b = _load_model_bundle(extracts_dir, model_b)
        except Exception as exc:
            reason = f"load failed: {exc}"
            print(f"  skip: {reason}", flush=True)
            skipped_pairs.append({"model_a": model_a, "model_b": model_b, "reason": reason})
            continue

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
        try:
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
        except Exception as exc:
            reason = f"bootstrap failed: {exc}"
            print(f"  skip remaining bootstraps for this pair: {reason}", flush=True)
            skipped_pairs.append({"model_a": model_a, "model_b": model_b, "reason": reason})

        del data_a, data_b
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    skipped_csv = out_dir / "runner_skipped_pairs.csv"
    if skipped_pairs:
        with skipped_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["model_a", "model_b", "reason"])
            writer.writeheader()
            for row in skipped_pairs:
                writer.writerow(row)
        print(
            f"runner-skipped pairs: {len(skipped_pairs)} (see {skipped_csv})",
            flush=True,
        )
    elif skipped_csv.is_file():
        skipped_csv.unlink()

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
