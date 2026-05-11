"""任务五：仿射 R² 关系（E↔E / U↔U / intra E→U）。

复用 task1..4 已生成的 pairs YAML，按 (model_a, model_b) 并集去重后逐对做仿射拟合：

    Y ≈ X · A + b   （中心化最小二乘，``torch.linalg.lstsq``）

并报告 R²、相对误差、||A||_F、||b||_2。同时对所有出现过的模型并集做
intra E→U 仿射检验（U ≈ A·E + b），用于判断该模型 E 与 U 是否近似仿射相关。

设计要点：
- 数学一致性：仿射拟合实现与 ``legacy/exp2_affine_cross_model`` 等价（中心化拟合 +
  Y_pred = X·A^T + b 评估），方便和老结果数值对比。
- 数据复用：直接读各 task 的 ``generated_pairs.yaml``，token 对齐用
  ``ijcai_clean.alignment.build_pair_token_info``，加载用 ``data._load_model_bundle``。
- 多卡并行：单 pair 一个 ``torch.linalg.lstsq``，按 GPU 数开 ThreadPool；同卡不并发，
  避免 lstsq 工作区互踩。
- 模型 LRU 缓存：避免大模型反复 load。
- 容错：单对失败仅写入 ``runner_skipped_pairs.csv``，不中断整体流程。
"""
from __future__ import annotations

import csv
import gc
import json
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from ijcai_clean.alignment import TokenSampler, build_pair_token_info
from ijcai_clean.data import (
    actual_tied,
    find_model_cache_dir,
    load_E_U_matrices,
    load_info_json,
)
from ijcai_clean.experiments.task1_io import load_pairs_yaml, try_git_commit
from ijcai_clean.paths import rel_to_repo

DEFAULT_MAX_FIT_ROWS = 24_000
DEFAULT_MIN_COMMON_TOKENS = 5_000

PAIR_CSV_FIELDS = [
    "model_a",
    "model_b",
    "source_tasks",
    "align_mode",
    "n_common",
    "n_fit",
    "hidden_dim_a",
    "hidden_dim_b",
    "vocab_size_a",
    "vocab_size_b",
    "actual_tied_a",
    "actual_tied_b",
    "R2_E",
    "rel_err_E",
    "norm_A_E",
    "norm_b_E",
    "R2_U",
    "rel_err_U",
    "norm_A_U",
    "norm_b_U",
]

INTRA_CSV_FIELDS = [
    "model",
    "hidden_dim",
    "vocab_size",
    "actual_tied",
    "n_fit",
    "R2_EU",
    "rel_err_EU",
    "norm_A_EU",
    "norm_b_EU",
]


# ---------------------------------------------------------------------------
# 仿射拟合核心
# ---------------------------------------------------------------------------


def fit_affine_general(
    X: np.ndarray, Y: np.ndarray, device: torch.device
) -> Tuple[float, float, float, float]:
    """中心化一般仿射拟合：Y ≈ X·A + b。

    返回 (R², rel_err, ||A||_F, ||b||_2)。
    与 legacy 行为一致：rcond=None，float32，GPU 上执行。
    """
    x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)
    y = torch.from_numpy(np.ascontiguousarray(Y, dtype=np.float32)).to(device)
    try:
        mx = x.mean(dim=0)
        my = y.mean(dim=0)
        x_c = x - mx
        y_c = y - my
        # Y_c = X_c · A  =>  A = lstsq(X_c, Y_c).solution，shape (d_x, d_y)
        a = torch.linalg.lstsq(x_c, y_c, rcond=None).solution
        b = my - mx @ a
        y_pred = x @ a + b
        ss_res = (y - y_pred).pow(2).sum().item()
        ss_tot = y_c.pow(2).sum().item()
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        y_norm = y.pow(2).sum().sqrt().item()
        rel_err = (ss_res ** 0.5) / (y_norm + 1e-20)
        norm_a = a.norm("fro").item()
        norm_b = b.norm().item() if b.dim() > 0 else float(b.item())
    finally:
        del x, y
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    return r2, rel_err, norm_a, norm_b


# ---------------------------------------------------------------------------
# Pair 来源装载
# ---------------------------------------------------------------------------


def _resolve_source_path(repo_root: Path, sources_file: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    cand = (repo_root / p).resolve()
    if cand.is_file():
        return cand
    return (sources_file.parent / p).resolve()


def load_affine_sources(
    sources_file: Path, repo_root: Path
) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], List[str]]]:
    """读 affine_pairs.yaml -> (去重后的 pairs, pair -> 来源 task 名列表)。

    去重 key 用 ``tuple(sorted((a, b)))`` 以避免 (a,b) / (b,a) 重复；
    实际写入时仍保留首次出现的 (model_a, model_b) 顺序。
    """
    data = yaml.safe_load(sources_file.read_text(encoding="utf-8"))
    sources_raw = data.get("sources") or []
    if not isinstance(sources_raw, list) or not sources_raw:
        raise ValueError(f"{sources_file}: sources 必须是非空 list")

    seen: Dict[Tuple[str, str], Tuple[str, str]] = {}
    src_index: Dict[Tuple[str, str], List[str]] = {}
    for entry in sources_raw:
        name = str(entry.get("name") or "").strip()
        pairs_yaml_raw = entry.get("pairs_yaml")
        if not name or not pairs_yaml_raw:
            continue
        path = _resolve_source_path(repo_root, sources_file, str(pairs_yaml_raw))
        if not path.is_file():
            print(f"  warn: source not found, skip: {name} ({path})", flush=True)
            continue
        for a, b in load_pairs_yaml(path):
            key = tuple(sorted((a, b)))
            if key not in seen:
                seen[key] = (a, b)
                src_index[key] = []
            if name not in src_index[key]:
                src_index[key].append(name)
    pairs = list(seen.values())
    pair_sources = {seen[k]: src_index[k] for k in seen}
    return pairs, pair_sources


# ---------------------------------------------------------------------------
# 模型 bundle LRU 缓存（与 task1._load_model_bundle 保持一致）
# ---------------------------------------------------------------------------


class _ModelBundleCache:
    def __init__(self, extracts_dir: Path, max_models: int = 8) -> None:
        self.extracts_dir = extracts_dir
        self.max_models = max(1, max_models)
        self._cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, name: str) -> Dict[str, Any]:
        with self._lock:
            if name in self._cache:
                self._cache.move_to_end(name)
                return self._cache[name]
        bundle = self._load(name)
        with self._lock:
            self._cache[name] = bundle
            self._cache.move_to_end(name)
            while len(self._cache) > self.max_models:
                old_name, old_bundle = self._cache.popitem(last=False)
                del old_bundle
            return bundle

    def _load(self, name: str) -> Dict[str, Any]:
        info = load_info_json(self.extracts_dir, name)
        E, U, info = load_E_U_matrices(self.extracts_dir, name, info=info)
        emb_shape = info["standardized_dims"]["embed"]
        vocab_size, hidden_dim = int(emb_shape[0]), int(emb_shape[1])
        return {
            "E": E,
            "U": U,
            "info": info,
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
            "actual_tied": actual_tied(E, U),
        }


# ---------------------------------------------------------------------------
# Tokenizer 缓存
# ---------------------------------------------------------------------------


class _TokenizerCache:
    def __init__(self, repo_root: Path, models_yaml: Path) -> None:
        self.repo_root = repo_root
        self.models_yaml = models_yaml
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, model_name: str):
        with self._lock:
            if model_name in self._cache:
                return self._cache[model_name]
        d = find_model_cache_dir(self.repo_root, model_name, self.models_yaml)
        tok = (
            None
            if d is None
            else AutoTokenizer.from_pretrained(str(d), trust_remote_code=True)
        )
        with self._lock:
            self._cache[model_name] = tok
        return tok


# ---------------------------------------------------------------------------
# 单对 / 单模型仿射执行
# ---------------------------------------------------------------------------


def _deterministic_subsample(n_total: int, n_keep: int, seed_str: str) -> np.ndarray:
    if n_total <= n_keep:
        return np.arange(n_total, dtype=np.int64)
    seed = sum(ord(c) for c in seed_str) % (2 ** 32)
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=n_keep, replace=False)


def _run_pair_affine(
    *,
    model_a: str,
    model_b: str,
    bundle_cache: _ModelBundleCache,
    tokenizer_cache: _TokenizerCache,
    sampler: TokenSampler,
    max_fit_rows: int,
    min_common_tokens: int,
    device: torch.device,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    info_a = load_info_json(bundle_cache.extracts_dir, model_a)
    info_b = load_info_json(bundle_cache.extracts_dir, model_b)
    vocab_a = int(info_a["standardized_dims"]["embed"][0])
    vocab_b = int(info_b["standardized_dims"]["embed"][0])

    pair_info = build_pair_token_info(
        vocab_a=vocab_a,
        vocab_b=vocab_b,
        tokenizer_a=tokenizer_cache.get(model_a),
        tokenizer_b=tokenizer_cache.get(model_b),
        n_tokens=max_fit_rows,
        sampler=sampler,
    )
    if pair_info is None:
        return None, "common tokens < 1000"
    ids_a = pair_info["ids_a"]
    ids_b = pair_info["ids_b"]
    n_common = int(len(ids_a))
    if n_common < min_common_tokens:
        return None, f"n_common={n_common} < min_common_tokens={min_common_tokens}"

    keep = _deterministic_subsample(n_common, max_fit_rows, model_a + "|" + model_b)
    n_fit = int(len(keep))
    sel_a = ids_a[keep]
    sel_b = ids_b[keep]

    data_a = bundle_cache.get(model_a)
    data_b = bundle_cache.get(model_b)

    E_a = data_a["E"][sel_a]
    E_b = data_b["E"][sel_b]
    U_a = data_a["U"][sel_a]
    U_b = data_b["U"][sel_b]

    r2_e, rel_e, norm_a_e, norm_b_e = fit_affine_general(E_a, E_b, device)
    r2_u, rel_u, norm_a_u, norm_b_u = fit_affine_general(U_a, U_b, device)

    return (
        {
            "model_a": model_a,
            "model_b": model_b,
            "align_mode": pair_info["align_mode"],
            "n_common": n_common,
            "n_fit": n_fit,
            "hidden_dim_a": data_a["hidden_dim"],
            "hidden_dim_b": data_b["hidden_dim"],
            "vocab_size_a": data_a["vocab_size"],
            "vocab_size_b": data_b["vocab_size"],
            "actual_tied_a": data_a["actual_tied"],
            "actual_tied_b": data_b["actual_tied"],
            "R2_E": r2_e,
            "rel_err_E": rel_e,
            "norm_A_E": norm_a_e,
            "norm_b_E": norm_b_e,
            "R2_U": r2_u,
            "rel_err_U": rel_u,
            "norm_A_U": norm_a_u,
            "norm_b_U": norm_b_u,
        },
        None,
    )


def _run_intra_EU(
    *,
    model: str,
    bundle_cache: _ModelBundleCache,
    tokenizer_cache: _TokenizerCache,
    sampler: TokenSampler,
    max_fit_rows: int,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    data = bundle_cache.get(model)
    if data["actual_tied"]:
        return {
            "model": model,
            "hidden_dim": data["hidden_dim"],
            "vocab_size": data["vocab_size"],
            "actual_tied": True,
            "n_fit": 0,
            "R2_EU": 1.0,
            "rel_err_EU": 0.0,
            "norm_A_EU": float("nan"),
            "norm_b_EU": float("nan"),
        }

    tok = tokenizer_cache.get(model)
    if tok is not None:
        valid = sampler.get_valid_token_ids(tok)
        if len(valid) >= 1000:
            valid = valid[valid < data["vocab_size"]]
        else:
            valid = np.arange(data["vocab_size"], dtype=np.int64)
    else:
        valid = np.arange(data["vocab_size"], dtype=np.int64)
    keep = _deterministic_subsample(len(valid), max_fit_rows, "intra|" + model)
    sel = valid[keep]
    n_fit = int(len(sel))

    E = data["E"][sel]
    U = data["U"][sel]
    r2, rel_err, norm_a, norm_b = fit_affine_general(E, U, device)

    return {
        "model": model,
        "hidden_dim": data["hidden_dim"],
        "vocab_size": data["vocab_size"],
        "actual_tied": False,
        "n_fit": n_fit,
        "R2_EU": r2,
        "rel_err_EU": rel_err,
        "norm_A_EU": norm_a,
        "norm_b_EU": norm_b,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _print_device_status(devices: Sequence[torch.device]) -> None:
    if not torch.cuda.is_available():
        return
    for dev in devices:
        if dev.type != "cuda":
            continue
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
        except Exception:
            free_bytes = total_bytes = 0
        print(
            f"  device {dev}: free={free_bytes / 1024**3:.1f}GB "
            f"total={total_bytes / 1024**3:.1f}GB",
            flush=True,
        )


def run_task5_affine_relations(
    *,
    repo_root: Path,
    sources_file: Path,
    extracts_dir: Path,
    models_yaml: Path,
    out_dir: Path,
    devices: Sequence[torch.device],
    max_fit_rows: int = DEFAULT_MAX_FIT_ROWS,
    min_common_tokens: int = DEFAULT_MIN_COMMON_TOKENS,
    cache_max_models: int = 8,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs, pair_sources = load_affine_sources(sources_file, repo_root)
    print(f"task5: loaded {len(pairs)} unique pair(s) from sources", flush=True)
    _print_device_status(devices)

    sampler = TokenSampler()
    bundle_cache = _ModelBundleCache(extracts_dir, max_models=cache_max_models)
    tokenizer_cache = _TokenizerCache(repo_root, models_yaml)

    pair_csv = out_dir / "summary_pair.csv"
    intra_csv = out_dir / "summary_intra_EU.csv"
    skipped_csv = out_dir / "runner_skipped_pairs.csv"
    meta_json = out_dir / "metadata.json"

    file_lock = threading.Lock()

    def append_row(path: Path, fields: Sequence[str], row: Dict[str, Any]) -> None:
        with file_lock:
            write_header = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
                if write_header:
                    w.writeheader()
                w.writerow(row)

    skipped_pairs: List[Dict[str, str]] = []
    n_devices = max(len(devices), 1)
    device_locks = [threading.Lock() for _ in range(n_devices)]

    def pair_job(idx: int, model_a: str, model_b: str) -> None:
        slot = (idx - 1) % n_devices
        device = devices[slot]
        with device_locks[slot]:
            try:
                row, skip_reason = _run_pair_affine(
                    model_a=model_a,
                    model_b=model_b,
                    bundle_cache=bundle_cache,
                    tokenizer_cache=tokenizer_cache,
                    sampler=sampler,
                    max_fit_rows=max_fit_rows,
                    min_common_tokens=min_common_tokens,
                    device=device,
                )
            except Exception as exc:
                row = None
                skip_reason = f"affine pair failed: {exc}"

        if row is None:
            print(
                f"[{idx}/{len(pairs)}] {model_a} -> {model_b}  skip: {skip_reason}",
                flush=True,
            )
            with file_lock:
                skipped_pairs.append(
                    {"model_a": model_a, "model_b": model_b, "reason": skip_reason or "unknown"}
                )
            return

        row["source_tasks"] = "|".join(pair_sources.get((model_a, model_b), []))
        append_row(pair_csv, PAIR_CSV_FIELDS, row)
        print(
            f"[{idx}/{len(pairs)}] {model_a} -> {model_b}  "
            f"R2_E={row['R2_E']:.4f}  R2_U={row['R2_U']:.4f}  "
            f"n_fit={row['n_fit']} on {device}",
            flush=True,
        )

    print("\n=== pair-wise affine fits ===", flush=True)
    if pair_csv.exists():
        pair_csv.unlink()
    with ThreadPoolExecutor(max_workers=n_devices) as ex:
        futures = [ex.submit(pair_job, i + 1, a, b) for i, (a, b) in enumerate(pairs)]
        for fut in as_completed(futures):
            fut.result()

    # ---------------- intra E→U ----------------
    print("\n=== intra-model E -> U affine fits ===", flush=True)
    if intra_csv.exists():
        intra_csv.unlink()
    all_models = sorted({m for pair in pairs for m in pair})
    print(f"intra: {len(all_models)} unique model(s)", flush=True)

    skipped_models: List[Dict[str, str]] = []

    def intra_job(idx: int, model: str) -> None:
        slot = (idx - 1) % n_devices
        device = devices[slot]
        with device_locks[slot]:
            try:
                row = _run_intra_EU(
                    model=model,
                    bundle_cache=bundle_cache,
                    tokenizer_cache=tokenizer_cache,
                    sampler=sampler,
                    max_fit_rows=max_fit_rows,
                    device=device,
                )
            except Exception as exc:
                row = None
                reason = f"intra failed: {exc}"
                with file_lock:
                    skipped_models.append({"model": model, "reason": reason})
                print(
                    f"[intra {idx}/{len(all_models)}] {model}  skip: {reason}",
                    flush=True,
                )
                return

        if row is None:
            with file_lock:
                skipped_models.append({"model": model, "reason": "unknown"})
            return
        append_row(intra_csv, INTRA_CSV_FIELDS, row)
        print(
            f"[intra {idx}/{len(all_models)}] {model}  "
            f"R2_EU={row['R2_EU']:.4f}  tied={row['actual_tied']}  "
            f"n_fit={row['n_fit']} on {device}",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=n_devices) as ex:
        futures = [ex.submit(intra_job, i + 1, m) for i, m in enumerate(all_models)]
        for fut in as_completed(futures):
            fut.result()

    # ---------------- 容错记录 ----------------
    if skipped_pairs or skipped_models:
        with skipped_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["kind", "model_a", "model_b", "reason"])
            writer.writeheader()
            for r in skipped_pairs:
                writer.writerow({"kind": "pair", **r})
            for r in skipped_models:
                writer.writerow({"kind": "intra", "model_a": r["model"], "model_b": "", "reason": r["reason"]})
        print(
            f"runner-skipped: {len(skipped_pairs)} pair(s), {len(skipped_models)} model(s) "
            f"(see {skipped_csv})",
            flush=True,
        )
    elif skipped_csv.is_file():
        skipped_csv.unlink()

    # ---------------- metadata ----------------
    meta = {
        "task": "task5_affine_relations",
        "sources_file": rel_to_repo(sources_file, repo_root),
        "extracts_dir": rel_to_repo(extracts_dir, repo_root),
        "models_yaml": rel_to_repo(models_yaml, repo_root),
        "n_pairs_unique": len(pairs),
        "n_models_intra": len(all_models),
        "max_fit_rows": max_fit_rows,
        "min_common_tokens": min_common_tokens,
        "devices": [str(d) for d in devices],
        "git_commit": try_git_commit(repo_root),
        "fit_method": "centered lstsq (Y - mY) ≈ (X - mX) · A; b = mY - mX · A",
    }
    meta_json.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\ndone: {out_dir}", flush=True)
