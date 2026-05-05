from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml
from safetensors import safe_open


def find_model_cache_dir(
    repo_root: Path,
    model_name: str,
    models_yaml: Path,
) -> Optional[Path]:
    if not models_yaml.is_file():
        return None
    cfg = yaml.safe_load(models_yaml.read_text(encoding="utf-8"))
    repo_ids: Dict[str, str] = cfg.get("model_repo_ids") or {}
    rid = repo_ids.get(model_name)
    if not rid:
        return None
    owner, repo = rid.split("/", 1)
    cache = repo_root / "downloaded_models" / owner / repo
    if cache.is_dir():
        return cache
    alt = repo.replace(".", "___")
    cache2 = repo_root / "downloaded_models" / owner / alt
    if cache2.is_dir():
        return cache2.resolve() if cache2.is_symlink() else cache2
    return None


def load_info_json(extracts_dir: Path, model_name: str) -> Dict[str, Any]:
    p = extracts_dir / f"{model_name}.info.json"
    if not p.is_file():
        raise FileNotFoundError(f"缺少元数据: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_E_U_matrices(
    extracts_dir: Path,
    model_name: str,
    info: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    从 extracts 加载 E（嵌入）与 U（解嵌/lm_head）float32 矩阵。
    键名优先使用 info['standardized_sources']。
    """
    info = info or load_info_json(extracts_dir, model_name)
    st_path = extracts_dir / f"{model_name}.safetensors"
    if not st_path.is_file():
        raise FileNotFoundError(f"缺少矩阵文件: {st_path}")

    src = info.get("standardized_sources") or {}
    embed_key = src.get("embed") or "model.embed_tokens.weight"
    head_key = src.get("lm_head") or "lm_head.weight"

    with safe_open(st_path, framework="pt", device="cpu") as f:
        if embed_key not in f.keys():
            raise KeyError(f"{model_name}: safetensors 中无 embed 键 {embed_key}, 现有: {list(f.keys())}")
        if head_key not in f.keys():
            raise KeyError(f"{model_name}: safetensors 中无 lm_head 键 {head_key}")
        E = f.get_tensor(embed_key).float().cpu().numpy()
        U = f.get_tensor(head_key).float().cpu().numpy()

    return E, U, info


def load_E_U_matrix_rows(
    extracts_dir: Path,
    model_name: str,
    row_ids: np.ndarray,
    info: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    从 extracts 中只加载指定 token 行，用于 smoke/validation。

    注意：正式实验仍可加载完整矩阵；这个函数避免为了小规模验证读取巨大的
    70B/72B E/U 全矩阵。
    """
    info = info or load_info_json(extracts_dir, model_name)
    st_path = extracts_dir / f"{model_name}.safetensors"
    if not st_path.is_file():
        raise FileNotFoundError(f"缺少矩阵文件: {st_path}")

    src = info.get("standardized_sources") or {}
    embed_key = src.get("embed") or "model.embed_tokens.weight"
    head_key = src.get("lm_head") or "lm_head.weight"
    ids = np.asarray(row_ids, dtype=np.int64)

    with safe_open(st_path, framework="pt", device="cpu") as f:
        E_slice = f.get_slice(embed_key)
        U_slice = f.get_slice(head_key)
        E = np.stack([E_slice[int(i)].float().cpu().numpy() for i in ids], axis=0)
        U = np.stack([U_slice[int(i)].float().cpu().numpy() for i in ids], axis=0)

    return E, U, info


def actual_tied(E: np.ndarray, U: np.ndarray) -> bool:
    return bool(np.allclose(E, U, rtol=1e-5, atol=1e-5))


def model_dims_and_vocab(info: Dict[str, Any]) -> Tuple[int, int, int]:
    dims = info.get("standardized_dims") or {}
    emb = dims.get("embed")
    if not emb or len(emb) != 2:
        raise ValueError("standardized_dims.embed 无效")
    vocab_size, hidden_dim = int(emb[0]), int(emb[1])
    return vocab_size, hidden_dim, vocab_size
