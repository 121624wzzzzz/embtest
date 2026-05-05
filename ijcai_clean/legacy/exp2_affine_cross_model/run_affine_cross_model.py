#!/usr/bin/env python3
"""
实验2：同系列内跨模型 E-E / U-U 仿射关系检验
============================================
对同系列模型对 (A, B)，在公共 token 上检验：
- E_B ≈ A_E @ E_A + b_E  (E 与 E 仿射)
- U_B ≈ A_U @ U_A + b_U  (U 与 U 仿射)
报告 R²、相对误差等。
"""

import os
import sys
import glob
import gc
import itertools
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoConfig, AutoTokenizer

# 双卡 5090：仿射拟合在 GPU 上做，两卡并行
NUM_GPUS = min(2, torch.cuda.device_count()) if torch.cuda.is_available() else 0
DEVICES = [torch.device(f"cuda:{i}") for i in range(NUM_GPUS)] if NUM_GPUS > 0 else [torch.device("cpu")]
print(f"仿射拟合使用设备: {DEVICES if NUM_GPUS > 0 else 'cpu'}")

# =============================================================================
# 同系列模型列表（仅主系列，用于仿射实验）
# =============================================================================
FAMILIES = {
    "Qwen3": [
        "Qwen3-0.6B-Base", "Qwen3-0.6B",
        "Qwen3-1.7B-Base", "Qwen3-1.7B",
        "Qwen3-4B-Base", "Qwen3-4B",
        "Qwen3-8B-Base", "Qwen3-8B",
        "Qwen3-14B-Base", "Qwen3-14B",
        "Qwen3-32B",
    ],
    "Qwen2.5": [
        "Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct",
        "Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B", "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B", "Qwen2.5-7B-Instruct",
        "Qwen2.5-14B", "Qwen2.5-14B-Instruct",
        "Qwen2.5-32B", "Qwen2.5-32B-Instruct",
        "Qwen2.5-72B-Base", "Qwen2.5-72B-Instruct",
    ],
    "Llama": [
        "Llama-3.2-1B", "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B", "Llama-3.2-3B-Instruct",
        "Llama-3.1-8B", "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Base", "Llama-3.1-70B-Instruct",
    ],
    "Gemma2": [
        "Gemma-2-2B", "Gemma-2-2B-Instruct",
        "Gemma-2-9B", "Gemma-2-9B-Instruct",
        "Gemma-2-27B", "Gemma-2-27B-Instruct",
    ],
}

# 按规模分组（仅含 FAMILIES 中有的模型），用于跨系列同规模仿射检验
MODELS_BY_SIZE = {
    "~0.5-0.6B": ["Qwen3-0.6B-Base", "Qwen3-0.6B", "Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct"],
    "~1-2B": ["Qwen3-1.7B-Base", "Qwen3-1.7B", "Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct",
              "Llama-3.2-1B", "Llama-3.2-1B-Instruct", "Gemma-2-2B", "Gemma-2-2B-Instruct"],
    "~3-4B": ["Qwen3-4B-Base", "Qwen3-4B", "Qwen2.5-3B", "Qwen2.5-3B-Instruct",
              "Llama-3.2-3B", "Llama-3.2-3B-Instruct"],
    "~7-9B": ["Qwen3-8B-Base", "Qwen3-8B", "Qwen2.5-7B", "Qwen2.5-7B-Instruct",
              "Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Gemma-2-9B", "Gemma-2-9B-Instruct"],
    "~14B": ["Qwen3-14B-Base", "Qwen3-14B", "Qwen2.5-14B", "Qwen2.5-14B-Instruct"],
    "~27-32B": ["Qwen3-32B", "Qwen2.5-32B", "Qwen2.5-32B-Instruct",
                "Gemma-2-27B", "Gemma-2-27B-Instruct"],
    "~70-72B": ["Llama-3.1-70B-Base", "Llama-3.1-70B-Instruct",
                "Qwen2.5-72B-Base", "Qwen2.5-72B-Instruct"],
}
# 只保留在 FAMILIES 里出现的模型
_all_models = set()
for lst in FAMILIES.values():
    _all_models.update(lst)
for k in list(MODELS_BY_SIZE.keys()):
    MODELS_BY_SIZE[k] = [m for m in MODELS_BY_SIZE[k] if m in _all_models]
MODELS_BY_SIZE = {k: v for k, v in MODELS_BY_SIZE.items() if len(v) >= 2}

_model_to_family = {}
for fam, lst in FAMILIES.items():
    for m in lst:
        _model_to_family[m] = fam


@dataclass
class Config:
    model_base_path: str = "/root/shared-nvme/models"
    output_dir: str = "/root/shared-nvme/ijcai/exp2_affine_cross_model/results"
    min_common_tokens: int = 5000  # 公共 token 至少这么多才做仿射拟合
    # 拟合时最多用多少行：避免 72B 等大 d 时 design(n,d+1)+lstsq 工作区爆显存
    # n=10万、d=8192 时单次拟合需 ~10GB+；cap 到 24k 后约 ~2GB/次，双卡安全
    max_fit_rows: int = 24_000

    def get_model_path(self, model_name: str) -> str:
        return os.path.join(self.model_base_path, model_name)


# =============================================================================
# 模型加载（仅 E、U 矩阵 + tokenizer）
# =============================================================================
def load_model_matrices(model_path: str, model_name: str) -> Dict[str, Any]:
    from safetensors import safe_open

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    E, U = None, None
    embed_names = ["model.embed_tokens.weight", "transformer.wte.weight"]
    lm_head_names = ["lm_head.weight", "output.weight"]

    for sf_file in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            keys = f.keys()
            for name in embed_names:
                if name in keys and E is None:
                    E = f.get_tensor(name).float().numpy()
            for name in lm_head_names:
                if name in keys and U is None:
                    U = f.get_tensor(name).float().numpy()
            if E is not None and U is not None:
                break

    if E is None:
        raise ValueError(f"未找到输入嵌入层: {model_name}")
    if U is None:
        U = E.copy()

    vocab_size, hidden_dim = E.shape
    # 用 float32 存缓存，省内存；拟合时再拷到 GPU
    return {
        "E": E.astype(np.float32),
        "U": U.astype(np.float32),
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
    }


def get_common_token_ids(tokenizer_a, tokenizer_b) -> Tuple[np.ndarray, np.ndarray]:
    """按 token 字符串对齐，返回 (ids_a, ids_b)，长度相同，一一对应同一 token。"""
    def valid_ids(tok):
        v = set(range(tok.vocab_size))
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
            i = getattr(tok, attr, None)
            if i is not None:
                v.discard(i)
        if hasattr(tok, "all_special_ids"):
            v -= set(tok.all_special_ids)
        return v

    str_to_id_a = {}
    for i in valid_ids(tokenizer_a):
        try:
            s = tokenizer_a.convert_ids_to_tokens(int(i))
            if s:
                str_to_id_a[s] = i
        except Exception:
            pass
    str_to_id_b = {}
    for i in valid_ids(tokenizer_b):
        try:
            s = tokenizer_b.convert_ids_to_tokens(int(i))
            if s:
                str_to_id_b[s] = i
        except Exception:
            pass

    common = set(str_to_id_a.keys()) & set(str_to_id_b.keys())
    ids_a = np.array([str_to_id_a[s] for s in sorted(common)], dtype=np.int64)
    ids_b = np.array([str_to_id_b[s] for s in sorted(common)], dtype=np.int64)
    return ids_a, ids_b


def fit_affine_and_r2(
    X: np.ndarray, Y: np.ndarray, device: torch.device
) -> Tuple[float, float]:
    """
    拟合 Y ≈ X @ A^T + b（行向量）。在指定 GPU 上做最小二乘，算完即释放显存。
    返回 R², 相对 Frobenius 误差。
    """
    x = torch.from_numpy(X.astype(np.float32, copy=False)).to(device)
    y = torch.from_numpy(Y.astype(np.float32, copy=False)).to(device)
    n, d_x = x.shape
    ones = torch.ones((n, 1), dtype=x.dtype, device=device)
    design = torch.cat([x, ones], dim=1)
    beta = torch.linalg.lstsq(design, y, rcond=None).solution
    y_pred = design @ beta
    ss_res = (y - y_pred).pow(2).sum().item()
    y_centered = y - y.mean(dim=0)
    ss_tot = (y_centered.pow(2).sum()).item()
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    y_norm = (y.pow(2).sum()).item() ** 0.5
    rel_err = (ss_res ** 0.5) / (y_norm + 1e-20)
    del x, y, design, beta, y_pred, y_centered
    return r2, rel_err


def fit_affine_general(
    X: np.ndarray, Y: np.ndarray, device: torch.device
) -> Tuple[float, float, float, float]:
    """
    一般仿射变换：y = A x + b，中心化后先拟合线性部分 A，再得平移 b。
    X_c = X - mean(X), Y_c = Y - mean(Y)；Y_c = X_c @ A^T；b = mean(Y) - mean(X) @ A^T。
    返回 R², rel_err, ||A||_F, ||b||_2。
    """
    x = torch.from_numpy(X.astype(np.float32, copy=False)).to(device)
    y = torch.from_numpy(Y.astype(np.float32, copy=False)).to(device)
    mx = x.mean(dim=0)
    my = y.mean(dim=0)
    x_c = x - mx
    y_c = y - my
    # Y_c = X_c @ A^T  =>  A^T = lstsq(X_c, Y_c).solution
    a_t = torch.linalg.lstsq(x_c, y_c, rcond=None).solution  # (d_x, d_y)
    b = (my - mx @ a_t).squeeze()
    y_pred = x @ a_t + b
    ss_res = (y - y_pred).pow(2).sum().item()
    ss_tot = (y_c.pow(2).sum()).item()
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    y_norm = (y.pow(2).sum()).item() ** 0.5
    rel_err = (ss_res ** 0.5) / (y_norm + 1e-20)
    norm_a = a_t.T.norm("fro").item()
    norm_b = b.norm().item() if b.dim() > 0 else float(b)
    del x, y, x_c, y_c, a_t, b, y_pred
    return r2, rel_err, norm_a, norm_b


def run_pair(
    model_a: str,
    model_b: str,
    data_a: Dict[str, Any],
    data_b: Dict[str, Any],
    config: Config,
    device: torch.device,
) -> Optional[Dict]:
    """对一对模型 (A, B) 在指定 device 上做 E-E 和 U-U 仿射拟合，用毕即释显存。"""
    ids_a, ids_b = get_common_token_ids(data_a["tokenizer"], data_b["tokenizer"])
    n = len(ids_a)
    if n < config.min_common_tokens:
        return None

    # 拟合行数上限：避免大 d（如 72B 的 8192）时 design(n,d+1)+lstsq 工作区爆显存
    # 用确定性种子（不依赖 str 的 hash，Python 会随机化）保证复现
    seed = (sum(ord(c) for c in model_a + model_b) % (2**32))
    rng = np.random.default_rng(seed)
    if n > config.max_fit_rows:
        idx = rng.choice(n, size=config.max_fit_rows, replace=False)
        ids_a, ids_b = ids_a[idx], ids_b[idx]
        n_used = config.max_fit_rows
    else:
        n_used = n

    E_A = data_a["E"][ids_a]   # (n_used, d_a)
    E_B = data_b["E"][ids_b]
    U_A = data_a["U"][ids_a]
    U_B = data_b["U"][ids_b]
    d_a, d_b = data_a["hidden_dim"], data_b["hidden_dim"]

    try:
        # 一般仿射：y = A x + b，中心化拟合，并得到 ||A||_F、||b||_2
        r2_E, rel_err_E, norm_A_E, norm_b_E = fit_affine_general(E_A, E_B, device)
        r2_U, rel_err_U, norm_A_U, norm_b_U = fit_affine_general(U_A, U_B, device)
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    return {
        "model_a": model_a,
        "model_b": model_b,
        "n_common": n,
        "n_fit": n_used,
        "d_a": d_a,
        "d_b": d_b,
        "R2_E": r2_E,
        "R2_U": r2_U,
        "rel_err_E": rel_err_E,
        "rel_err_U": rel_err_U,
        "norm_A_E": norm_A_E,
        "norm_b_E": norm_b_E,
        "norm_A_U": norm_A_U,
        "norm_b_U": norm_b_U,
    }


def run_intra_model(
    model_name: str,
    data: Dict[str, Any],
    config: Config,
    device: torch.device,
) -> Dict[str, Any]:
    """模型内部 E 与 U：拟合 U ≈ A @ E + b（同一模型内，用 E 预测 U）。"""
    E = data["E"]
    U = data["U"]
    n, d = E.shape
    rng = np.random.default_rng(sum(ord(c) for c in model_name) % (2**32))
    if n > config.max_fit_rows:
        idx = rng.choice(n, size=config.max_fit_rows, replace=False)
        E = E[idx]
        U = U[idx]
        n_used = config.max_fit_rows
    else:
        n_used = n
    try:
        r2, rel_err, norm_A, norm_b = fit_affine_general(E, U, device)
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    return {
        "model": model_name,
        "family": _model_to_family.get(model_name, "Unknown"),
        "d": d,
        "n_fit": n_used,
        "R2_EU": r2,
        "rel_err_EU": rel_err,
        "norm_A_EU": norm_A,
        "norm_b_EU": norm_b,
    }


def _run_one(args: Tuple) -> Optional[Dict]:
    """供线程池调用：单对 (model_a, model_b, data_a, data_b, config, device, family)。"""
    model_a, model_b, data_a, data_b, config, device, family = args
    try:
        res = run_pair(model_a, model_b, data_a, data_b, config, device)
        if res is not None:
            res["family"] = family
        return res
    except Exception as e:
        print(f"  [{family}] {model_a} vs {model_b}: {e}")
        return None


def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    cache: Dict[str, Dict[str, Any]] = {}
    n_workers = max(1, NUM_GPUS)

    # 先按系列加载全部模型到 CPU 缓存，避免多线程读盘
    print("预加载模型矩阵 (CPU)...")
    for family, models in FAMILIES.items():
        for name in models:
            if name in cache:
                continue
            try:
                cache[name] = load_model_matrices(config.get_model_path(name), name)
            except Exception as e:
                print(f"  加载失败 {name}: {e}")
    gc.collect()

    # -------------------------------------------------------------------------
    # 模型内部 E 与 U：U ≈ A @ E + b（同一模型内）
    # -------------------------------------------------------------------------
    print("\n模型内部 E–U 仿射拟合...")
    intra_rows = []
    for i, (name, data) in enumerate(cache.items()):
        device_i = DEVICES[i % n_workers]
        try:
            res = run_intra_model(name, data, config, device_i)
            intra_rows.append(res)
        except Exception as e:
            print(f"  [{name}] {e}")
    if intra_rows:
        df_intra = pd.DataFrame(intra_rows)
        df_intra = df_intra.sort_values(["family", "model"]).reset_index(drop=True)
        intra_path = os.path.join(config.output_dir, "affine_intra_model_EU_results.csv")
        df_intra.to_csv(intra_path, index=False)
        print(f"已写入 {intra_path}，共 {len(df_intra)} 个模型")
        intra_summary = df_intra.groupby("family").agg({
            "R2_EU": ["mean", "min", "max"],
            "rel_err_EU": "mean",
            "norm_A_EU": "mean",
            "norm_b_EU": "mean",
        }).round(6)
        intra_summary.columns = ["R2_EU_mean", "R2_EU_min", "R2_EU_max", "rel_err_EU_mean", "norm_A_EU_mean", "norm_b_EU_mean"]
        intra_summary.to_csv(os.path.join(config.output_dir, "affine_intra_model_EU_summary.csv"))
        print("--- 模型内部 E→U 仿射 R²_EU ---")
        print(intra_summary[["R2_EU_mean", "R2_EU_min", "R2_EU_max"]].to_string())
    if NUM_GPUS > 0:
        torch.cuda.empty_cache()
        gc.collect()

    # 构造所有 (model_a, model_b, data_a, data_b, config, device, family)，全局轮询双卡
    jobs: List[Tuple] = []
    for family, models in FAMILIES.items():
        if len(models) < 2:
            continue
        pairs = list(itertools.combinations(models, 2))
        for model_a, model_b in pairs:
            if model_a not in cache or model_b not in cache:
                continue
            device = DEVICES[len(jobs) % n_workers]
            jobs.append((model_a, model_b, cache[model_a], cache[model_b], config, device, family))

    rows = []
    if n_workers >= 2:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_run_one, j): j for j in jobs}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="仿射拟合"):
                res = fut.result()
                if res is not None:
                    rows.append(res)
    else:
        for j in tqdm(jobs, desc="仿射拟合"):
            res = _run_one(j)
            if res is not None:
                rows.append(res)

    if NUM_GPUS > 0:
        torch.cuda.empty_cache()
        gc.collect()

    # 按 family, model_a, model_b 排序，便于查看
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["family", "model_a", "model_b"]).reset_index(drop=True)
    if df.empty:
        print("没有有效结果")
        return

    out_path = os.path.join(config.output_dir, "affine_cross_model_results.csv")
    df.to_csv(out_path, index=False)
    print(f"已写入 {out_path}，共 {len(df)} 对")

    # 按系列汇总：R²、相对误差、一般仿射的 ||A||_F 与 ||b||_2
    summary = df.groupby("family").agg({
        "R2_E": ["mean", "min", "max"],
        "R2_U": ["mean", "min", "max"],
        "rel_err_E": "mean",
        "rel_err_U": "mean",
        "norm_A_E": "mean",
        "norm_b_E": "mean",
        "norm_A_U": "mean",
        "norm_b_U": "mean",
    }).round(6)
    summary.columns = [
        "R2_E_mean", "R2_E_min", "R2_E_max",
        "R2_U_mean", "R2_U_min", "R2_U_max",
        "rel_err_E_mean", "rel_err_U_mean",
        "norm_A_E_mean", "norm_b_E_mean", "norm_A_U_mean", "norm_b_U_mean",
    ]
    summary_path = os.path.join(config.output_dir, "affine_cross_model_summary.csv")
    summary.to_csv(summary_path)
    print(f"汇总已写入 {summary_path}")

    # 分别打印 E 与 U 的结果
    print("\n--- R²_E（E 与 E 一般仿射拟合）---")
    print(summary[["R2_E_mean", "R2_E_min", "R2_E_max"]].to_string())
    print("\n--- R²_U（U 与 U 一般仿射拟合）---")
    print(summary[["R2_U_mean", "R2_U_min", "R2_U_max"]].to_string())
    print("\n--- 相对误差 ---")
    print(summary[["rel_err_E_mean", "rel_err_U_mean"]].to_string())
    print("\n--- 一般仿射：||A||_F、||b||_2 均值 ---")
    print(summary[["norm_A_E_mean", "norm_b_E_mean", "norm_A_U_mean", "norm_b_U_mean"]].to_string())

    # -------------------------------------------------------------------------
    # 跨系列同规模：同参数量级、不同系列（如 Qwen3-8B vs Qwen2.5-7B vs Llama-8B）
    # -------------------------------------------------------------------------
    cross_jobs: List[Tuple] = []
    for size_bucket, model_list in MODELS_BY_SIZE.items():
        for (model_a, model_b) in itertools.combinations(model_list, 2):
            if _model_to_family.get(model_a) == _model_to_family.get(model_b):
                continue
            if model_a not in cache or model_b not in cache:
                continue
            device = DEVICES[len(cross_jobs) % n_workers]
            cross_jobs.append((
                model_a, model_b, cache[model_a], cache[model_b], config, device,
                "cross_" + size_bucket,
            ))

    if not cross_jobs:
        print("\n无跨系列同规模对（或规模组内只有单系列），跳过。")
        return

    print(f"\n跨系列同规模：共 {len(cross_jobs)} 对，开始仿射拟合...")
    cross_rows = []
    if n_workers >= 2:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_run_one, j): j for j in cross_jobs}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="跨系列同规模"):
                res = fut.result()
                if res is not None:
                    cross_rows.append(res)
    else:
        for j in tqdm(cross_jobs, desc="跨系列同规模"):
            res = _run_one(j)
            if res is not None:
                cross_rows.append(res)

    if NUM_GPUS > 0:
        torch.cuda.empty_cache()
        gc.collect()

    if cross_rows:
        df_cross = pd.DataFrame(cross_rows)
        df_cross = df_cross.sort_values(["family", "model_a", "model_b"]).reset_index(drop=True)
        cross_path = os.path.join(config.output_dir, "affine_cross_family_same_scale_results.csv")
        df_cross.to_csv(cross_path, index=False)
        print(f"已写入 {cross_path}，共 {len(df_cross)} 对")

        # 按规模桶汇总
        cross_summary = df_cross.groupby("family").agg({
            "R2_E": ["mean", "min", "max"],
            "R2_U": ["mean", "min", "max"],
            "rel_err_E": "mean",
            "rel_err_U": "mean",
        }).round(6)
        cross_summary.columns = ["R2_E_mean", "R2_E_min", "R2_E_max", "R2_U_mean", "R2_U_min", "R2_U_max",
                                 "rel_err_E_mean", "rel_err_U_mean"]
        cross_summary_path = os.path.join(config.output_dir, "affine_cross_family_same_scale_summary.csv")
        cross_summary.to_csv(cross_summary_path)
        print(f"跨系列同规模汇总已写入 {cross_summary_path}")
        print("\n--- 跨系列同规模 R²_E / R²_U ---")
        print(cross_summary[["R2_E_mean", "R2_E_min", "R2_E_max", "R2_U_mean", "R2_U_min", "R2_U_max"]].to_string())
        print("\n--- 跨系列同规模 rel_err ---")
        print(cross_summary[["rel_err_E_mean", "rel_err_U_mean"]].to_string())
    else:
        print("跨系列同规模无有效结果。")


if __name__ == "__main__":
    main()
