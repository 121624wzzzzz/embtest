"""
GPU 流式 GCorr（与 legacy run_exp1_v4.compute_gcorr_gpu_streaming_v4 数值路径一致）。

实现优化：
- 单次流式累加可控显存预算（默认 32GB）；
- 单 GPU 内 E 与 U 双 CUDA stream 并行执行（仅当 device 当前空闲显存够 2 倍预算时启用）；
- 当并行启用时每个 stream 自动按一半预算划分 batch，叠加峰值仍接近原 32GB；
- 数值语义与原实现一致：仅累加分块和 stream 调度顺序变化，对 fp64 累加器的最终
  Pearson 结果只有可忽略的 ulp-level 差异。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEFAULT_TARGET_MEMORY_BYTES = 32 * 1024**3
MAX_BATCH_SIZE = 2_000_000
MIN_BATCH_SIZE = 50_000


def _generate_pairs_gpu(
    n: int, n_pairs: int, seed: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed if seed else 42)

    total_pairs = n * (n - 1) // 2
    if n_pairs >= total_pairs:
        i_indices = torch.arange(n, device=device).unsqueeze(1).expand(n, n)
        j_indices = torch.arange(n, device=device).unsqueeze(0).expand(n, n)
        mask = i_indices < j_indices
        return i_indices[mask], j_indices[mask]

    collected_i = []
    collected_j = []
    remaining = n_pairs

    while remaining > 0:
        batch = min(remaining * 3, 10_000_000)
        idx_i = torch.randint(0, n, (batch,), device=device, generator=generator)
        idx_j = torch.randint(0, n, (batch,), device=device, generator=generator)
        mask = idx_i < idx_j
        valid_i = idx_i[mask]
        valid_j = idx_j[mask]
        take = min(len(valid_i), remaining)
        collected_i.append(valid_i[:take])
        collected_j.append(valid_j[:take])
        remaining -= take

    i_indices = torch.cat(collected_i)
    j_indices = torch.cat(collected_j)
    sort_indices = torch.argsort(i_indices)
    return i_indices[sort_indices], j_indices[sort_indices]


def _compute_batch_stats(
    xi: torch.Tensor,
    xj: torch.Tensor,
    yi: torch.Tensor,
    yj: torch.Tensor,
    xi_norm: torch.Tensor,
    xj_norm: torch.Tensor,
    yi_norm: torch.Tensor,
    yj_norm: torch.Tensor,
    xi_norm2: torch.Tensor,
    xj_norm2: torch.Tensor,
    yi_norm2: torch.Tensor,
    yj_norm2: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    dot_x = (xi * xj).sum(dim=1)
    dot_y = (yi * yj).sum(dim=1)

    cos_x = (dot_x / (xi_norm * xj_norm).clamp(min=1e-10)).double()
    cos_y = (dot_y / (yi_norm * yj_norm).clamp(min=1e-10)).double()

    euc2_x = (xi_norm2 + xj_norm2 - 2 * dot_x).clamp(min=0).double()
    euc2_y = (yi_norm2 + yj_norm2 - 2 * dot_y).clamp(min=0).double()

    euc_x = torch.sqrt(euc2_x)
    euc_y = torch.sqrt(euc2_y)

    return (
        cos_x.sum(),
        cos_y.sum(),
        (cos_x * cos_x).sum(),
        (cos_y * cos_y).sum(),
        (cos_x * cos_y).sum(),
        euc_x.sum(),
        euc_y.sum(),
        (euc_x * euc_x).sum(),
        (euc_y * euc_y).sum(),
        (euc_x * euc_y).sum(),
        euc2_x.sum(),
        euc2_y.sum(),
        (euc2_x * euc2_x).sum(),
        (euc2_y * euc2_y).sum(),
        (euc2_x * euc2_y).sum(),
    )


_SUM_KEYS = (
    "cos_sum_x",
    "cos_sum_y",
    "cos_sum_x2",
    "cos_sum_y2",
    "cos_sum_xy",
    "euc_sum_x",
    "euc_sum_y",
    "euc_sum_x2",
    "euc_sum_y2",
    "euc_sum_xy",
    "euc2_sum_x",
    "euc2_sum_y",
    "euc2_sum_x2",
    "euc2_sum_y2",
    "euc2_sum_xy",
)


def _gcorr_streaming_accumulate(
    X: np.ndarray,
    Y: np.ndarray,
    n_pairs: int,
    seed: int,
    device: torch.device,
    target_memory_bytes: int,
    batch_size: Optional[int] = None,
) -> Dict[str, "torch.Tensor"]:
    """计算并返回 GPU 上的累加和（不调用 .item()，便于跨 stream 并行）。

    调用方需要保证当前线程的 CUDA stream 已经设置为期望的 stream，本函数内
    所有 kernel 启动都会落到该 stream 上。
    """

    n, d_x = X.shape
    _, d_y = Y.shape
    if batch_size is None:
        bytes_per_sample = 4 * d_x * 4 + 4 * d_y * 4
        batch_size = max(
            MIN_BATCH_SIZE,
            min(MAX_BATCH_SIZE, target_memory_bytes // max(bytes_per_sample, 1)),
        )

    with torch.inference_mode():
        X_t = torch.from_numpy(X).float().to(device, non_blocking=True)
        Y_t = torch.from_numpy(Y).float().to(device, non_blocking=True)

        X_norm2 = (X_t * X_t).sum(dim=1)
        Y_norm2 = (Y_t * Y_t).sum(dim=1)
        X_norm = torch.sqrt(X_norm2.clamp(min=1e-20))
        Y_norm = torch.sqrt(Y_norm2.clamp(min=1e-20))

        i_indices, j_indices = _generate_pairs_gpu(
            n, n_pairs, seed if seed is not None else 42, device
        )
        num_pairs = len(i_indices)

        sums: Dict[str, torch.Tensor] = {
            k: torch.tensor(0.0, dtype=torch.float64, device=device) for k in _SUM_KEYS
        }

        for start in range(0, num_pairs, batch_size):
            end = min(start + batch_size, num_pairs)
            i_batch = i_indices[start:end]
            j_batch = j_indices[start:end]

            xi, xj = X_t[i_batch], X_t[j_batch]
            yi, yj = Y_t[i_batch], Y_t[j_batch]

            xi_norm, xj_norm = X_norm[i_batch], X_norm[j_batch]
            yi_norm, yj_norm = Y_norm[i_batch], Y_norm[j_batch]
            xi_norm2, xj_norm2 = X_norm2[i_batch], X_norm2[j_batch]
            yi_norm2, yj_norm2 = Y_norm2[i_batch], Y_norm2[j_batch]

            stats = _compute_batch_stats(
                xi,
                xj,
                yi,
                yj,
                xi_norm,
                xj_norm,
                yi_norm,
                yj_norm,
                xi_norm2,
                xj_norm2,
                yi_norm2,
                yj_norm2,
            )

            for key, value in zip(_SUM_KEYS, stats):
                sums[key] += value

        sums["n_total"] = torch.tensor(float(num_pairs), dtype=torch.float64, device=device)
        del X_t, Y_t, X_norm, Y_norm, X_norm2, Y_norm2, i_indices, j_indices

    return sums


def _pearson_from_sums(sums: Dict[str, "torch.Tensor"]) -> Dict[str, float]:
    n = sums["n_total"]

    def pearson(sum_x, sum_y, sum_x2, sum_y2, sum_xy) -> float:
        numerator = n * sum_xy - sum_x * sum_y
        denominator = torch.sqrt(
            (n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)
        )
        return (numerator / denominator.clamp(min=1e-10)).item()

    return {
        "gcorr_cos": pearson(
            sums["cos_sum_x"],
            sums["cos_sum_y"],
            sums["cos_sum_x2"],
            sums["cos_sum_y2"],
            sums["cos_sum_xy"],
        ),
        "gcorr_euc": pearson(
            sums["euc_sum_x"],
            sums["euc_sum_y"],
            sums["euc_sum_x2"],
            sums["euc_sum_y2"],
            sums["euc_sum_xy"],
        ),
        "gcorr_euc2": pearson(
            sums["euc2_sum_x"],
            sums["euc2_sum_y"],
            sums["euc2_sum_x2"],
            sums["euc2_sum_y2"],
            sums["euc2_sum_xy"],
        ),
    }


def compute_gcorr_gpu_streaming_v4(
    X: np.ndarray,
    Y: np.ndarray,
    n_pairs: int,
    seed: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    target_memory_bytes: int = DEFAULT_TARGET_MEMORY_BYTES,
) -> Dict[str, float]:
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sums = _gcorr_streaming_accumulate(
        X,
        Y,
        n_pairs,
        seed if seed is not None else 42,
        device,
        target_memory_bytes=target_memory_bytes,
        batch_size=batch_size,
    )
    return _pearson_from_sums(sums)


def can_use_parallel_streams(
    device: torch.device, target_memory_bytes: int = DEFAULT_TARGET_MEMORY_BYTES
) -> bool:
    """检查 device 当前空闲显存能否同时驻留两路 GCorr 流（每路按 target/2 预算）。"""

    if device.type != "cuda":
        return False
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
    except Exception:
        return False
    return free_bytes >= 2 * target_memory_bytes + 1024**3


def compute_single_pair_bootstrap(
    E_a: np.ndarray,
    E_b: np.ndarray,
    U_a: np.ndarray,
    U_b: np.ndarray,
    n_pairs: int,
    seed: int,
    device: torch.device,
    target_memory_bytes: int = DEFAULT_TARGET_MEMORY_BYTES,
    parallel_streams: Optional[bool] = None,
) -> Dict[str, float]:
    if parallel_streams is None:
        parallel_streams = can_use_parallel_streams(device, target_memory_bytes)

    if device.type == "cuda" and parallel_streams:
        per_stream_target = target_memory_bytes // 2
        stream_E = torch.cuda.Stream(device=device)
        stream_U = torch.cuda.Stream(device=device)

        with torch.cuda.stream(stream_E):
            sums_E = _gcorr_streaming_accumulate(
                E_a,
                E_b,
                n_pairs,
                seed * 1000,
                device,
                target_memory_bytes=per_stream_target,
            )
        with torch.cuda.stream(stream_U):
            sums_U = _gcorr_streaming_accumulate(
                U_a,
                U_b,
                n_pairs,
                seed * 1000 + 1,
                device,
                target_memory_bytes=per_stream_target,
            )

        stream_E.synchronize()
        stream_U.synchronize()

        gcorr_E = _pearson_from_sums(sums_E)
        gcorr_U = _pearson_from_sums(sums_U)
    else:
        gcorr_E = compute_gcorr_gpu_streaming_v4(
            E_a,
            E_b,
            n_pairs,
            seed * 1000,
            device=device,
            target_memory_bytes=target_memory_bytes,
        )
        gcorr_U = compute_gcorr_gpu_streaming_v4(
            U_a,
            U_b,
            n_pairs,
            seed * 1000 + 1,
            device=device,
            target_memory_bytes=target_memory_bytes,
        )

    return {
        "gcorr_E_cos": gcorr_E["gcorr_cos"],
        "gcorr_E_euc": gcorr_E["gcorr_euc"],
        "gcorr_E_euc2": gcorr_E["gcorr_euc2"],
        "gcorr_U_cos": gcorr_U["gcorr_cos"],
        "gcorr_U_euc": gcorr_U["gcorr_euc"],
        "gcorr_U_euc2": gcorr_U["gcorr_euc2"],
    }
