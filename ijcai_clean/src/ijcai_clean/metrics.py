"""
GPU 流式 GCorr（与 legacy run_exp1_v4.compute_gcorr_gpu_streaming_v4 数值路径一致）。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

# 与 V4 实验一致
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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


def compute_gcorr_gpu_streaming_v4(
    X: np.ndarray,
    Y: np.ndarray,
    n_pairs: int,
    seed: int | None = None,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> Dict[str, float]:
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n, d_x = X.shape
    _, d_y = Y.shape

    if batch_size is None:
        target_memory = 4 * 1024**3
        bytes_per_sample = 4 * d_x * 4 + 4 * d_y * 4
        batch_size = max(50_000, min(500_000, target_memory // max(bytes_per_sample, 1)))

    with torch.inference_mode():
        X_t = torch.from_numpy(X).float().to(device)
        Y_t = torch.from_numpy(Y).float().to(device)

        X_norm2 = (X_t * X_t).sum(dim=1)
        Y_norm2 = (Y_t * Y_t).sum(dim=1)
        X_norm = torch.sqrt(X_norm2.clamp(min=1e-20))
        Y_norm = torch.sqrt(Y_norm2.clamp(min=1e-20))

        i_indices, j_indices = _generate_pairs_gpu(n, n_pairs, seed if seed is not None else 42, device)
        num_pairs = len(i_indices)

        cos_sum_x = torch.tensor(0.0, dtype=torch.float64, device=device)
        cos_sum_y = torch.tensor(0.0, dtype=torch.float64, device=device)
        cos_sum_x2 = torch.tensor(0.0, dtype=torch.float64, device=device)
        cos_sum_y2 = torch.tensor(0.0, dtype=torch.float64, device=device)
        cos_sum_xy = torch.tensor(0.0, dtype=torch.float64, device=device)

        euc_sum_x = torch.tensor(0.0, dtype=torch.float64, device=device)
        euc_sum_y = torch.tensor(0.0, dtype=torch.float64, device=device)
        euc_sum_x2 = torch.tensor(0.0, dtype=torch.float64, device=device)
        euc_sum_y2 = torch.tensor(0.0, dtype=torch.float64, device=device)
        euc_sum_xy = torch.tensor(0.0, dtype=torch.float64, device=device)

        euc2_sum_x = torch.tensor(0.0, dtype=torch.float64, device=device)
        euc2_sum_y = torch.tensor(0.0, dtype=torch.float64, device=device)
        euc2_sum_x2 = torch.tensor(0.0, dtype=torch.float64, device=device)
        euc2_sum_y2 = torch.tensor(0.0, dtype=torch.float64, device=device)
        euc2_sum_xy = torch.tensor(0.0, dtype=torch.float64, device=device)

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

            cos_sum_x += stats[0]
            cos_sum_y += stats[1]
            cos_sum_x2 += stats[2]
            cos_sum_y2 += stats[3]
            cos_sum_xy += stats[4]
            euc_sum_x += stats[5]
            euc_sum_y += stats[6]
            euc_sum_x2 += stats[7]
            euc_sum_y2 += stats[8]
            euc_sum_xy += stats[9]
            euc2_sum_x += stats[10]
            euc2_sum_y += stats[11]
            euc2_sum_x2 += stats[12]
            euc2_sum_y2 += stats[13]
            euc2_sum_xy += stats[14]

        n_total = float(num_pairs)

        def pearson_from_sums(sum_x, sum_y, sum_x2, sum_y2, sum_xy, n):
            numerator = n * sum_xy - sum_x * sum_y
            denominator = torch.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
            return (numerator / denominator.clamp(min=1e-10)).item()

        gcorr_cos = pearson_from_sums(cos_sum_x, cos_sum_y, cos_sum_x2, cos_sum_y2, cos_sum_xy, n_total)
        gcorr_euc = pearson_from_sums(euc_sum_x, euc_sum_y, euc_sum_x2, euc_sum_y2, euc_sum_xy, n_total)
        gcorr_euc2 = pearson_from_sums(euc2_sum_x, euc2_sum_y, euc2_sum_x2, euc2_sum_y2, euc2_sum_xy, n_total)

    del X_t, Y_t, X_norm, Y_norm, X_norm2, Y_norm2, i_indices, j_indices

    return {"gcorr_cos": gcorr_cos, "gcorr_euc": gcorr_euc, "gcorr_euc2": gcorr_euc2}


def compute_single_pair_bootstrap(
    E_a: np.ndarray,
    E_b: np.ndarray,
    U_a: np.ndarray,
    U_b: np.ndarray,
    n_pairs: int,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    gcorr_E = compute_gcorr_gpu_streaming_v4(E_a, E_b, n_pairs, seed * 1000, device=device)
    gcorr_U = compute_gcorr_gpu_streaming_v4(U_a, U_b, n_pairs, seed * 1000 + 1, device=device)
    return {
        "gcorr_E_cos": gcorr_E["gcorr_cos"],
        "gcorr_E_euc": gcorr_E["gcorr_euc"],
        "gcorr_E_euc2": gcorr_E["gcorr_euc2"],
        "gcorr_U_cos": gcorr_U["gcorr_cos"],
        "gcorr_U_euc": gcorr_U["gcorr_euc"],
        "gcorr_U_euc2": gcorr_U["gcorr_euc2"],
    }
