from __future__ import annotations

import gc
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch

from ijcai_clean.data import load_E_U_matrix_rows, load_info_json
from ijcai_clean.metrics import compute_single_pair_bootstrap


def validate_completed_pairs_compute(
    *,
    extracts_dir: Path,
    pairs: Sequence[Tuple[str, str]],
    validation_n_tokens: int,
    validation_n_pairs: int,
    random_seed: int,
    devices: Sequence[torch.device],
) -> int:
    """
    在已有完整 bootstrap 结果时，仍逐 pair 加载 E/U 并跑一次小规模 GCorr。

    这个路径用于验证“当前代码确实能加载并计算这些矩阵”，但不重复写入
    bootstrap_results.csv，避免覆盖正式实验结果。
    """
    n_ok = 0
    n_devices = max(len(devices), 1)
    for idx, (model_a, model_b) in enumerate(pairs, start=1):
        device = devices[(idx - 1) % n_devices]
        print(
            f"[validate {idx}/{len(pairs)}] {model_a} -> {model_b} "
            f"on {device}",
            flush=True,
        )
        info_a = load_info_json(extracts_dir, model_a)
        info_b = load_info_json(extracts_dir, model_b)
        vocab_a = int(info_a["standardized_dims"]["embed"][0])
        vocab_b = int(info_b["standardized_dims"]["embed"][0])

        n_sample = min(validation_n_tokens, vocab_a, vocab_b)
        if n_sample < 2:
            raise ValueError(f"{model_a}->{model_b}: 可采样 token 数过少: {n_sample}")
        rng = np.random.default_rng(random_seed + idx)
        ids = rng.choice(min(vocab_a, vocab_b), size=n_sample, replace=False)
        E_a, U_a, _ = load_E_U_matrix_rows(extracts_dir, model_a, ids, info=info_a)
        E_b, U_b, _ = load_E_U_matrix_rows(extracts_dir, model_b, ids, info=info_b)

        result = compute_single_pair_bootstrap(
            E_a,
            E_b,
            U_a,
            U_b,
            validation_n_pairs,
            random_seed + idx,
            device,
        )
        print(
            f"  ok: E_euc={result['gcorr_E_euc']:.6f}, "
            f"U_euc={result['gcorr_U_euc']:.6f}",
            flush=True,
        )
        n_ok += 1

        del E_a, E_b, U_a, U_b, result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return n_ok
