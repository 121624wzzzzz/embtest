from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class TokenSampler:
    """与 legacy run_exp1_v4.TokenSampler 对齐的采样逻辑。"""

    def get_valid_token_ids(self, tokenizer) -> np.ndarray:
        vocab_size = tokenizer.vocab_size
        all_ids = set(range(vocab_size))
        special_ids: set = set()

        for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
            token_id = getattr(tokenizer, attr, None)
            if token_id is not None:
                special_ids.add(token_id)

        if hasattr(tokenizer, "all_special_ids"):
            special_ids.update(tokenizer.all_special_ids)

        return np.array(sorted(all_ids - special_ids), dtype=np.int64)

    def sample_tokens(self, valid_ids: np.ndarray, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if len(valid_ids) <= n:
            return valid_ids.copy()
        return rng.choice(valid_ids, size=n, replace=False)

    def get_common_tokens(self, tokenizer1, tokenizer2) -> Tuple[np.ndarray, np.ndarray]:
        valid1 = self.get_valid_token_ids(tokenizer1)
        valid2 = self.get_valid_token_ids(tokenizer2)

        str_to_id1: Dict[str, Any] = {}
        str_to_id2: Dict[str, Any] = {}
        for tid in valid1:
            try:
                s = tokenizer1.convert_ids_to_tokens(int(tid))
                if s:
                    str_to_id1[s] = tid
            except Exception:
                pass
        for tid in valid2:
            try:
                s = tokenizer2.convert_ids_to_tokens(int(tid))
                if s:
                    str_to_id2[s] = tid
            except Exception:
                pass

        common = sorted(set(str_to_id1.keys()) & set(str_to_id2.keys()))
        return (
            np.array([str_to_id1[s] for s in common], dtype=np.int64),
            np.array([str_to_id2[s] for s in common], dtype=np.int64),
        )


def build_pair_token_info(
    *,
    vocab_a: int,
    vocab_b: int,
    tokenizer_a,
    tokenizer_b,
    n_tokens: int,
    sampler: TokenSampler,
) -> Optional[Dict[str, Any]]:
    """返回 align_mode、采样用 ids、n_sample；不可对齐时返回 None。"""
    if vocab_a == vocab_b and tokenizer_a is not None:
        valid_ids = sampler.get_valid_token_ids(tokenizer_a)
        return {
            "align_mode": "id",
            "same_tokenizer": True,
            "ids_a": valid_ids,
            "ids_b": valid_ids,
            "n_sample": min(n_tokens, len(valid_ids)),
        }

    if tokenizer_a is None or tokenizer_b is None:
        raise ValueError("词表不一致或缺少 tokenizer 缓存目录，无法进行跨词表对齐")

    ids_a, ids_b = sampler.get_common_tokens(tokenizer_a, tokenizer_b)
    if len(ids_a) < 1000:
        return None

    return {
        "align_mode": "string",
        "same_tokenizer": False,
        "ids_a": ids_a,
        "ids_b": ids_b,
        "n_sample": min(n_tokens, len(ids_a)),
    }
