#!/usr/bin/env python3 
"""
实验1 v4：全局几何相似性验证 (Global Geometry) - 多卡 40/50 系列优化版
=====================================================================
相比 v3 的优化（原先假定 RTX 5090 双卡 + 1TB 内存环境，现在自动探测 GPU 数量，可在 8×4090 等多卡环境下工作）：
1. 纯 GPU 随机采样：移除 CPU 索引计算，使用 GPU 拒绝采样生成 pair 索引
2. torch.compile：JIT 编译核心计算函数，融合 CUDA Kernel
3. TF32 加速：启用 Tensor Core TF32 模式（适用于 30/40/50 系列）
4. 大 Batch Size：从 200K 提升到 1M，充分利用单卡 24–32GB 显存
5. 移除频繁 empty_cache：减少 GPU 同步开销
6. Index Sorting：按 i_indices 排序，提升显存访问局部性（L2 Cache 命中率）

核心性能优化：
- 欧氏距离：用 ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x·y，消除 diff 大张量
- 余弦相似度：不构造归一化矩阵，用 1D norm 向量 + dot/(norm_i*norm_j)
- GPU 原生随机数生成 + 拒绝采样，避免 CPU-GPU 传输瓶颈
- torch.inference_mode() + torch.compile 减少开销
- 索引排序使显存访问接近顺序读取，提升带宽利用率

指标：
- GCorr_cos: 余弦相似度矩阵的相关系数
- GCorr_euc: 欧氏距离矩阵的相关系数
- GCorr_euc2: 平方欧氏距离矩阵的相关系数
"""

import os
import sys
import json
import time
import warnings
import itertools
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM

# =============================================================================
# GPU 配置 & 性能优化
# =============================================================================
NUM_GPUS = torch.cuda.device_count()
print(f"检测到 {NUM_GPUS} 个 GPU")
DEVICES = [torch.device(f"cuda:{i}") for i in range(NUM_GPUS)] if NUM_GPUS > 0 else [torch.device("cpu")]
DEVICE = DEVICES[0]

# 启用 TF32 加速（RTX 30xx/40xx/50xx 支持）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("TF32 加速已启用")

# 线程锁（用于多线程安全）
GPU_LOCKS = [threading.Lock() for _ in range(max(NUM_GPUS, 1))]


# =============================================================================
# tools 模型配置集成：统一使用 /root/shared-nvme/tools/models.yaml
# =============================================================================
TOOLS_MODELS_YAML = "/root/shared-nvme/tools/models.yaml"


def _load_tools_model_index() -> Dict[str, Dict[str, str]]:
    """
    读取 tools/models.yaml，建立:
      model_name -> { 'repo_root': <绝对路径>, 'emb_file': <extracted_embeddings 路径> }
    若文件不存在或解析失败，返回空字典，不影响原有逻辑。
    """
    index: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(TOOLS_MODELS_YAML):
        return index
    try:
        with open(TOOLS_MODELS_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return index

    config_block = data.get("config", {}) or {}
    cache_dir = config_block.get("cache_dir", "./downloaded_models")
    cache_dir_abs = os.path.abspath(
        os.path.join(os.path.dirname(TOOLS_MODELS_YAML), cache_dir)
    )

    model_repo_ids = data.get("model_repo_ids", {}) or {}
    for model_name, repo_id in model_repo_ids.items():
        # 形如 creator/model_dir
        if "/" not in repo_id:
            continue
        creator, model_dir = repo_id.split("/", 1)
        repo_root = os.path.join(cache_dir_abs, creator, model_dir)
        emb_file = os.path.join(repo_root, "extracted_embeddings.safetensors")
        index[model_name] = {
            "repo_root": repo_root,
            "emb_file": emb_file,
        }
    return index


TOOLS_MODEL_INDEX: Dict[str, Dict[str, str]] = _load_tools_model_index()


# =============================================================================
# 完整模型列表 (v4 更新)
# =============================================================================
ALL_MODELS = {
    "Qwen3": [
        "Qwen3-0.6B-Base", "Qwen3-0.6B",
        "Qwen3-1.7B-Base", "Qwen3-1.7B",
        "Qwen3-4B-Base", "Qwen3-4B",
        "Qwen3-8B-Base", "Qwen3-8B",
        "Qwen3-14B-Base", "Qwen3-14B",
        "Qwen3-32B",
    ],
    "Qwen3-MoE": [
        "Qwen3-30B-A3B",
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
    # Qwen2.5 分组（用于内存优化）
    "Qwen2.5-small": [  # 0.5B-7B (10 模型)
        "Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct",
        "Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B", "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B", "Qwen2.5-7B-Instruct",
        "Qwen2.5-14B", "Qwen2.5-14B-Instruct",
    ],
    "Qwen2.5-large": [  # 32B-72B (4 模型)
        "Qwen2.5-32B", "Qwen2.5-32B-Instruct",
        "Qwen2.5-72B-Base", "Qwen2.5-72B-Instruct",
    ],
    "Llama": [
        "Llama-3.2-1B", "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B", "Llama-3.2-3B-Instruct",
        "Llama-3.1-8B", "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Base", "Llama-3.1-70B-Instruct",
    ],
    # Llama 分组
    "Llama-small": [  # 1B-8B (6 模型)
        "Llama-3.2-1B", "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B", "Llama-3.2-3B-Instruct",
        "Llama-3.1-8B", "Llama-3.1-8B-Instruct",
    ],
    "Llama-large": [  # 70B (2 模型)
        "Llama-3.1-70B-Base", "Llama-3.1-70B-Instruct",
    ],
    "Gemma2": [
        "Gemma-2-2B", "Gemma-2-2B-Instruct",
        "Gemma-2-9B", "Gemma-2-9B-Instruct",
        "Gemma-2-27B", "Gemma-2-27B-Instruct",
    ],
    "Mistral": ["Mistral-7B-v0.3"],
    "Yi": ["Yi-1.5-9B"],
    "DeepSeek": ["DeepSeek-V2-Lite-Chat"],
}

# 同尺寸 Base-Instruct 配对（实验 1.1）
BASE_INSTRUCT_PAIRS = [
    # Qwen3 (5对)
    ("Qwen3-0.6B-Base", "Qwen3-0.6B"),
    ("Qwen3-1.7B-Base", "Qwen3-1.7B"),
    ("Qwen3-4B-Base", "Qwen3-4B"),
    ("Qwen3-8B-Base", "Qwen3-8B"),
    ("Qwen3-14B-Base", "Qwen3-14B"),
    # Qwen2.5 (7对)
    ("Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct"),
    ("Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct"),
    ("Qwen2.5-3B", "Qwen2.5-3B-Instruct"),
    ("Qwen2.5-7B", "Qwen2.5-7B-Instruct"),
    ("Qwen2.5-14B", "Qwen2.5-14B-Instruct"),
    ("Qwen2.5-32B", "Qwen2.5-32B-Instruct"),
    ("Qwen2.5-72B-Base", "Qwen2.5-72B-Instruct"),
    # Llama (4对)
    ("Llama-3.2-1B", "Llama-3.2-1B-Instruct"),
    ("Llama-3.2-3B", "Llama-3.2-3B-Instruct"),
    ("Llama-3.1-8B", "Llama-3.1-8B-Instruct"),
    ("Llama-3.1-70B-Base", "Llama-3.1-70B-Instruct"),
    # Gemma 2 (3对)
    ("Gemma-2-2B", "Gemma-2-2B-Instruct"),
    ("Gemma-2-9B", "Gemma-2-9B-Instruct"),
    ("Gemma-2-27B", "Gemma-2-27B-Instruct"),
]

# 同尺寸 Base-Instruct 配对集合（用于实验 1.2 排除）
BASE_INSTRUCT_PAIRS_SET = {tuple(sorted(pair)) for pair in BASE_INSTRUCT_PAIRS}

# 按规模分组（用于跨系列对比）
MODELS_BY_SIZE = {
    "~0.5-0.6B": ["Qwen3-0.6B-Base", "Qwen3-0.6B", "Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct"],
    "~1-2B": ["Qwen3-1.7B-Base", "Qwen3-1.7B", "Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct", 
              "Llama-3.2-1B", "Llama-3.2-1B-Instruct",
              "Gemma-2-2B", "Gemma-2-2B-Instruct"],
    "~3-4B": ["Qwen3-4B-Base", "Qwen3-4B", "Qwen2.5-3B", "Qwen2.5-3B-Instruct",
              "Llama-3.2-3B", "Llama-3.2-3B-Instruct"],
    "~7-9B": ["Qwen3-8B-Base", "Qwen3-8B", "Qwen2.5-7B", "Qwen2.5-7B-Instruct",
              "Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Mistral-7B-v0.3", "Yi-1.5-9B",
              "Gemma-2-9B", "Gemma-2-9B-Instruct"],
    "~14B": ["Qwen3-14B-Base", "Qwen3-14B", "Qwen2.5-14B", "Qwen2.5-14B-Instruct"],
    "~27-32B": ["Qwen3-32B", "Qwen2.5-32B", "Qwen2.5-32B-Instruct",
                "Gemma-2-27B", "Gemma-2-27B-Instruct"],
    "~70-72B": ["Llama-3.1-70B-Base", "Llama-3.1-70B-Instruct",
                "Qwen2.5-72B-Base", "Qwen2.5-72B-Instruct"],
}


# =============================================================================
# 配置
# =============================================================================
@dataclass
class Config:
    model_base_path: str = "/root/shared-nvme/models"
    output_dir: str = "/root/shared-nvme/ijcai/exp1_global_geometry/results"
    n_tokens: int = 20000
    n_bootstrap: int = 100
    n_pairs: int = 5_000_000
    random_seed: int = 42
    
    def get_model_path(self, model_name: str) -> str:
        """
        统一的模型路径解析策略：
        1. 若 tools/models.yaml 中存在该模型，优先返回对应 repo_root
        2. 否则退回到原始的 /root/shared-nvme/models/<model_name>
        """
        info = TOOLS_MODEL_INDEX.get(model_name)
        if info:
            repo_root = info.get("repo_root")
            if repo_root and os.path.isdir(repo_root):
                return repo_root
        return os.path.join(self.model_base_path, model_name)


# =============================================================================
# 模型加载器
# =============================================================================
class ModelLoader:
    def __init__(self, config: Config):
        self.config = config
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def load(self, model_name: str) -> Dict[str, Any]:
        if model_name in self.cache:
            return self.cache[model_name]
        
        model_path = self.config.get_model_path(model_name)
        print(f"  加载: {model_name}", end=" ... ")
        
        try:
            result = self._load_from_safetensors(model_path, model_name)
        except Exception as e:
            print(f"safetensors失败, 回退transformers")
            result = self._load_from_transformers(model_path, model_name)
        
        self.cache[model_name] = result
        print(f"vocab={result['vocab_size']}, dim={result['hidden_dim']}, tied={result['actual_tied']}")
        return result
    
    def _load_from_safetensors(self, model_path: str, model_name: str) -> Dict[str, Any]:
        from safetensors import safe_open
        import glob
        
        # 先根据 tools 索引判断是否存在专门的 extracted_embeddings.safetensors
        tools_info = TOOLS_MODEL_INDEX.get(model_name)
        emb_file_from_tools = None
        if tools_info:
            emb_candidate = tools_info.get("emb_file")
            if emb_candidate and os.path.isfile(emb_candidate):
                emb_file_from_tools = emb_candidate
        
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        is_tied = getattr(config, 'tie_word_embeddings', True)
        
        E, U = None, None
        embed_names = ["model.embed_tokens.weight", "transformer.wte.weight"]
        lm_head_names = ["lm_head.weight", "output.weight"]

        # 分两种来源：
        # 1) 若 tools 提供了 extracted_embeddings.safetensors，则只读这一份
        # 2) 否则回退到扫描当前目录下的 *.safetensors 分片
        safetensor_files: List[str] = []
        if emb_file_from_tools is not None:
            safetensor_files = [emb_file_from_tools]
        else:
            safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
            if not safetensor_files:
                raise FileNotFoundError("No safetensors files found")
        
        for sf_file in safetensor_files:
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
            raise ValueError("未找到输入嵌入层")
        if U is None:
            U = E.copy()
        
        actual_tied = np.allclose(E, U, rtol=1e-5, atol=1e-5)
        vocab_size, hidden_dim = E.shape
        
        family = "Unknown"
        for fam, models in ALL_MODELS.items():
            if model_name in models:
                family = fam
                break
        
        return {
            'E': E, 'U': U, 'is_tied': is_tied, 'actual_tied': actual_tied,
            'vocab_size': vocab_size, 'hidden_dim': hidden_dim,
            'tokenizer': tokenizer, 'config': config, 'model_name': model_name,
            'family': family,
        }
    
    def _load_from_transformers(self, model_path: str, model_name: str) -> Dict[str, Any]:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
        E = model.embed_tokens.weight.detach().numpy().astype(np.float32)
        
        is_tied = getattr(config, 'tie_word_embeddings', True)
        
        full_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
        U = full_model.lm_head.weight.detach().numpy().astype(np.float32) if hasattr(full_model, 'lm_head') else E.copy()
        
        actual_tied = np.allclose(E, U, rtol=1e-5, atol=1e-5)
        vocab_size, hidden_dim = E.shape
        
        del model, full_model
        # 注意：移除了 empty_cache，让 PyTorch 缓存分配器自动管理
        
        family = "Unknown"
        for fam, models in ALL_MODELS.items():
            if model_name in models:
                family = fam
                break
        
        return {
            'E': E, 'U': U, 'is_tied': is_tied, 'actual_tied': actual_tied,
            'vocab_size': vocab_size, 'hidden_dim': hidden_dim,
            'tokenizer': tokenizer, 'config': config, 'model_name': model_name,
            'family': family,
        }
    
    def clear_cache(self):
        """清理模型缓存并强制释放 GPU 显存"""
        self.cache.clear()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# =============================================================================
# Token 采样器
# =============================================================================
class TokenSampler:
    def __init__(self, config: Config):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
    
    def get_valid_token_ids(self, tokenizer) -> np.ndarray:
        vocab_size = tokenizer.vocab_size
        all_ids = set(range(vocab_size))
        special_ids = set()
        
        for attr in ['bos_token_id', 'eos_token_id', 'pad_token_id', 'unk_token_id']:
            token_id = getattr(tokenizer, attr, None)
            if token_id is not None:
                special_ids.add(token_id)
        
        if hasattr(tokenizer, 'all_special_ids'):
            special_ids.update(tokenizer.all_special_ids)
        
        return np.array(sorted(all_ids - special_ids), dtype=np.int64)
    
    def sample_tokens(self, valid_ids: np.ndarray, n: int, seed: int = None) -> np.ndarray:
        rng = np.random.default_rng(seed) if seed else self.rng
        if len(valid_ids) <= n:
            return valid_ids.copy()
        return rng.choice(valid_ids, size=n, replace=False)
    
    def get_common_tokens(self, tokenizer1, tokenizer2) -> Tuple[np.ndarray, np.ndarray]:
        valid1 = self.get_valid_token_ids(tokenizer1)
        valid2 = self.get_valid_token_ids(tokenizer2)
        
        str_to_id1, str_to_id2 = {}, {}
        for tid in valid1:
            try:
                s = tokenizer1.convert_ids_to_tokens(int(tid))
                if s: str_to_id1[s] = tid
            except: pass
        for tid in valid2:
            try:
                s = tokenizer2.convert_ids_to_tokens(int(tid))
                if s: str_to_id2[s] = tid
            except: pass
        
        common = set(str_to_id1.keys()) & set(str_to_id2.keys())
        return (np.array([str_to_id1[s] for s in common], dtype=np.int64),
                np.array([str_to_id2[s] for s in common], dtype=np.int64))


# =============================================================================
# GPU 流式 Pearson 相关计算（核心优化 v4 - RTX 5090 专用）
# =============================================================================
# v4 优化点：
# 1. 纯 GPU 随机采样：移除 CPU 索引计算，使用拒绝采样在 GPU 上生成 pair
# 2. torch.compile：JIT 编译核心计算逻辑，融合 kernel
# 3. 大 batch_size：从 200K 提升到 1M
# 4. 移除频繁 empty_cache：只在最终清理时调用
# 5. TF32 自动启用（全局设置）
# 6. Index Sorting：按 i_indices 排序提升显存访问局部性

def _generate_pairs_gpu(n: int, n_pairs: int, seed: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    纯 GPU 随机采样生成 pair 索引（拒绝采样策略 + 排序优化）
    
    策略：
    1. 生成随机 (i, j)，过滤 i < j 的上三角部分
    2. 按 i_indices 排序，提升后续显存访问的局部性（L2 Cache 命中率）
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed if seed else 42)
    
    total_pairs = n * (n - 1) // 2
    if n_pairs >= total_pairs:
        # 全量：直接生成上三角索引（已经是有序的）
        i_indices = torch.arange(n, device=device).unsqueeze(1).expand(n, n)
        j_indices = torch.arange(n, device=device).unsqueeze(0).expand(n, n)
        mask = i_indices < j_indices
        return i_indices[mask], j_indices[mask]
    
    # 采样模式：拒绝采样，多生成一些以弥补过滤损失
    collected_i = []
    collected_j = []
    remaining = n_pairs
    
    while remaining > 0:
        # 生成 3x 所需数量（约 50% 会被过滤）
        batch = min(remaining * 3, 10_000_000)  # 单批最多 1000 万
        idx_i = torch.randint(0, n, (batch,), device=device, generator=generator)
        idx_j = torch.randint(0, n, (batch,), device=device, generator=generator)
        
        # 过滤：只保留 i < j（上三角）
        mask = idx_i < idx_j
        valid_i = idx_i[mask]
        valid_j = idx_j[mask]
        
        # 取需要的数量
        take = min(len(valid_i), remaining)
        collected_i.append(valid_i[:take])
        collected_j.append(valid_j[:take])
        remaining -= take
    
    i_indices = torch.cat(collected_i)
    j_indices = torch.cat(collected_j)
    
    # 关键优化：按 i_indices 排序，提升显存访问局部性
    # 当后续执行 X_t[i_batch] 时，排序后的索引使得访问更接近顺序读取
    # 这能显著提升 L2 Cache 命中率，充分利用 5090 的显存带宽
    sort_indices = torch.argsort(i_indices)
    i_indices = i_indices[sort_indices]
    j_indices = j_indices[sort_indices]
    
    return i_indices, j_indices


def _compute_batch_stats(
    xi: torch.Tensor, xj: torch.Tensor, yi: torch.Tensor, yj: torch.Tensor,
    xi_norm: torch.Tensor, xj_norm: torch.Tensor, yi_norm: torch.Tensor, yj_norm: torch.Tensor,
    xi_norm2: torch.Tensor, xj_norm2: torch.Tensor, yi_norm2: torch.Tensor, yj_norm2: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """
    计算单批次的统计量（设计为可被 torch.compile 优化）
    """
    # 计算点积
    dot_x = (xi * xj).sum(dim=1)
    dot_y = (yi * yj).sum(dim=1)
    
    # 余弦相似度
    cos_x = (dot_x / (xi_norm * xj_norm).clamp(min=1e-10)).double()
    cos_y = (dot_y / (yi_norm * yj_norm).clamp(min=1e-10)).double()
    
    # 平方欧氏距离
    euc2_x = (xi_norm2 + xj_norm2 - 2 * dot_x).clamp(min=0).double()
    euc2_y = (yi_norm2 + yj_norm2 - 2 * dot_y).clamp(min=0).double()
    
    # 欧氏距离
    euc_x = torch.sqrt(euc2_x)
    euc_y = torch.sqrt(euc2_y)
    
    # 返回各统计量的 sum
    return (
        cos_x.sum(), cos_y.sum(), (cos_x * cos_x).sum(), (cos_y * cos_y).sum(), (cos_x * cos_y).sum(),
        euc_x.sum(), euc_y.sum(), (euc_x * euc_x).sum(), (euc_y * euc_y).sum(), (euc_x * euc_y).sum(),
        euc2_x.sum(), euc2_y.sum(), (euc2_x * euc2_x).sum(), (euc2_y * euc2_y).sum(), (euc2_x * euc2_y).sum(),
    )

# 尝试编译核心函数（PyTorch 2.0+）
# 注意：torch.compile 在多线程环境下可能有兼容性问题，暂时禁用
# try:
#     _compute_batch_stats_compiled = torch.compile(_compute_batch_stats, mode="reduce-overhead")
#     USE_COMPILE = True
#     print("torch.compile 已启用 (reduce-overhead 模式)")
# except Exception as e:
#     _compute_batch_stats_compiled = _compute_batch_stats
#     USE_COMPILE = False
#     print(f"torch.compile 不可用，使用原生模式: {e}")
_compute_batch_stats_compiled = _compute_batch_stats
USE_COMPILE = False
print("torch.compile 已禁用（多线程兼容性）")


def compute_gcorr_gpu_streaming_v4(X: np.ndarray, Y: np.ndarray, n_pairs: int,
                                    seed: int = None, batch_size: int = None,
                                    device: torch.device = None) -> Dict[str, float]:
    """
    GPU 流式计算 GCorr（余弦、欧氏、平方欧氏）- v4 RTX 5090 优化版
    
    v4 优化：
    - 纯 GPU 随机采样（拒绝采样策略）
    - 动态 batch_size（根据 hidden_dim 自动调整）
    - 移除循环内 empty_cache
    """
    if device is None:
        device = DEVICE
    
    n, d_x = X.shape
    _, d_y = Y.shape
    
    # 动态计算 batch_size：根据 hidden_dim 调整
    # 每个 batch 需要 8 个张量 (xi, xj, yi, yj 各 2 个)，每个 shape [batch, d]
    # 显存估算：4 * batch * d_x * 4 + 4 * batch * d_y * 4 bytes (float32)
    # RTX 5090 32GB，保守使用 4GB 给批次计算（因为还有其他数据占用显存）
    if batch_size is None:
        target_memory = 4 * 1024**3  # 4GB（更保守）
        bytes_per_sample = 4 * d_x * 4 + 4 * d_y * 4  # xi,xj,yi,yj 各 2 个张量
        batch_size = max(50_000, min(500_000, target_memory // bytes_per_sample))
    
    # print(f"    [DEBUG] hidden_dim={d}, batch_size={batch_size:,}")
    
    with torch.inference_mode():
        # 转移数据到 GPU
        X_t = torch.from_numpy(X).float().to(device)
        Y_t = torch.from_numpy(Y).float().to(device)
        
        # 预计算每行的 L2 范数
        X_norm2 = (X_t * X_t).sum(dim=1)
        Y_norm2 = (Y_t * Y_t).sum(dim=1)
        X_norm = torch.sqrt(X_norm2.clamp(min=1e-20))
        Y_norm = torch.sqrt(Y_norm2.clamp(min=1e-20))
        
        # GPU 上生成采样索引
        i_indices, j_indices = _generate_pairs_gpu(n, n_pairs, seed, device)
        num_pairs = len(i_indices)
        
        # 初始化流式统计量（FP64 保证精度）
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
        
        # 选择计算函数（编译版或原生版）
        compute_fn = _compute_batch_stats_compiled if USE_COMPILE else _compute_batch_stats
        
        # 分批计算并累加
        for start in range(0, num_pairs, batch_size):
            end = min(start + batch_size, num_pairs)
            i_batch = i_indices[start:end]
            j_batch = j_indices[start:end]
            
            # 提取批次向量
            xi, xj = X_t[i_batch], X_t[j_batch]
            yi, yj = Y_t[i_batch], Y_t[j_batch]
            
            # 提取预计算的范数
            xi_norm, xj_norm = X_norm[i_batch], X_norm[j_batch]
            yi_norm, yj_norm = Y_norm[i_batch], Y_norm[j_batch]
            xi_norm2, xj_norm2 = X_norm2[i_batch], X_norm2[j_batch]
            yi_norm2, yj_norm2 = Y_norm2[i_batch], Y_norm2[j_batch]
            
            # 计算统计量
            stats = compute_fn(
                xi, xj, yi, yj,
                xi_norm, xj_norm, yi_norm, yj_norm,
                xi_norm2, xj_norm2, yi_norm2, yj_norm2,
            )
            
            # 累加
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
        
        # 计算 Pearson 相关系数
        n_total = float(num_pairs)
        
        def pearson_from_sums(sum_x, sum_y, sum_x2, sum_y2, sum_xy, n):
            numerator = n * sum_xy - sum_x * sum_y
            denominator = torch.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
            return (numerator / denominator.clamp(min=1e-10)).item()
        
        gcorr_cos = pearson_from_sums(cos_sum_x, cos_sum_y, cos_sum_x2, cos_sum_y2, cos_sum_xy, n_total)
        gcorr_euc = pearson_from_sums(euc_sum_x, euc_sum_y, euc_sum_x2, euc_sum_y2, euc_sum_xy, n_total)
        gcorr_euc2 = pearson_from_sums(euc2_sum_x, euc2_sum_y, euc2_sum_x2, euc2_sum_y2, euc2_sum_xy, n_total)
    
    # 清理 GPU（仅删除引用，不调用 empty_cache）
    del X_t, Y_t, X_norm, Y_norm, X_norm2, Y_norm2, i_indices, j_indices
    
    return {'gcorr_cos': gcorr_cos, 'gcorr_euc': gcorr_euc, 'gcorr_euc2': gcorr_euc2}


# =============================================================================
# 批量计算器（双卡优化版 v4）
# =============================================================================
class BatchGCorrCalculatorV4:
    """
    v4 批量计算器 - RTX 5090 优化版
    优化：
    1. 使用 v4 版 GPU 流式计算（纯 GPU 采样 + torch.compile）
    2. 双卡并行计算模型对
    3. 移除不必要的 empty_cache 调用
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.loader = ModelLoader(config)
        self.sampler = TokenSampler(config)
    
    def _compute_single_pair_bootstrap(self, E_a, E_b, U_a, U_b, n_pairs, seed, device):
        """计算单个模型对的一次 Bootstrap"""
        gcorr_E = compute_gcorr_gpu_streaming_v4(E_a, E_b, n_pairs, seed * 1000, device=device)
        gcorr_U = compute_gcorr_gpu_streaming_v4(U_a, U_b, n_pairs, seed * 1000 + 1, device=device)
        
        return {
            'gcorr_E_cos': gcorr_E['gcorr_cos'],
            'gcorr_E_euc': gcorr_E['gcorr_euc'],
            'gcorr_E_euc2': gcorr_E['gcorr_euc2'],
            'gcorr_U_cos': gcorr_U['gcorr_cos'],
            'gcorr_U_euc': gcorr_U['gcorr_euc'],
            'gcorr_U_euc2': gcorr_U['gcorr_euc2'],
        }
    
    def compute_batch(
        self,
        model_names: List[str],
        description: str = "",
        exclude_pairs: set = None,
    ) -> List[Dict]:
        """
        批量计算同 tokenizer 模型组的 GCorr
        优化：共享采样 + 双卡并行
        
        当只有 1 对模型时，将 Bootstrap 分配到双卡并行
        当有多对模型时，将模型对分配到双卡并行
        
        Args:
            exclude_pairs: 要排除的模型对集合，格式为 {tuple(sorted([model_a, model_b])), ...}
        """
        print(f"\n{'='*70}")
        print(f"批量计算: {description}")
        print(f"模型数: {len(model_names)}, 对比组数: {len(model_names)*(len(model_names)-1)//2}")
        print(f"参数: n_tokens={self.config.n_tokens}, n_bootstrap={self.config.n_bootstrap}, n_pairs={self.config.n_pairs:,}")
        print(f"{'='*70}")
        
        # 加载所有模型
        models_data = {}
        for name in model_names:
            try:
                models_data[name] = self.loader.load(name)
            except Exception as e:
                print(f"  ❌ 加载失败 {name}: {e}")
        
        if len(models_data) < 2:
            print("  模型数不足，跳过")
            return []
        
        first_model = list(models_data.values())[0]
        tokenizer = first_model['tokenizer']
        valid_ids = self.sampler.get_valid_token_ids(tokenizer)
        
        results = []
        model_list = list(models_data.keys())
        pairs = list(itertools.combinations(model_list, 2))
        
        # 排除已做过的配对（实验1.2需要排除实验1.1中的同尺寸Base-Instruct对）
        if exclude_pairs:
            original_count = len(pairs)
            pairs = [p for p in pairs if tuple(sorted(p)) not in exclude_pairs]
            excluded_count = original_count - len(pairs)
            if excluded_count > 0:
                print(f"  排除已做过的配对: {excluded_count} 对")
        
        if not pairs:
            print("  无有效配对，跳过")
            return []
        
        print(f"  实际对比组数: {len(pairs)}")
        
        all_bootstrap_results = {pair: [] for pair in pairs}
        
        # 构建所有任务：(pair, bootstrap_idx, gpu_id)
        all_tasks = []
        task_idx = 0
        for pair in pairs:
            for b in range(self.config.n_bootstrap):
                all_tasks.append((pair, b, task_idx % NUM_GPUS))
                task_idx += 1
        
        def compute_single_bootstrap(task):
            """计算单个 Bootstrap 任务"""
            (model_a, model_b), b, gpu_id = task
            device = DEVICES[gpu_id]
            
            seed = self.config.random_seed + b
            sample_ids = self.sampler.sample_tokens(valid_ids, self.config.n_tokens, seed)
            
            data_a = models_data[model_a]
            data_b = models_data[model_b]
            
            E_a = data_a['E'][sample_ids]
            E_b = data_b['E'][sample_ids]
            U_a = data_a['U'][sample_ids]
            U_b = data_b['U'][sample_ids]
            
            # 使用锁保护 GPU 访问，避免显存碎片化
            with GPU_LOCKS[gpu_id]:
                result = self._compute_single_pair_bootstrap(E_a, E_b, U_a, U_b, self.config.n_pairs, seed, device)
            
            return (model_a, model_b), result
        
        # 多卡并行：每卡同时跑 1 个任务，8 卡即 8 个任务同步执行
        max_workers = NUM_GPUS
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(compute_single_bootstrap, task) for task in all_tasks]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Bootstrap ({NUM_GPUS}卡并行)"):
                pair, result = future.result()
                all_bootstrap_results[pair].append(result)
        
        # 汇总统计
        for (model_a, model_b) in pairs:
            data_a = models_data[model_a]
            data_b = models_data[model_b]
            bootstrap_results = all_bootstrap_results[(model_a, model_b)]
            
            result = self._summarize(model_a, model_b, data_a, data_b, bootstrap_results, self.config.n_tokens)
            results.append(result)
            
            print(f"  {model_a} vs {model_b}: cos(E)={result['gcorr_E_cos_mean']:.4f}, cos(U)={result['gcorr_U_cos_mean']:.4f}")
        
        return results
    
    def compute_cross_tokenizer(
        self,
        model_names: List[str],
        description: str = "",
        include_pairs: Optional[Set[Tuple[str, str]]] = None,
    ) -> List[Dict]:
        """
        跨 tokenizer 计算（多卡并行优化）
        优化策略：将所有 (模型对, bootstrap) 任务一次性投入线程池，分配到所有可用 GPU。
        include_pairs: 若指定，则只计算其中列出的模型对（用于 Base vs Instruct 等场景）。
        """
        print(f"\n{'='*70}")
        print(f"跨 tokenizer 计算: {description}")
        print(f"模型数: {len(model_names)}, GPU 数: {NUM_GPUS}")
        print(f"{'='*70}")
        
        # 加载所有模型
        models_data = {}
        for name in model_names:
            try:
                models_data[name] = self.loader.load(name)
            except Exception as e:
                print(f"  ❌ 加载失败 {name}: {e}")
        
        if len(models_data) < 2:
            return []
        
        model_list = list(models_data.keys())
        pairs = list(itertools.combinations(model_list, 2))

        # 若指定了 include_pairs，只保留这些配对（忽略方向）
        if include_pairs is not None:
            include_norm = {tuple(sorted(p)) for p in include_pairs}
            pairs = [p for p in pairs if tuple(sorted(p)) in include_norm]
            print(f"  include_pairs 过滤后对数: {len(pairs)}")
        
        # 预计算每个模型对的 common token ids（主线程完成）
        pair_token_info = {}
        for (model_a, model_b) in pairs:
            data_a = models_data[model_a]
            data_b = models_data[model_b]
            same_tokenizer = (data_a['vocab_size'] == data_b['vocab_size'])
            
            if same_tokenizer:
                valid_ids = self.sampler.get_valid_token_ids(data_a['tokenizer'])
                pair_token_info[(model_a, model_b)] = {
                    'same_tokenizer': True,
                    'ids_a': valid_ids,
                    'ids_b': valid_ids,
                    'n_sample': min(self.config.n_tokens, len(valid_ids))
                }
            else:
                ids_a, ids_b = self.sampler.get_common_tokens(data_a['tokenizer'], data_b['tokenizer'])
                if len(ids_a) < 1000:
                    pair_token_info[(model_a, model_b)] = None  # 跳过
                else:
                    pair_token_info[(model_a, model_b)] = {
                        'same_tokenizer': False,
                        'ids_a': ids_a,
                        'ids_b': ids_b,
                        'n_sample': min(self.config.n_tokens, len(ids_a))
                    }
        
        # 过滤有效的模型对
        valid_pairs = [(p, info) for p, info in pair_token_info.items() if info is not None]
        if not valid_pairs:
            return []
        
        print(f"有效模型对数: {len(valid_pairs)}, 总任务数: {len(valid_pairs) * self.config.n_bootstrap}  (分配到 {NUM_GPUS} 张 GPU)")
        
        # 构建所有任务: (model_a, model_b, bootstrap_idx, gpu_id)
        all_tasks = []
        task_idx = 0
        for (model_a, model_b), info in valid_pairs:
            for b in range(self.config.n_bootstrap):
                all_tasks.append((model_a, model_b, b, task_idx % NUM_GPUS, info))
                task_idx += 1
        
        # 收集结果
        all_bootstrap_results = {p: [] for p, _ in valid_pairs}
        
        def compute_single_task(task):
            """计算单个 (模型对, bootstrap) 任务"""
            model_a, model_b, b, gpu_id, info = task
            device = DEVICES[gpu_id]
            
            data_a = models_data[model_a]
            data_b = models_data[model_b]
            
            seed = self.config.random_seed + b
            n_sample = info['n_sample']
            rng = np.random.default_rng(seed)
            
            if info['same_tokenizer']:
                sample_ids = self.sampler.sample_tokens(info['ids_a'], n_sample, seed)
                sample_a = sample_b = sample_ids
            else:
                indices = rng.choice(len(info['ids_a']), size=n_sample, replace=False)
                sample_a = info['ids_a'][indices]
                sample_b = info['ids_b'][indices]
            
            E_a = data_a['E'][sample_a]
            E_b = data_b['E'][sample_b]
            U_a = data_a['U'][sample_a]
            U_b = data_b['U'][sample_b]
            
            # 使用锁保护 GPU 访问，避免显存碎片化
            with GPU_LOCKS[gpu_id]:
                result = self._compute_single_pair_bootstrap(E_a, E_b, U_a, U_b, self.config.n_pairs, seed, device)
            
            return (model_a, model_b), result
        
        # 多卡并行：每卡同时跑 1 个任务，8 卡即 8 个任务同步执行
        max_workers = NUM_GPUS
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(compute_single_task, task) for task in all_tasks]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Bootstrap ({NUM_GPUS}卡并行)"):
                pair, result = future.result()
                all_bootstrap_results[pair].append(result)
        
        # 汇总统计
        results = []
        for (model_a, model_b), info in valid_pairs:
            data_a = models_data[model_a]
            data_b = models_data[model_b]
            bootstrap_results = all_bootstrap_results[(model_a, model_b)]
            
            result = self._summarize(model_a, model_b, data_a, data_b, bootstrap_results, info['n_sample'])
            results.append(result)
            print(f"  {model_a} vs {model_b}: cos(E)={result['gcorr_E_cos_mean']:.4f}, cos(U)={result['gcorr_U_cos_mean']:.4f}")
        
        return results
    
    def _summarize(self, model_a, model_b, data_a, data_b, bootstrap_results, n_tokens):
        def summarize_key(key):
            values = np.array([r[key] for r in bootstrap_results])
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'ci95_low': np.percentile(values, 2.5),
                'ci95_high': np.percentile(values, 97.5),
                'median': np.median(values),
                'se': np.std(values) / np.sqrt(len(values)),
            }
        
        E_cos = summarize_key('gcorr_E_cos')
        E_euc = summarize_key('gcorr_E_euc')
        E_euc2 = summarize_key('gcorr_E_euc2')
        U_cos = summarize_key('gcorr_U_cos')
        U_euc = summarize_key('gcorr_U_euc')
        U_euc2 = summarize_key('gcorr_U_euc2')
        
        return {
            'model_a': model_a,
            'model_b': model_b,
            'family_a': data_a['family'],
            'family_b': data_b['family'],
            'same_family': data_a['family'] == data_b['family'],
            'same_tokenizer': data_a['vocab_size'] == data_b['vocab_size'],
            'n_tokens': n_tokens,
            'n_bootstrap': self.config.n_bootstrap,
            'n_pairs': self.config.n_pairs,
            'hidden_dim_a': data_a['hidden_dim'],
            'hidden_dim_b': data_b['hidden_dim'],
            'vocab_size_a': data_a['vocab_size'],
            'vocab_size_b': data_b['vocab_size'],
            'is_tied_a': data_a['is_tied'],
            'is_tied_b': data_b['is_tied'],
            'actual_tied_a': data_a['actual_tied'],
            'actual_tied_b': data_b['actual_tied'],
            # E 余弦
            'gcorr_E_cos_mean': E_cos['mean'],
            'gcorr_E_cos_std': E_cos['std'],
            'gcorr_E_cos_se': E_cos['se'],
            'gcorr_E_cos_ci95_low': E_cos['ci95_low'],
            'gcorr_E_cos_ci95_high': E_cos['ci95_high'],
            # U 余弦
            'gcorr_U_cos_mean': U_cos['mean'],
            'gcorr_U_cos_std': U_cos['std'],
            'gcorr_U_cos_se': U_cos['se'],
            'gcorr_U_cos_ci95_low': U_cos['ci95_low'],
            'gcorr_U_cos_ci95_high': U_cos['ci95_high'],
            # E 欧氏
            'gcorr_E_euc_mean': E_euc['mean'],
            'gcorr_E_euc_std': E_euc['std'],
            'gcorr_E_euc_se': E_euc['se'],
            'gcorr_E_euc_ci95_low': E_euc['ci95_low'],
            'gcorr_E_euc_ci95_high': E_euc['ci95_high'],
            # U 欧氏
            'gcorr_U_euc_mean': U_euc['mean'],
            'gcorr_U_euc_std': U_euc['std'],
            'gcorr_U_euc_se': U_euc['se'],
            'gcorr_U_euc_ci95_low': U_euc['ci95_low'],
            'gcorr_U_euc_ci95_high': U_euc['ci95_high'],
            # E 平方欧氏（新增）
            'gcorr_E_euc2_mean': E_euc2['mean'],
            'gcorr_E_euc2_std': E_euc2['std'],
            'gcorr_E_euc2_se': E_euc2['se'],
            'gcorr_E_euc2_ci95_low': E_euc2['ci95_low'],
            'gcorr_E_euc2_ci95_high': E_euc2['ci95_high'],
            # U 平方欧氏（新增）
            'gcorr_U_euc2_mean': U_euc2['mean'],
            'gcorr_U_euc2_std': U_euc2['std'],
            'gcorr_U_euc2_se': U_euc2['se'],
            'gcorr_U_euc2_ci95_low': U_euc2['ci95_low'],
            'gcorr_U_euc2_ci95_high': U_euc2['ci95_high'],
        }
    
    def clear_cache(self):
        """清理模型缓存（仅在切换模型组时调用）"""
        self.loader.clear_cache()


# =============================================================================
# 实验运行器 v4
# =============================================================================
class Experiment1RunnerV4:
    def __init__(self, config: Config):
        self.config = config
        self.calculator = BatchGCorrCalculatorV4(config)
        self.all_results = []
        os.makedirs(config.output_dir, exist_ok=True)
    
    def run_base_vs_instruct(self):
        """1. 所有 Base vs Instruct 对比（一次性加载 + 全部并行，充分利用多卡）"""
        print("\n" + "="*80)
        print("实验 1.1: 所有 Base vs Instruct 对比")
        print("="*80)

        # 把 19 对涉及的全部 38 个模型一次性传入 compute_cross_tokenizer，
        # 用 include_pairs 过滤只算这 19 对，
        # 让 19 × n_bootstrap 个任务一起投入线程池分发到所有 GPU。
        # compute_cross_tokenizer 内部会对每对单独做 tokenizer 对齐，
        # 彻底避免跨 tokenizer 索引越界。
        all_models: List[str] = sorted({m for pair in BASE_INSTRUCT_PAIRS for m in pair})
        results = self.calculator.compute_cross_tokenizer(
            all_models,
            f"Base vs Instruct 全部 {len(BASE_INSTRUCT_PAIRS)} 对并行",
            include_pairs=BASE_INSTRUCT_PAIRS_SET,
        )
        self.all_results.extend(results)
        self.calculator.clear_cache()
    
    def run_intra_family(self, family: str):
        """2. 同系列全模型互比（排除实验1.1中已做过的同尺寸Base-Instruct对）"""
        models = ALL_MODELS.get(family, [])
        if len(models) < 2:
            return
        
        print(f"\n" + "="*80)
        print(f"实验 1.2: {family} 系列内部对比（排除已做的Base-Instruct对）")
        print("="*80)
        
        results = self.calculator.compute_batch(
            models, 
            f"{family} 系列内部对比",
            exclude_pairs=BASE_INSTRUCT_PAIRS_SET
        )
        self.all_results.extend(results)
        self.calculator.clear_cache()
    
    def run_cross_family_by_size(self, size_group: str):
        """3. 跨系列同规模对比"""
        models = MODELS_BY_SIZE.get(size_group, [])
        if len(models) < 2:
            return
        
        results = self.calculator.compute_cross_tokenizer(models, f"跨系列 {size_group} 规模对比")
        self.all_results.extend(results)
        self.calculator.clear_cache()
    
    def run_moe_comparison(self):
        """4. MoE 模型对比"""
        moe_models = ["Qwen3-30B-A3B", "DeepSeek-V2-Lite-Chat"]
        dense_models = ["Qwen3-8B", "Qwen3-14B", "Qwen2.5-7B", "Qwen2.5-14B"]
        
        all_models = moe_models + dense_models
        results = self.calculator.compute_cross_tokenizer(all_models, "MoE 模型对比")
        self.all_results.extend(results)
        self.calculator.clear_cache()
    
    def run_all(self):
        """运行所有实验"""
        start_time = time.time()
        
        self.run_base_vs_instruct()
        
        for family in ["Qwen3", "Qwen2.5", "Llama", "Gemma2"]:
            self.run_intra_family(family)
        
        for size_group in ["~0.5-0.6B", "~1-2B", "~3-4B", "~7-9B", "~14B", "~27-32B", "~70-72B"]:
            self.run_cross_family_by_size(size_group)
        
        self.run_moe_comparison()
        
        self.save_results()
        
        elapsed = time.time() - start_time
        print(f"\n总耗时: {elapsed/60:.2f} 分钟")
        print(f"总对比组数: {len(self.all_results)}")
    
    def save_results(self):
        if not self.all_results:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        seen = set()
        unique_results = []
        for r in self.all_results:
            key = tuple(sorted([r['model_a'], r['model_b']]))
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        
        df = pd.DataFrame(unique_results)
        csv_path = os.path.join(self.config.output_dir, f"exp1_global_v4_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n结果已保存: {csv_path}")
        print(f"去重后对比组数: {len(unique_results)}")
        
        metadata = {
            'timestamp': timestamp,
            'version': 'v4',
            'config': {
                'n_tokens': self.config.n_tokens,
                'n_bootstrap': self.config.n_bootstrap,
                'n_pairs': self.config.n_pairs,
                'random_seed': self.config.random_seed,
            },
            'optimizations': [
                '纯 GPU 随机采样（拒绝采样策略）',
                'torch.compile JIT 编译',
                'TF32 Tensor Core 加速',
                '大 Batch Size (1M)',
                '移除频繁 empty_cache 调用',
                'Index Sorting 显存访问局部性优化',
            ],
            'total_comparisons': len(unique_results),
            'torch_compile_enabled': USE_COMPILE,
        }
        meta_path = os.path.join(self.config.output_dir, f"metadata_v4_{timestamp}.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)


# =============================================================================
# 主函数
# =============================================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='实验1 v4：全局几何相似性验证（RTX 5090 极致优化版）')
    parser.add_argument('--n_tokens', type=int, default=20000)
    parser.add_argument('--n_bootstrap', type=int, default=100)
    parser.add_argument('--n_pairs', type=int, default=5000000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'base_instruct', 'intra_family', 'cross_size', 'moe'],
                        help='运行模式')
    parser.add_argument('--family', type=str, default=None)
    parser.add_argument('--size', type=str, default=None)
    args = parser.parse_args()
    
    config = Config(
        n_tokens=args.n_tokens,
        n_bootstrap=args.n_bootstrap,
        n_pairs=args.n_pairs,
        random_seed=args.seed,
    )
    
    runner = Experiment1RunnerV4(config)
    
    if args.mode == 'all':
        runner.run_all()
    elif args.mode == 'base_instruct':
        runner.run_base_vs_instruct()
        runner.save_results()
    elif args.mode == 'intra_family':
        if args.family:
            runner.run_intra_family(args.family)
        else:
            for family in ["Qwen3", "Qwen2.5", "Llama", "Gemma2"]:
                runner.run_intra_family(family)
        runner.save_results()
    elif args.mode == 'cross_size':
        if args.size:
            runner.run_cross_family_by_size(args.size)
        else:
            for size in ["~0.5-0.6B", "~1-2B", "~3-4B", "~7-9B", "~14B", "~27-32B", "~70-72B"]:
                runner.run_cross_family_by_size(size)
        runner.save_results()
    elif args.mode == 'moe':
        runner.run_moe_comparison()
        runner.save_results()


if __name__ == "__main__":
    main()
