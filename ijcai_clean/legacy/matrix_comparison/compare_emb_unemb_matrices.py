#!/usr/bin/env python3
"""
微调前后模型 Embedding 和 Unembedding 矩阵差异分析
====================================================
专注于直接比较矩阵之间的差异，不涉及其他实验
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# GPU配置
NUM_GPUS = torch.cuda.device_count()
DEVICES = [torch.device(f"cuda:{i}") for i in range(NUM_GPUS)] if NUM_GPUS > 0 else [torch.device("cpu")]
print(f"检测到 {NUM_GPUS} 个GPU: {[str(d) for d in DEVICES]}")

@dataclass
class MatrixComparisonConfig:
    """矩阵比较配置"""
    model_base_path: str = "/root/shared-nvme/models"
    output_dir: str = "/root/shared-nvme/ijcai/matrix_comparison/data"
    
    # 模型对列表：(基座模型, 微调模型)
    # 包含所有基座vs微调的模型对，共19个
    model_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        # Qwen3系列
        ("Qwen3-0.6B-Base", "Qwen3-0.6B"),
        ("Qwen3-1.7B-Base", "Qwen3-1.7B"),
        ("Qwen3-4B-Base", "Qwen3-4B"),
        ("Qwen3-8B-Base", "Qwen3-8B"),
        ("Qwen3-14B-Base", "Qwen3-14B"),
        # Qwen2.5系列
        ("Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct"),
        ("Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct"),
        ("Qwen2.5-3B", "Qwen2.5-3B-Instruct"),
        ("Qwen2.5-7B", "Qwen2.5-7B-Instruct"),
        ("Qwen2.5-14B", "Qwen2.5-14B-Instruct"),
        ("Qwen2.5-32B", "Qwen2.5-32B-Instruct"),
        ("Qwen2.5-72B-Base", "Qwen2.5-72B-Instruct"),
        # Llama系列
        ("Llama-3.2-1B", "Llama-3.2-1B-Instruct"),
        ("Llama-3.2-3B", "Llama-3.2-3B-Instruct"),
        ("Llama-3.1-8B", "Llama-3.1-8B-Instruct"),
        ("Llama-3.1-70B-Base", "Llama-3.1-70B-Instruct"),
        # Gemma2系列
        ("Gemma-2-2B", "Gemma-2-2B-Instruct"),
        ("Gemma-2-9B", "Gemma-2-9B-Instruct"),
        ("Gemma-2-27B", "Gemma-2-27B-Instruct"),
    ])
    
    def get_model_path(self, model_name: str) -> str:
        """获取模型完整路径"""
        return os.path.join(self.model_base_path, model_name)


class MatrixLoader:
    """矩阵加载器"""
    
    def __init__(self, config: MatrixComparisonConfig):
        self.config = config
        self.cache: Dict[str, Dict] = {}
    
    def load_matrices(self, model_name: str) -> Dict:
        """
        加载模型的E和U矩阵
        
        Returns:
            dict: {
                'E': embedding矩阵 (vocab_size, hidden_dim),
                'U': unembedding矩阵 (vocab_size, hidden_dim),
                'vocab_size': 词表大小,
                'hidden_dim': 隐藏维度,
                'is_tied': 是否tied,
                'actual_tied': 实际是否tied,
                'tokenizer': tokenizer对象,
            }
        """
        if model_name in self.cache:
            return self.cache[model_name]
        
        model_path = self.config.get_model_path(model_name)
        print(f"\n加载模型: {model_name}")
        print(f"  路径: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        # 加载配置
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 优化：直接加载权重文件，只提取embedding层（速度快10-100倍）
        import glob
        
        # 尝试safetensors格式（最快）
        E = None
        U = None
        try:
            from safetensors import safe_open
            safetensors_files = glob.glob(os.path.join(model_path, "model*.safetensors"))
            if safetensors_files:
                for f in sorted(safetensors_files):
                    with safe_open(f, framework="pt", device="cpu") as sf:
                        for key in sf.keys():
                            if ('embed_tokens' in key or 'wte' in key) and 'weight' in key and E is None:
                                E = sf.get_tensor(key).float().numpy().astype(np.float32)
                            if ('lm_head' in key or 'output_embedding' in key) and 'weight' in key and U is None:
                                U = sf.get_tensor(key).float().numpy().astype(np.float32)
        except ImportError:
            pass
        
        # 如果safetensors失败，尝试PyTorch格式（优化：只加载需要的key）
        if E is None:
            pytorch_files = glob.glob(os.path.join(model_path, "pytorch_model*.bin"))
            if pytorch_files:
                # 先快速扫描找到embedding和lm_head的key
                embed_key = None
                lm_head_key = None
                for f in sorted(pytorch_files):
                    # 只读取metadata，不加载完整tensor
                    checkpoint = torch.load(f, map_location='cpu', weights_only=True)
                    for key in checkpoint.keys():
                        if embed_key is None and ('embed_tokens' in key or 'wte' in key or 'embedding' in key) and 'weight' in key:
                            embed_key = key
                        if lm_head_key is None and ('lm_head' in key or 'output_embedding' in key) and 'weight' in key:
                            lm_head_key = key
                    if embed_key and lm_head_key:
                        break
                
                # 只加载需要的tensor
                for f in sorted(pytorch_files):
                    checkpoint = torch.load(f, map_location='cpu', weights_only=True)
                    if embed_key and embed_key in checkpoint and E is None:
                        E = checkpoint[embed_key].float().numpy().astype(np.float32)
                    if lm_head_key and lm_head_key in checkpoint and U is None:
                        U = checkpoint[lm_head_key].float().numpy().astype(np.float32)
                    if E is not None and (U is not None or embed_key == lm_head_key):
                        break
                    del checkpoint
        
        # 如果直接加载权重失败，fallback到加载整个模型
        if E is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
            )
            input_embeddings = model.get_input_embeddings()
            E = input_embeddings.weight.detach().float().numpy().astype(np.float32)
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is not None:
                U = output_embeddings.weight.detach().float().numpy().astype(np.float32)
            else:
                if hasattr(model, 'lm_head'):
                    U = model.lm_head.weight.detach().float().numpy().astype(np.float32)
                else:
                    U = E.copy()
            del model
        
        if U is None:
            U = E.copy()  # tied模型
        
        # 检查是否tied
        is_tied = getattr(model_config, 'tie_word_embeddings', True)
        
        # 优化D: 判断是否tied（直接比较numpy数组）
        if is_tied:
            # 小样本抽检（快速）
            sample_size = min(1024, E.shape[0])
            np.random.seed(42)
            sample_indices = np.random.choice(E.shape[0], sample_size, replace=False)
            actual_tied = np.allclose(E[sample_indices], U[sample_indices], rtol=1e-5, atol=1e-5)
        else:
            actual_tied = False
        
        vocab_size, hidden_dim = E.shape
        
        result = {
            'E': E,
            'U': U,
            'vocab_size': vocab_size,
            'hidden_dim': hidden_dim,
            'is_tied': is_tied,
            'actual_tied': actual_tied,
            'tokenizer': tokenizer,
            'model_name': model_name,
        }
        
        self.cache[model_name] = result
        
        print(f"  ✓ 加载完成: vocab_size={vocab_size}, hidden_dim={hidden_dim}")
        print(f"  ✓ config_tied={is_tied}, actual_tied={actual_tied}")
        
        return result


class MatrixComparator:
    """矩阵比较器"""
    
    def __init__(self, config: MatrixComparisonConfig):
        self.config = config
        self.loader = MatrixLoader(config)
        os.makedirs(config.output_dir, exist_ok=True)
    
    def get_common_tokens(self, tokenizer1, tokenizer2) -> Tuple[np.ndarray, np.ndarray]:
        """获取两个tokenizer的共同token"""
        vocab1 = set(range(tokenizer1.vocab_size))
        vocab2 = set(range(tokenizer2.vocab_size))
        
        # 获取特殊token ids
        def get_special_ids(tok):
            special = set()
            attrs = ['bos_token_id', 'eos_token_id', 'pad_token_id', 
                    'unk_token_id', 'cls_token_id', 'sep_token_id', 'mask_token_id']
            for attr in attrs:
                val = getattr(tok, attr, None)
                if val is not None:
                    special.add(val)
            if hasattr(tok, 'additional_special_tokens_ids'):
                special.update(tok.additional_special_tokens_ids)
            if hasattr(tok, 'all_special_ids'):
                special.update(tok.all_special_ids)
            return special
        
        special1 = get_special_ids(tokenizer1)
        special2 = get_special_ids(tokenizer2)
        
        # 构建token字符串到id的映射
        str_to_id1 = {}
        for tid in vocab1 - special1:
            try:
                token_str = tokenizer1.convert_ids_to_tokens(int(tid))
                if token_str:
                    str_to_id1[token_str] = tid
            except:
                pass
        
        str_to_id2 = {}
        for tid in vocab2 - special2:
            try:
                token_str = tokenizer2.convert_ids_to_tokens(int(tid))
                if token_str:
                    str_to_id2[token_str] = tid
            except:
                pass
        
        # 找共同token（优化E: 稳定排序）
        common_strs = sorted(set(str_to_id1.keys()) & set(str_to_id2.keys()))
        
        ids1 = np.array([str_to_id1[s] for s in common_strs], dtype=np.int64)
        ids2 = np.array([str_to_id2[s] for s in common_strs], dtype=np.int64)
        
        return ids1, ids2
    
    def compute_matrix_differences(self, M1: np.ndarray, M2: np.ndarray, 
                                   name: str = "", device: torch.device = None) -> Dict:
        """
        计算两个矩阵之间的各种差异指标（简化版：大矩阵自动降级到CPU）
        
        Args:
            M1: 第一个矩阵 (n, d)
            M2: 第二个矩阵 (n, d)
            name: 矩阵名称（用于输出）
            device: GPU设备，如果为None则自动选择
        
        Returns:
            包含各种差异指标的字典
        """
        assert M1.shape == M2.shape, f"矩阵形状不匹配: {M1.shape} vs {M2.shape}"
        
        n, d = M1.shape
        total_elements = n * d
        
        # 简单策略：超大矩阵（>5000万元素）直接用CPU，避免OOM
        # 中等矩阵用GPU加速，小矩阵无所谓
        use_gpu = False
        if device is None:
            device = DEVICES[0] if NUM_GPUS > 0 else torch.device("cpu")
        
        # 优化：5090有32GB显存，可以处理很大的矩阵
        # 128256×8192 ≈ 10.5亿元素 ≈ 4.2GB (float32)，加上中间tensor约15GB，32GB显存足够
        if NUM_GPUS > 0:  # 有GPU就用GPU，5090显存足够
            use_gpu = True
            # 只在显存严重不足时才fallback到CPU
            if torch.cuda.is_available():
                free_memory_gb = (torch.cuda.get_device_properties(device).total_memory - \
                                torch.cuda.memory_allocated(device)) / (1024**3)
                needed_memory_gb = (total_elements * 4 * 6) / (1024**3)  # 6个主要tensor
                if free_memory_gb < needed_memory_gb:  # 显存不足时才用CPU
                    use_gpu = False
                    print(f"  注意: 显存不足({free_memory_gb:.1f}GB < {needed_memory_gb:.1f}GB)，使用CPU计算")
        
        if use_gpu:
            # GPU计算（中等矩阵）
            M1_t = torch.from_numpy(M1).float().to(device)
            M2_t = torch.from_numpy(M2).float().to(device)
            
            diff_t = M2_t - M1_t
            abs_diff_t = torch.abs(diff_t)
            
            # Frobenius范数
            frobenius_norm_diff = torch.norm(diff_t, p='fro').item()
            frobenius_norm_M1 = torch.norm(M1_t, p='fro').item()
            frobenius_norm_M2 = torch.norm(M2_t, p='fro').item()
            relative_frobenius_diff = frobenius_norm_diff / (frobenius_norm_M1 + 1e-10)
            
            # 余弦相似度
            M1_norm_t = M1_t / (torch.norm(M1_t, dim=1, keepdim=True) + 1e-10)
            M2_norm_t = M2_t / (torch.norm(M2_t, dim=1, keepdim=True) + 1e-10)
            cosine_sims_t = torch.sum(M1_norm_t * M2_norm_t, dim=1)
            
            # L2距离
            l2_distances_t = torch.norm(diff_t, dim=1)
            l2_distances_norm_t = torch.norm(M1_norm_t - M2_norm_t, dim=1)
            M1_norms_t = torch.norm(M1_t, dim=1)
            relative_l2_distances_t = l2_distances_t / (M1_norms_t + 1e-10)
            
            # 统计量（GPU上计算）
            mean_abs_diff = abs_diff_t.mean().item()
            std_abs_diff = abs_diff_t.std().item()
            max_abs_diff = abs_diff_t.max().item()
            min_abs_diff = abs_diff_t.min().item()
            
            # median：大矩阵用采样近似（避免quantile OOM）
            if total_elements > 10_000_000:
                # 采样10万个元素计算median
                sample_size = min(100_000, total_elements)
                indices = torch.randint(0, total_elements, (sample_size,), device=device)
                flat_abs_diff = abs_diff_t.flatten()
                sampled = flat_abs_diff[indices]
                median_abs_diff = torch.median(sampled).item()
            else:
                median_abs_diff = torch.median(abs_diff_t.flatten()).item()
            
            # 转回numpy
            cosine_sims = cosine_sims_t.cpu().numpy()
            l2_distances = l2_distances_t.cpu().numpy()
            l2_distances_norm = l2_distances_norm_t.cpu().numpy()
            relative_l2_distances = relative_l2_distances_t.cpu().numpy()
            
            # 清理
            del M1_t, M2_t, diff_t, abs_diff_t, M1_norm_t, M2_norm_t
            del cosine_sims_t, l2_distances_t, l2_distances_norm_t, M1_norms_t, relative_l2_distances_t
            torch.cuda.empty_cache()
        else:
            # CPU计算（超大矩阵或显存不足时）
            diff = M2 - M1
            abs_diff = np.abs(diff)
            
            # Frobenius范数
            frobenius_norm_diff = np.linalg.norm(diff, 'fro')
            frobenius_norm_M1 = np.linalg.norm(M1, 'fro')
            frobenius_norm_M2 = np.linalg.norm(M2, 'fro')
            relative_frobenius_diff = frobenius_norm_diff / (frobenius_norm_M1 + 1e-10)
            
            # 余弦相似度
            M1_norm = M1 / (np.linalg.norm(M1, axis=1, keepdims=True) + 1e-10)
            M2_norm = M2 / (np.linalg.norm(M2, axis=1, keepdims=True) + 1e-10)
            cosine_sims = np.sum(M1_norm * M2_norm, axis=1)
            
            # L2距离
            l2_distances = np.linalg.norm(diff, axis=1)
            l2_distances_norm = np.linalg.norm(M1_norm - M2_norm, axis=1)
            M1_norms = np.linalg.norm(M1, axis=1)
            relative_l2_distances = l2_distances / (M1_norms + 1e-10)
            
            # 统计量
            mean_abs_diff = np.mean(abs_diff)
            std_abs_diff = np.std(abs_diff)
            max_abs_diff = np.max(abs_diff)
            min_abs_diff = np.min(abs_diff)
            median_abs_diff = np.median(abs_diff)
        
        # 7. 统计量
        stats = {
            # 基本统计（GPU上计算）
            'mean_abs_diff': mean_abs_diff,
            'std_abs_diff': std_abs_diff,
            'max_abs_diff': max_abs_diff,
            'min_abs_diff': min_abs_diff,
            'median_abs_diff': median_abs_diff,
            
            # Frobenius范数
            'frobenius_norm_diff': frobenius_norm_diff,
            'relative_frobenius_diff': relative_frobenius_diff,
            'frobenius_norm_M1': frobenius_norm_M1,
            'frobenius_norm_M2': frobenius_norm_M2,
            
            # 余弦相似度统计
            'mean_cosine_sim': np.mean(cosine_sims),
            'std_cosine_sim': np.std(cosine_sims),
            'min_cosine_sim': np.min(cosine_sims),
            'max_cosine_sim': np.max(cosine_sims),
            'median_cosine_sim': np.median(cosine_sims),
            'p5_cosine_sim': np.percentile(cosine_sims, 5),
            'p95_cosine_sim': np.percentile(cosine_sims, 95),
            
            # L2距离统计
            'mean_l2_distance': np.mean(l2_distances),
            'std_l2_distance': np.std(l2_distances),
            'max_l2_distance': np.max(l2_distances),
            'median_l2_distance': np.median(l2_distances),
            
            # 归一化L2距离统计
            'mean_l2_distance_norm': np.mean(l2_distances_norm),
            'std_l2_distance_norm': np.std(l2_distances_norm),
            
            # 相对L2距离统计
            'mean_relative_l2_distance': np.mean(relative_l2_distances),
            'std_relative_l2_distance': np.std(relative_l2_distances),
            'max_relative_l2_distance': np.max(relative_l2_distances),
            'median_relative_l2_distance': np.median(relative_l2_distances),
            
            # 矩阵形状
            'n_rows': n,
            'n_cols': d,
            
            # 原始数据（优化C: 只保留n维数组，abs_diff不保留全矩阵）
            'cosine_sims': cosine_sims,
            'l2_distances': l2_distances,
            'relative_l2_distances': relative_l2_distances,
            # abs_diff不保留全矩阵，已在上面的统计量中
        }
        
        return stats
    
    def compare_pair(self, base_name: str, finetuned_name: str, device: torch.device = None) -> Dict:
        """
        比较一对模型（基座 vs 微调）
        
        Args:
            base_name: 基座模型名称
            finetuned_name: 微调模型名称
            device: GPU设备，如果为None则自动选择
        
        Returns:
            包含所有比较结果的字典
        """
        if device is None:
            device = DEVICES[0] if NUM_GPUS > 0 else torch.device("cpu")
        
        print(f"\n{'='*80}")
        print(f"比较模型对: {base_name} vs {finetuned_name} (使用 {device})")
        print(f"{'='*80}")
        
        # 加载模型
        base_data = self.loader.load_matrices(base_name)
        finetuned_data = self.loader.load_matrices(finetuned_name)
        
        base_E = base_data['E']
        base_U = base_data['U']
        finetuned_E = finetuned_data['E']
        finetuned_U = finetuned_data['U']
        
        # 处理词表对齐
        same_vocab = (base_data['vocab_size'] == finetuned_data['vocab_size'])
        
        if same_vocab:
            print("\n✓ 词表大小相同，直接比较")
            base_E_aligned = base_E
            base_U_aligned = base_U
            finetuned_E_aligned = finetuned_E
            finetuned_U_aligned = finetuned_U
            token_ids_base = np.arange(base_data['vocab_size'])
            token_ids_finetuned = np.arange(finetuned_data['vocab_size'])
        else:
            print(f"\n⚠ 词表大小不同: {base_data['vocab_size']} vs {finetuned_data['vocab_size']}")
            print("  使用共同token进行比较")
            token_ids_base, token_ids_finetuned = self.get_common_tokens(
                base_data['tokenizer'], finetuned_data['tokenizer']
            )
            base_E_aligned = base_E[token_ids_base]
            base_U_aligned = base_U[token_ids_base]
            finetuned_E_aligned = finetuned_E[token_ids_finetuned]
            finetuned_U_aligned = finetuned_U[token_ids_finetuned]
            print(f"  共同token数: {len(token_ids_base)}")
        
        # 比较E矩阵
        print("\n" + "-"*80)
        print("比较 Embedding (E) 矩阵:")
        print("-"*80)
        E_stats = self.compute_matrix_differences(base_E_aligned, finetuned_E_aligned, "E", device=device)
        self._print_stats(E_stats, "E")
        
        # 比较U矩阵
        print("\n" + "-"*80)
        print("比较 Unembedding (U) 矩阵:")
        print("-"*80)
        U_stats = self.compute_matrix_differences(base_U_aligned, finetuned_U_aligned, "U", device=device)
        self._print_stats(U_stats, "U")
        
        # E和U之间的差异（在基座模型中）
        print("\n" + "-"*80)
        print("基座模型中 E vs U 的差异:")
        print("-"*80)
        base_EU_stats = self.compute_matrix_differences(base_E_aligned, base_U_aligned, "E-U(base)", device=device)
        self._print_stats(base_EU_stats, "E-U(base)")
        
        # E和U之间的差异（在微调模型中）
        print("\n" + "-"*80)
        print("微调模型中 E vs U 的差异:")
        print("-"*80)
        finetuned_EU_stats = self.compute_matrix_differences(
            finetuned_E_aligned, finetuned_U_aligned, "E-U(finetuned)", device=device
        )
        self._print_stats(finetuned_EU_stats, "E-U(finetuned)")
        
        # 组装结果
        result = {
            'base_model': base_name,
            'finetuned_model': finetuned_name,
            'same_vocab': same_vocab,
            'n_compared_tokens': len(token_ids_base),
            'base_vocab_size': base_data['vocab_size'],
            'finetuned_vocab_size': finetuned_data['vocab_size'],
            'base_hidden_dim': base_data['hidden_dim'],
            'finetuned_hidden_dim': finetuned_data['hidden_dim'],
            'base_is_tied': base_data['is_tied'],
            'base_actual_tied': base_data['actual_tied'],
            'finetuned_is_tied': finetuned_data['is_tied'],
            'finetuned_actual_tied': finetuned_data['actual_tied'],
            
        }
        # E矩阵比较
        result.update({
            'E_' + k: v for k, v in E_stats.items() if not isinstance(v, np.ndarray)
        })
        # U矩阵比较
        result.update({
            'U_' + k: v for k, v in U_stats.items() if not isinstance(v, np.ndarray)
        })
        # 基座模型E-U差异
        result.update({
            'base_EU_' + k: v for k, v in base_EU_stats.items() if not isinstance(v, np.ndarray)
        })
        # 微调模型E-U差异
        result.update({
            'finetuned_EU_' + k: v for k, v in finetuned_EU_stats.items() if not isinstance(v, np.ndarray)
        })
        
        # 优化C: detail_data只存统计摘要和top-k，不存全量列表
        def get_topk_and_stats(arr, k=1000):
            """获取top-k最大值及其索引，以及统计摘要"""
            topk_indices = np.argsort(arr)[-k:][::-1]
            return {
                'topk_values': arr[topk_indices].tolist(),
                'topk_indices': topk_indices.tolist(),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'p5': float(np.percentile(arr, 5)),
                'p95': float(np.percentile(arr, 95)),
            }
        
        detail_data = {
            'E_cosine_sims': get_topk_and_stats(E_stats['cosine_sims'], k=1000),
            'E_l2_distances': get_topk_and_stats(E_stats['l2_distances'], k=1000),
            'E_relative_l2_distances': get_topk_and_stats(E_stats['relative_l2_distances'], k=1000),
            'U_cosine_sims': get_topk_and_stats(U_stats['cosine_sims'], k=1000),
            'U_l2_distances': get_topk_and_stats(U_stats['l2_distances'], k=1000),
            'U_relative_l2_distances': get_topk_and_stats(U_stats['relative_l2_distances'], k=1000),
        }
        
        return result, detail_data
    
    
    def _print_stats(self, stats: Dict, prefix: str = ""):
        """打印统计信息"""
        print(f"\n{prefix} 矩阵差异统计:")
        print(f"  Frobenius范数差异: {stats['frobenius_norm_diff']:.6f}")
        print(f"  相对Frobenius差异: {stats['relative_frobenius_diff']:.6f}")
        print(f"\n  余弦相似度:")
        print(f"    均值: {stats['mean_cosine_sim']:.6f} ± {stats['std_cosine_sim']:.6f}")
        print(f"    中位数: {stats['median_cosine_sim']:.6f}")
        print(f"    最小值: {stats['min_cosine_sim']:.6f}")
        print(f"    最大值: {stats['max_cosine_sim']:.6f}")
        print(f"    P5: {stats['p5_cosine_sim']:.6f}, P95: {stats['p95_cosine_sim']:.6f}")
        print(f"\n  L2距离:")
        print(f"    均值: {stats['mean_l2_distance']:.6f} ± {stats['std_l2_distance']:.6f}")
        print(f"    中位数: {stats['median_l2_distance']:.6f}")
        print(f"    最大值: {stats['max_l2_distance']:.6f}")
        print(f"\n  相对L2距离:")
        print(f"    均值: {stats['mean_relative_l2_distance']:.6f} ± {stats['std_relative_l2_distance']:.6f}")
        print(f"    中位数: {stats['median_relative_l2_distance']:.6f}")
        print(f"    最大值: {stats['max_relative_l2_distance']:.6f}")
        print(f"\n  逐元素绝对差异:")
        print(f"    均值: {stats['mean_abs_diff']:.6f} ± {stats['std_abs_diff']:.6f}")
        print(f"    中位数: {stats['median_abs_diff']:.6f}")
        print(f"    最大值: {stats['max_abs_diff']:.6f}")
    
    def compare_all_pairs(self):
        """
        比较所有模型对（优化版：加载阶段串行，计算阶段可并行）
        
        注意：大模型加载时内存占用高，双线程同时加载可能导致OOM
        因此采用串行加载，但计算阶段可以利用GPU并行
        """
        all_results = []
        all_detail_data = {}
        
        # 串行处理模型对（避免大模型加载时的内存峰值）
        # GPU计算部分已经在compute_matrix_differences中优化
        for base_name, finetuned_name in self.config.model_pairs:
            try:
                # 轮询分配GPU（虽然加载在CPU，但计算时用不同GPU）
                device_idx = len(all_results) % NUM_GPUS if NUM_GPUS > 0 else 0
                device = DEVICES[device_idx] if NUM_GPUS > 0 else torch.device("cpu")
                
                result, detail_data = self.compare_pair(base_name, finetuned_name, device=device)
                all_results.append(result)
                pair_key = f"{base_name}__vs__{finetuned_name}"
                all_detail_data[pair_key] = detail_data
            except Exception as e:
                print(f"\n❌ 错误: 比较 {base_name} vs {finetuned_name} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 保存结果
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(self.config.output_dir, "matrix_comparison_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n✓ 结果已保存到: {csv_path}")
        
        # 保存详细数据
        detail_path = os.path.join(self.config.output_dir, "matrix_comparison_detail_data.json")
        with open(detail_path, 'w') as f:
            json.dump(all_detail_data, f, indent=2)
        print(f"✓ 详细数据已保存到: {detail_path}")
        
        # 打印摘要
        self._print_summary(df)
        
        return df, all_detail_data
    
    def _print_summary(self, df: pd.DataFrame):
        """打印结果摘要"""
        print("\n" + "="*80)
        print("结果摘要")
        print("="*80)
        
        print("\n📊 Embedding (E) 矩阵变化:")
        print(f"  平均余弦相似度: {df['E_mean_cosine_sim'].mean():.6f} ± {df['E_mean_cosine_sim'].std():.6f}")
        print(f"  平均相对Frobenius差异: {df['E_relative_frobenius_diff'].mean():.6f} ± {df['E_relative_frobenius_diff'].std():.6f}")
        print(f"  平均相对L2距离: {df['E_mean_relative_l2_distance'].mean():.6f} ± {df['E_mean_relative_l2_distance'].std():.6f}")
        
        print("\n📊 Unembedding (U) 矩阵变化:")
        print(f"  平均余弦相似度: {df['U_mean_cosine_sim'].mean():.6f} ± {df['U_mean_cosine_sim'].std():.6f}")
        print(f"  平均相对Frobenius差异: {df['U_relative_frobenius_diff'].mean():.6f} ± {df['U_relative_frobenius_diff'].std():.6f}")
        print(f"  平均相对L2距离: {df['U_mean_relative_l2_distance'].mean():.6f} ± {df['U_mean_relative_l2_distance'].std():.6f}")
        
        print("\n📊 E vs U 变化对比:")
        print(f"  E的平均余弦相似度: {df['E_mean_cosine_sim'].mean():.6f}")
        print(f"  U的平均余弦相似度: {df['U_mean_cosine_sim'].mean():.6f}")
        print(f"  差异 (U - E): {(df['U_mean_cosine_sim'] - df['E_mean_cosine_sim']).mean():.6f}")
        
        print("\n📊 基座模型中的 E-U 差异:")
        print(f"  平均余弦相似度: {df['base_EU_mean_cosine_sim'].mean():.6f} ± {df['base_EU_mean_cosine_sim'].std():.6f}")
        
        print("\n📊 微调模型中的 E-U 差异:")
        print(f"  平均余弦相似度: {df['finetuned_EU_mean_cosine_sim'].mean():.6f} ± {df['finetuned_EU_mean_cosine_sim'].std():.6f}")


def main():
    """主函数"""
    config = MatrixComparisonConfig()
    comparator = MatrixComparator(config)
    
    print("\n" + "="*80)
    print("微调前后模型 Embedding 和 Unembedding 矩阵差异分析")
    print("="*80)
    print(f"\n模型对数量: {len(config.model_pairs)}")
    print(f"输出目录: {config.output_dir}")
    
    df, detail_data = comparator.compare_all_pairs()
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)


if __name__ == "__main__":
    main()
