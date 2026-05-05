#!/usr/bin/env python3
"""
补充实验：Qwen2.5 和 Llama 系列的跨大小分组对比
============================================
优化版：每次只加载 2 个模型，避免内存爆炸

Qwen2.5: small (10) × large (4) = 40 组
Llama: small (6) × large (2) = 12 组
总计: 52 组对比
"""

import os
import sys
import time
import itertools
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_exp1_v4 import (
    Config, 
    ModelLoader, 
    TokenSampler, 
    compute_gcorr_gpu_streaming_v4,
    BASE_INSTRUCT_PAIRS_SET,
    NUM_GPUS,
    DEVICES,
    GPU_LOCKS,
)

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# 跨大小分组配置
# =============================================================================
QWEN25_SMALL = [
    "Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B", "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B", "Qwen2.5-7B-Instruct",
    "Qwen2.5-14B", "Qwen2.5-14B-Instruct",
]

QWEN25_LARGE = [
    "Qwen2.5-32B", "Qwen2.5-32B-Instruct",
    "Qwen2.5-72B-Base", "Qwen2.5-72B-Instruct",
]

LLAMA_SMALL = [
    "Llama-3.2-1B", "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B", "Llama-3.2-3B-Instruct",
    "Llama-3.1-8B", "Llama-3.1-8B-Instruct",
]

LLAMA_LARGE = [
    "Llama-3.1-70B-Base", "Llama-3.1-70B-Instruct",
]


class PairwiseCalculator:
    """每次只加载 2 个模型，计算完立即释放"""
    
    def __init__(self, config: Config):
        self.config = config
        self.loader = ModelLoader(config)
        self.sampler = TokenSampler(config)
    
    def compute_pair(self, model_a_name: str, model_b_name: str) -> dict:
        """计算单个模型对的 GCorr（带 bootstrap）"""
        print(f"\n计算: {model_a_name} vs {model_b_name}")
        
        # 计算前强制清理所有 GPU 显存
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 加载两个模型
        data_a = self.loader.load(model_a_name)
        data_b = self.loader.load(model_b_name)
        
        # 获取有效 token ids（同 tokenizer）
        valid_ids = self.sampler.get_valid_token_ids(data_a['tokenizer'])
        n_sample = min(self.config.n_tokens, len(valid_ids))
        
        # batch_size 已修复，可以双卡并行（每张卡独立处理不同的 bootstrap）
        max_dim = max(data_a['hidden_dim'], data_b['hidden_dim'])
        print(f"  [双卡并行] max_dim={max_dim}")
        
        bootstrap_results = []
        
        def compute_single_bootstrap(b, gpu_id):
            device = DEVICES[gpu_id]
            seed = self.config.random_seed + b
            
            # 采样
            sample_ids = self.sampler.sample_tokens(valid_ids, n_sample, seed)
            
            E_a = data_a['E'][sample_ids]
            E_b = data_b['E'][sample_ids]
            U_a = data_a['U'][sample_ids]
            U_b = data_b['U'][sample_ids]
            
            with GPU_LOCKS[gpu_id]:
                gcorr_E = compute_gcorr_gpu_streaming_v4(E_a, E_b, self.config.n_pairs, seed * 1000, device=device)
                gcorr_U = compute_gcorr_gpu_streaming_v4(U_a, U_b, self.config.n_pairs, seed * 1000 + 1, device=device)
            
            return {
                'gcorr_E_cos': gcorr_E['gcorr_cos'],
                'gcorr_E_euc': gcorr_E['gcorr_euc'],
                'gcorr_E_euc2': gcorr_E['gcorr_euc2'],
                'gcorr_U_cos': gcorr_U['gcorr_cos'],
                'gcorr_U_euc': gcorr_U['gcorr_euc'],
                'gcorr_U_euc2': gcorr_U['gcorr_euc2'],
            }
        
        # 双卡并行：每张卡处理一半的 bootstrap
        all_tasks = [(b, b % NUM_GPUS) for b in range(self.config.n_bootstrap)]
        with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
            futures = [executor.submit(compute_single_bootstrap, b, gpu_id) for b, gpu_id in all_tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Bootstrap"):
                bootstrap_results.append(future.result())
        
        # 汇总统计
        def summarize(key):
            values = np.array([r[key] for r in bootstrap_results])
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'se': np.std(values) / np.sqrt(len(values)),
                'ci95_low': np.percentile(values, 2.5),
                'ci95_high': np.percentile(values, 97.5),
            }
        
        E_cos = summarize('gcorr_E_cos')
        E_euc = summarize('gcorr_E_euc')
        E_euc2 = summarize('gcorr_E_euc2')
        U_cos = summarize('gcorr_U_cos')
        U_euc = summarize('gcorr_U_euc')
        U_euc2 = summarize('gcorr_U_euc2')
        
        result = {
            'model_a': model_a_name,
            'model_b': model_b_name,
            'family_a': data_a['family'],
            'family_b': data_b['family'],
            'same_family': data_a['family'] == data_b['family'],
            'same_tokenizer': data_a['vocab_size'] == data_b['vocab_size'],
            'n_tokens': n_sample,
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
            # E 平方欧氏
            'gcorr_E_euc2_mean': E_euc2['mean'],
            'gcorr_E_euc2_std': E_euc2['std'],
            'gcorr_E_euc2_se': E_euc2['se'],
            'gcorr_E_euc2_ci95_low': E_euc2['ci95_low'],
            'gcorr_E_euc2_ci95_high': E_euc2['ci95_high'],
            # U 平方欧氏
            'gcorr_U_euc2_mean': U_euc2['mean'],
            'gcorr_U_euc2_std': U_euc2['std'],
            'gcorr_U_euc2_se': U_euc2['se'],
            'gcorr_U_euc2_ci95_low': U_euc2['ci95_low'],
            'gcorr_U_euc2_ci95_high': U_euc2['ci95_high'],
        }
        
        print(f"  结果: cos(E)={E_cos['mean']:.4f}, cos(U)={U_cos['mean']:.4f}")
        
        # 强制清理内存
        self.loader.cache.clear()
        del data_a, data_b
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc
        gc.collect()
        
        return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='补充实验：跨大小分组对比（内存优化版）')
    parser.add_argument('--n_tokens', type=int, default=20000)
    parser.add_argument('--n_bootstrap', type=int, default=100)
    parser.add_argument('--n_pairs', type=int, default=5000000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--group', type=str, default='all', 
                        choices=['all', 'qwen25', 'llama'],
                        help='运行哪个系列的跨组对比')
    parser.add_argument('--resume', action='store_true',
                        help='从断点续传')
    args = parser.parse_args()
    
    config = Config(
        n_tokens=args.n_tokens,
        n_bootstrap=args.n_bootstrap,
        n_pairs=args.n_pairs,
        random_seed=args.seed,
    )
    
    print("="*80)
    print("补充实验：跨大小分组对比（内存优化版 - 每次只加载2个模型）")
    print(f"参数: n_tokens={config.n_tokens}, n_bootstrap={config.n_bootstrap}, n_pairs={config.n_pairs:,}")
    print(f"GPU 数量: {NUM_GPUS}")
    print("="*80)
    
    calculator = PairwiseCalculator(config)
    all_results = []
    completed_pairs = set()
    start_time = time.time()
    
    # 检查断点续传
    progress_csv = os.path.join(config.output_dir, "exp1_cross_size_progress.csv")
    if args.resume and os.path.exists(progress_csv):
        existing_df = pd.read_csv(progress_csv)
        all_results = existing_df.to_dict('records')
        completed_pairs = set(zip(existing_df['model_a'], existing_df['model_b']))
        print(f"\n[断点续传] 已加载 {len(completed_pairs)} 组已完成的结果")
    
    # 生成所有需要计算的模型对
    pairs_to_compute = []
    
    if args.group in ['all', 'qwen25']:
        for small, large in itertools.product(QWEN25_SMALL, QWEN25_LARGE):
            pairs_to_compute.append((small, large))
    
    if args.group in ['all', 'llama']:
        for small, large in itertools.product(LLAMA_SMALL, LLAMA_LARGE):
            pairs_to_compute.append((small, large))
    
    # 过滤已完成的对
    remaining_pairs = [(a, b) for a, b in pairs_to_compute if (a, b) not in completed_pairs]
    
    print(f"\n总模型对数: {len(pairs_to_compute)}")
    print(f"已完成: {len(completed_pairs)}")
    print(f"待计算: {len(remaining_pairs)}")
    
    for i, (model_a, model_b) in enumerate(remaining_pairs):
        print(f"\n[{len(completed_pairs) + i + 1}/{len(pairs_to_compute)}]")
        try:
            result = calculator.compute_pair(model_a, model_b)
            all_results.append(result)
            
            # 每完成一组保存一次
            df = pd.DataFrame(all_results)
            df.to_csv(progress_csv, index=False)
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            
            # 异常时也要清理资源，防止显存泄漏影响下一组
            calculator.loader.cache.clear()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed/60:.2f} 分钟")
    print(f"完成对比组数: {len(all_results)}")
    
    # 最终保存
    if all_results:
        df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(config.output_dir, f"exp1_cross_size_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n最终结果已保存: {csv_path}")


if __name__ == "__main__":
    main()
