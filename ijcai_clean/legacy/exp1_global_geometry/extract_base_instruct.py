#!/usr/bin/env python3
"""
提取 Base vs Instruct 配对对比数据
"""

import csv
from pathlib import Path

# Base-Instruct 配对列表（实验 1.1）
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

def extract_base_instruct_data():
    """提取 Base vs Instruct 配对数据"""
    
    base_dir = Path(__file__).parent
    results_dir = base_dir / "results"
    output_dir = base_dir / "useful_data"
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    
    # 加载主实验结果
    main_file = results_dir / "exp1_global_v4_merged.csv"
    
    if not main_file.exists():
        print(f"❌ 文件不存在: {main_file}")
        return
    
    print(f"📂 加载数据: {main_file}")
    
    # 创建配对集合（考虑顺序）
    pair_set = set()
    for a, b in BASE_INSTRUCT_PAIRS:
        pair_set.add((a, b))
        pair_set.add((b, a))  # 也考虑反向
    
    # 读取CSV并筛选
    extracted_rows = []
    total_rows = 0
    
    with open(main_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        
        for row in reader:
            total_rows += 1
            model_a = row['model_a']
            model_b = row['model_b']
            
            if (model_a, model_b) in pair_set:
                extracted_rows.append(row)
    
    print(f"   总数据量: {total_rows} 组")
    print(f"✅ 提取到 {len(extracted_rows)} 组 Base vs Instruct 配对数据")
    
    # 按系列分组统计
    family_count = {}
    for row in extracted_rows:
        family = row['family_a']
        family_count[family] = family_count.get(family, 0) + 1
    
    print("\n📊 按系列统计:")
    for family, count in sorted(family_count.items()):
        print(f"   {family}: {count} 组")
    
    # 保存数据
    output_file = output_dir / "exp1_base_vs_instruct.csv"
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(extracted_rows)
    
    print(f"\n💾 数据已保存: {output_file}")
    
    # 显示前几行
    print("\n📋 数据预览（前5组）:")
    print(f"{'模型A':<30} {'模型B':<30} {'系列':<12} {'GCorr_E_cos':<15} {'GCorr_U_cos':<15}")
    print("-" * 100)
    for i, row in enumerate(extracted_rows[:5]):
        print(f"{row['model_a']:<30} {row['model_b']:<30} {row['family_a']:<12} "
              f"{float(row['gcorr_E_cos_mean']):<15.6f} {float(row['gcorr_U_cos_mean']):<15.6f}")
    
    return extracted_rows


if __name__ == "__main__":
    extract_base_instruct_data()
