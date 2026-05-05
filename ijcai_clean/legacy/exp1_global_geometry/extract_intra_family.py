#!/usr/bin/env python3
"""
提取系列内部全模型互比数据
按模型系列分别提取，并合并Qwen2.5和Llama的大小模型分组数据
"""

import csv
from pathlib import Path
from collections import defaultdict

# Base-Instruct 配对集合（需要排除）
BASE_INSTRUCT_PAIRS_SET = {
    ("Qwen3-0.6B-Base", "Qwen3-0.6B"),
    ("Qwen3-1.7B-Base", "Qwen3-1.7B"),
    ("Qwen3-4B-Base", "Qwen3-4B"),
    ("Qwen3-8B-Base", "Qwen3-8B"),
    ("Qwen3-14B-Base", "Qwen3-14B"),
    ("Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct"),
    ("Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct"),
    ("Qwen2.5-3B", "Qwen2.5-3B-Instruct"),
    ("Qwen2.5-7B", "Qwen2.5-7B-Instruct"),
    ("Qwen2.5-14B", "Qwen2.5-14B-Instruct"),
    ("Qwen2.5-32B", "Qwen2.5-32B-Instruct"),
    ("Qwen2.5-72B-Base", "Qwen2.5-72B-Instruct"),
    ("Llama-3.2-1B", "Llama-3.2-1B-Instruct"),
    ("Llama-3.2-3B", "Llama-3.2-3B-Instruct"),
    ("Llama-3.1-8B", "Llama-3.1-8B-Instruct"),
    ("Llama-3.1-70B-Base", "Llama-3.1-70B-Instruct"),
    ("Gemma-2-2B", "Gemma-2-2B-Instruct"),
    ("Gemma-2-9B", "Gemma-2-9B-Instruct"),
    ("Gemma-2-27B", "Gemma-2-27B-Instruct"),
}

# 各系列的完整模型列表
FAMILY_MODELS = {
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


def load_csv_data(file_path):
    """加载CSV数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def is_base_instruct_pair(model_a, model_b):
    """判断是否是Base-Instruct配对"""
    return (model_a, model_b) in BASE_INSTRUCT_PAIRS_SET or \
           (model_b, model_a) in BASE_INSTRUCT_PAIRS_SET


def is_same_family_pair(row, family):
    """判断是否是同一系列内的配对"""
    return row['family_a'] == family and row['family_b'] == family


def extract_intra_family_data():
    """提取系列内部全模型互比数据"""
    
    base_dir = Path(__file__).parent
    results_dir = base_dir / "results"
    output_dir = base_dir / "useful_data"
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    
    # 加载主实验结果
    main_file = results_dir / "exp1_global_v4_merged.csv"
    supplement_file = results_dir / "exp1_cross_size_20260129_015053.csv"
    
    if not main_file.exists():
        print(f"❌ 文件不存在: {main_file}")
        return
    
    print(f"📂 加载主实验结果: {main_file}")
    main_data, header = load_csv_data(main_file)
    print(f"   主实验数据量: {len(main_data)} 组")
    
    # 加载补充实验结果（跨大小分组）
    supplement_data = []
    if supplement_file.exists():
        print(f"📂 加载补充实验结果: {supplement_file}")
        supplement_data, _ = load_csv_data(supplement_file)
        print(f"   补充实验数据量: {len(supplement_data)} 组")
    else:
        print(f"⚠️  补充实验结果文件不存在: {supplement_file}")
    
    # 合并所有数据
    all_data = main_data + supplement_data
    print(f"   总数据量: {len(all_data)} 组")
    
    # 按系列提取数据
    for family, models in FAMILY_MODELS.items():
        print(f"\n{'='*70}")
        print(f"提取 {family} 系列内部对比数据")
        print(f"{'='*70}")
        
        family_set = set(models)
        extracted_rows = []
        
        for row in all_data:
            model_a = row['model_a']
            model_b = row['model_b']
            
            # 检查是否是同一系列内的配对
            if not is_same_family_pair(row, family):
                continue
            
            # 检查两个模型是否都在该系列中
            if model_a not in family_set or model_b not in family_set:
                continue
            
            # 排除Base-Instruct配对（实验1.1中已做）
            if is_base_instruct_pair(model_a, model_b):
                continue
            
            extracted_rows.append(row)
        
        print(f"✅ 提取到 {len(extracted_rows)} 组 {family} 系列内部对比数据")
        
        # 统计信息
        model_set = set()
        for row in extracted_rows:
            model_set.add(row['model_a'])
            model_set.add(row['model_b'])
        
        print(f"   涉及模型数: {len(model_set)}")
        print(f"   模型列表: {sorted(model_set)}")
        
        # 保存数据
        output_file = output_dir / f"exp1_intra_family_{family.lower().replace('.', '')}.csv"
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(extracted_rows)
        
        print(f"💾 数据已保存: {output_file}")
        
        # 显示前几组配对
        print(f"\n📋 配对预览（前10组）:")
        for i, row in enumerate(extracted_rows[:10], 1):
            print(f"   {i:2d}. {row['model_a']:<35} vs {row['model_b']:<35} "
                  f"(cos_E={float(row['gcorr_E_cos_mean']):.4f}, cos_U={float(row['gcorr_U_cos_mean']):.4f})")
        
        if len(extracted_rows) > 10:
            print(f"   ... 还有 {len(extracted_rows) - 10} 组")


if __name__ == "__main__":
    extract_intra_family_data()
