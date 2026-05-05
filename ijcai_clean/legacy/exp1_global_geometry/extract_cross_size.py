#!/usr/bin/env python3
"""
提取跨系列同规模对比数据
按7个规模组分别提取
"""

import csv
from pathlib import Path

# 按规模分组（用于跨系列对比）
MODELS_BY_SIZE = {
    "~0.5-0.6B": [
        "Qwen3-0.6B-Base", "Qwen3-0.6B", 
        "Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct"
    ],
    "~1-2B": [
        "Qwen3-1.7B-Base", "Qwen3-1.7B", 
        "Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct", 
        "Llama-3.2-1B", "Llama-3.2-1B-Instruct",
        "Gemma-2-2B", "Gemma-2-2B-Instruct"
    ],
    "~3-4B": [
        "Qwen3-4B-Base", "Qwen3-4B", 
        "Qwen2.5-3B", "Qwen2.5-3B-Instruct",
        "Llama-3.2-3B", "Llama-3.2-3B-Instruct"
    ],
    "~7-9B": [
        "Qwen3-8B-Base", "Qwen3-8B", 
        "Qwen2.5-7B", "Qwen2.5-7B-Instruct", 
        "Llama-3.1-8B", "Llama-3.1-8B-Instruct", 
        "Mistral-7B-v0.3", "Yi-1.5-9B",
        "Gemma-2-9B", "Gemma-2-9B-Instruct"
    ],
    "~14B": [
        "Qwen3-14B-Base", "Qwen3-14B", 
        "Qwen2.5-14B", "Qwen2.5-14B-Instruct"
    ],
    "~27-32B": [
        "Qwen3-32B", 
        "Qwen2.5-32B", "Qwen2.5-32B-Instruct", 
        "Gemma-2-27B", "Gemma-2-27B-Instruct"
    ],
    "~70-72B": [
        "Llama-3.1-70B-Base", "Llama-3.1-70B-Instruct", 
        "Qwen2.5-72B-Base", "Qwen2.5-72B-Instruct"
    ],
}


def load_csv_data(file_path):
    """加载CSV数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def extract_cross_size_data():
    """提取跨系列同规模对比数据"""
    
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
    
    print(f"📂 加载主实验结果: {main_file}")
    all_data, header = load_csv_data(main_file)
    print(f"   总数据量: {len(all_data)} 组")
    
    # 按规模组提取数据
    all_extracted = []
    
    for size_group, models in MODELS_BY_SIZE.items():
        print(f"\n{'='*70}")
        print(f"提取 {size_group} 规模组对比数据")
        print(f"{'='*70}")
        
        model_set = set(models)
        extracted_rows = []
        
        for row in all_data:
            model_a = row['model_a']
            model_b = row['model_b']
            
            # 检查两个模型是否都在该规模组中
            if model_a not in model_set or model_b not in model_set:
                continue
            
            # 只保留跨系列的对比（排除同系列内的对比）
            if row['family_a'] == row['family_b']:
                continue
            
            extracted_rows.append(row)
        
        print(f"✅ 提取到 {len(extracted_rows)} 组 {size_group} 规模组跨系列对比数据")
        
        # 统计信息
        families_involved = set()
        for row in extracted_rows:
            families_involved.add(row['family_a'])
            families_involved.add(row['family_b'])
        
        print(f"   涉及模型系列: {sorted(families_involved)}")
        print(f"   涉及模型数: {len(model_set)}")
        
        # 保存数据
        if extracted_rows:
            output_file = output_dir / f"exp1_cross_size_{size_group.replace('~', '').replace('-', '_')}.csv"
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(extracted_rows)
            
            print(f"💾 数据已保存: {output_file}")
            
            # 显示前几组配对
            print(f"\n📋 配对预览（前10组）:")
            for i, row in enumerate(extracted_rows[:10], 1):
                print(f"   {i:2d}. {row['model_a']:<35} vs {row['model_b']:<35} "
                      f"({row['family_a']} vs {row['family_b']}, "
                      f"cos_E={float(row['gcorr_E_cos_mean']):.4f})")
            
            if len(extracted_rows) > 10:
                print(f"   ... 还有 {len(extracted_rows) - 10} 组")
            
            all_extracted.extend(extracted_rows)
        else:
            print(f"⚠️  该规模组无跨系列对比数据")
    
    # 保存合并后的所有跨系列同规模对比数据
    if all_extracted:
        merged_file = output_dir / "exp1_cross_size_all.csv"
        with open(merged_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(all_extracted)
        
        print(f"\n{'='*70}")
        print(f"✅ 合并数据已保存: {merged_file}")
        print(f"   总跨系列同规模对比组数: {len(all_extracted)}")
        print(f"{'='*70}")


if __name__ == "__main__":
    extract_cross_size_data()
