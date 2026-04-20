import os
import json
import time
import yaml
import concurrent.futures
from modelscope import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

def extract_and_save_emb_matrices(model_dir: str, model_name: str):
    """
    遍历指定目录中的所有 safetensors/bin 文件，提取 embedding 和 head 权重并保存到单独的文件。
    """
    print(f"🛠️  开始为 {model_name} 提取 embedding / unembedding 矩阵...")
    
    # 查找可能的层名
    target_keys = {
        "embed_tokens.weight", "model.embed_tokens.weight", "transformer.wte.weight", "transformer.word_embeddings.weight",
        "lm_head.weight", "head.weight", "output.weight"
    }
    extracted_tensors = {}
    
    # 查找目录下所有 safetensors 文件
    file_list = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]
    if not file_list:
        print(f"⚠️  未找到 safetensors 文件，也许使用的是 pytorch_model.bin 格式，这里主要做基于 safetensors 的快速提取。")
        return False
        
    for file_name in file_list:
        file_path = os.path.join(model_dir, file_name)
        try:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                keys = f.keys()
                for k in keys:
                    if k in target_keys or 'embed_tokens' in k or 'lm_head' in k:
                        if k not in extracted_tensors:
                            print(f"    🌟 找到目标张量: {k} (来源: {file_name})")
                            extracted_tensors[k] = f.get_tensor(k)
        except Exception as e:
            print(f"    ⚠️ 读取 {file_name} 失败: {e}")
            continue

    if not extracted_tensors:
        print(f"❌  未能在 {model_dir} 中找到相关的 emb/unemb 权重。")
        return False
        
    # 分析并保存提取的维度以及 tie 状态到 JSON 文件中
    config_path = os.path.join(model_dir, "config.json")
    is_tied = None
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                model_config = json.load(f)
            # 这里先不要提供默认值 False，因为如果 config 没写，可能是由框架模型默认属性决定的（比如 Gemma 默认是 true）
            is_tied = model_config.get("tie_word_embeddings")
        except Exception as e:
            print(f"    ⚠️ 读取 config.json 失败: {e}")

    # 分离出 emb 和 head 的原始 key
    embed_keys = [k for k in extracted_tensors.keys() if 'embed' in k or 'wte' in k]
    head_keys = [k for k in extracted_tensors.keys() if 'head' in k or 'output' in k]

    # 【解决 Gemma 的最主要问题】：如果 config.json 没写 tie 状态，而且提取出来的所有权重里根本没有 head_keys，说明必定是 tied
    if is_tied is None:
        is_tied = (len(head_keys) == 0)

    tensor_info = {}
    for k, t in extracted_tensors.items():
        tensor_info[k] = list(t.shape)

    # 【解决整体标准化问题】：无论它物理上怎么储存，这里补齐统一的双维度字段，供下游无脑读取防报错
    standardized_dims = {}
    if embed_keys:
        standardized_dims["embed"] = list(extracted_tensors[embed_keys[0]].shape)
    
    if head_keys:
        standardized_dims["lm_head"] = list(extracted_tensors[head_keys[0]].shape)
    elif is_tied and embed_keys:
        # 如果是 tied，且物理文件/键值确实少了 lm_head 这一项，则主动映射过去补全！
        standardized_dims["lm_head"] = standardized_dims["embed"]
    # 【解决 Gemma 缺 lm_head 的主要问题】如果不是 tied，且少了 lm_head 这一项，说明漏提了，从 embed 复制过去（这种通常也是权重相等的）
    elif not is_tied and embed_keys:
        standardized_dims["lm_head"] = standardized_dims["embed"]

    metadata = {
        "model_name": model_name,
        "tie_word_embeddings": bool(is_tied),
        "standardized_dims": standardized_dims,
        "raw_tensors_dimensions": tensor_info
    }

    info_path = os.path.join(model_dir, "extracted_embeddings_info.json")
    try:
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        print(f"✅  元数据信息已保存至: {info_path}")
    except Exception as e:
        print(f"❌  保存元数据 JSON 失败: {e}")

    # 保存结果
    output_path = os.path.join(model_dir, "extracted_embeddings.safetensors")
    try:
        save_file(extracted_tensors, output_path)
        print(f"✅  提取成功！文件已保存至: {output_path} (大小: {os.path.getsize(output_path)/1024/1024:.2f} MB)")
        return True
    except Exception as e:
        print(f"❌  保存失败: {e}")
        return False

def download_emb_only_with_retry(model_name: str, repo_id: str, cache_dir: str, max_retries: int) -> tuple[bool, str]:
    """
    【精准下载模式】只下载 Tokenizer、配置、Embedding 和 Unembedding (LM Head) 权重
    """
    if not repo_id:
        return False, f"❌ 失败: 找不到 '{model_name}' 的 repo_id。"

    print(f"\n🔬 [精简模式] 正在解析 {model_name} 的权重分布...")
    
    for attempt in range(1, max_retries + 1):
        try:
            # 第一步：极其快速地只下载轻量级的元数据、Tokenizer 和权重索引文件
            meta_dir = snapshot_download(
                model_id=repo_id, 
                cache_dir=cache_dir,
                ignore_patterns=['*.safetensors', '*.bin', '*.pt', '*.pth', 'original/*']
            )
            
            # 寻找权重索引文件
            index_path = os.path.join(meta_dir, "model.safetensors.index.json")
            if not os.path.exists(index_path):
                index_path = os.path.join(meta_dir, "pytorch_model.bin.index.json")

            target_files = ['*.json', '*.model', '*.tiktoken', 'tokenizer*'] 
            
            if os.path.exists(index_path):
                with open(index_path, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                    
                weight_map = index_data.get("weight_map", {})
                shard_set = set()
                
                for param_name, shard_file in weight_map.items():
                    name_lower = param_name.lower()
                    if "embed" in name_lower or "head" in name_lower or "wte" in name_lower:
                        shard_set.add(shard_file)
                
                print(f"🎯 成功定位到首尾特征矩阵，分布在分块: {list(shard_set)}")
                target_files.extend(list(shard_set))
            else:
                print(f"ℹ️  {model_name} 可能是小模型或本身未分块，将下载基础 safetensors。")
                target_files.append('*.safetensors')
                target_files.append('*.bin')
                target_files.append('*.pth')

            # 第二步：精准下载对应的巨大的权重文件
            print(f"⬇️ 开始只下载指定的 Embedding/Head 文件: {target_files}")
            final_dir = snapshot_download(
                model_id=repo_id, 
                cache_dir=cache_dir,
                allow_patterns=target_files 
            )
            
            # 第三步：下载完成后直接提取文件中的 emb/unemb
            extract_and_save_emb_matrices(final_dir, model_name)
            
            return True, f"✅ 成功完成全流程: {model_name} -> {final_dir}"
            
        except Exception as e:
            if attempt < max_retries:
                wait_time = 10 * attempt 
                print(f"⚠️ 异常: {model_name} (第 {attempt}/{max_retries} 次尝试)。等待 {wait_time}s... 原因: {e}")
                time.sleep(wait_time)
            else:
                return False, f"❌ 彻底失败: {model_name}。最终错误: {e}"

def main():
#    yaml_path = "/root/shared-nvme/tools/models.yaml"
    yaml_path = "/home/wz/projects/mypro/get_useful/models.yaml"
    if not os.path.exists(yaml_path):
        print(f"❌ 找不到配置文件: {yaml_path}")
        return

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
        
    base_cache_dir = config_data.get("config", {}).get("cache_dir", "./downloaded_models")
    max_retries = config_data.get("config", {}).get("max_retries", 3)
    max_workers = config_data.get("config", {}).get("max_workers", 3)
    repo_mappings = config_data.get("model_repo_ids", {})

    print(f"🚀 开始批量 [下载并提取] 模型 (缓存于: {base_cache_dir}, 并发线程数: {max_workers}) ...")
    
    # 获取所有的模型全集进行并发下载
    target_models = list(repo_mappings.keys())
    
    valid_models = [m for m in target_models if m in repo_mappings]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {
            executor.submit(
                download_emb_only_with_retry, 
                model_name, 
                repo_mappings[model_name], 
                base_cache_dir, 
                max_retries
            ): model_name for model_name in valid_models
        }
        
        for future in concurrent.futures.as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                success, msg = future.result()
                print(msg)
            except Exception as exc:
                print(f"❌ {model_name} 下载/提取任务抛出异常: {exc}")
            print("-" * 50)

    print(f"🎉 所有下载和提取任务完成！正在生成全局汇总报告...")
    
    summary_data = {}
    # 遍历缓存目录，搜集所有的 json 摘要文件
    for root, dirs, files in os.walk(base_cache_dir):
        if "extracted_embeddings_info.json" in files:
            info_path = os.path.join(root, "extracted_embeddings_info.json")
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    model_name = data.get("model_name", "Unknown")
                    summary_data[model_name] = {
                        "tie_word_embeddings": data.get("tie_word_embeddings"),
                        "standardized_dims": data.get("standardized_dims", {}),
                        "raw_tensors_dimensions": data.get("raw_tensors_dimensions", data.get("tensors_dimensions"))
                    }
            except Exception as e:
                print(f"⚠️ 读取 {info_path} 失败: {e}")

    # 将汇总文件保存到脚本所在的同级目录中（与 models.yaml 同级）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(script_dir, "all_models_summary.json")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
        print(f"✅ 全局汇总文件已生成: {summary_path}")
    except Exception as e:
        print(f"❌ 生成全局汇总文件失败: {e}")

if __name__ == "__main__":
    main()