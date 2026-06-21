import os
import sys
import json
import time
import shutil
from pathlib import Path

import yaml
import concurrent.futures
import torch
from modelscope import snapshot_download

_TMP_REPO = Path(__file__).resolve().parents[1]
if str(_TMP_REPO) not in sys.path:
    sys.path.insert(0, str(_TMP_REPO))
from tools.paths import repository_root  # noqa: E402

_REPO_ROOT = repository_root(__file__)
from safetensors import safe_open
from safetensors.torch import save_file

EXTRACTED_WEIGHTS_FILENAME = "extracted_embeddings.safetensors"
EXTRACTED_INFO_FILENAME = "extracted_embeddings_info.json"
WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")

def is_target_tensor_key(key: str) -> bool:
    key_lower = key.lower()
    return (
        key in {
            "embed.weight",
            "embed_tokens.weight",
            "model.embed_tokens.weight",
            "model.tok_embeddings.weight",
            "transformer.wte.weight",
            "transformer.word_embeddings.weight",
            "lm_head.weight",
            "head.weight",
            "output.weight",
        }
        or key_lower == "embed.weight"
        or "embed_tokens" in key_lower
        or "tok_embeddings" in key_lower
        or "word_embeddings" in key_lower
        or "lm_head" in key_lower
        or "transformer.wte" in key_lower
    )

def unwrap_state_dict(data):
    """
    兼容常见的 PyTorch checkpoint 包装结构，尽量拿到真正的 state_dict。
    """
    if not isinstance(data, dict):
        return {}

    candidate_keys = ("state_dict", "model_state_dict", "model", "module")
    for key in candidate_keys:
        value = data.get(key)
        if isinstance(value, dict):
            return value
    return data

def extract_and_save_emb_matrices(model_dir: str, model_name: str):
    """
    遍历指定目录中的权重文件，提取 embedding 和 head 权重并保存到单独的文件。
    """
    print(f"🛠️  开始为 {model_name} 提取 embedding / unembedding 矩阵...")

    extracted_tensors = {}

    # 查找目录下所有可能的权重文件，但排除上一次跑留下的提取产物，避免自循环扫描
    file_list = [
        f for f in os.listdir(model_dir)
        if f.endswith(WEIGHT_FILE_SUFFIXES) and f != EXTRACTED_WEIGHTS_FILENAME
    ]
    if not file_list:
        print(f"⚠️  未找到可识别的权重文件（safetensors/bin/pt/pth）。")
        return False

    for file_name in file_list:
        file_path = os.path.join(model_dir, file_name)
        try:
            if file_name.endswith(".safetensors"):
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    keys = list(f.keys())
                    for k in keys:
                        if is_target_tensor_key(k) and k not in extracted_tensors:
                            print(f"    🌟 找到目标张量: {k} (来源: {file_name})")
                            extracted_tensors[k] = f.get_tensor(k)
            else:
                state_dict = unwrap_state_dict(torch.load(file_path, map_location="cpu"))
                for k, tensor in state_dict.items():
                    if torch.is_tensor(tensor) and is_target_tensor_key(k):
                        if k not in extracted_tensors:
                            print(f"    🌟 找到目标张量: {k} (来源: {file_name})")
                            extracted_tensors[k] = tensor
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

    # 分离出 emb 和 head 的原始 key。DeepSeek-V3/R1 等模型里可能还有
    # model.layers.*.embed_tokens.weight 这类附加模块权重；标准输入 embedding
    # 必须优先选择 model.embed_tokens.weight。
    embed_keys = [k for k in extracted_tensors.keys() if 'embed' in k.lower() or 'wte' in k.lower()]
    head_keys = [k for k in extracted_tensors.keys() if 'head' in k.lower() or 'output' in k.lower()]

    def choose_embed_key(keys):
        exact_priority = [
            "embed.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "transformer.wte.weight",
            "transformer.word_embeddings.weight",
            "model.tok_embeddings.weight",
        ]
        for key in exact_priority:
            if key in keys:
                return key
        return keys[0] if keys else None

    def choose_head_key(keys):
        exact_priority = ["lm_head.weight", "output.weight", "head.weight", "model.output.weight"]
        for key in exact_priority:
            if key in keys:
                return key
        return keys[0] if keys else None

    embed_key = choose_embed_key(embed_keys)
    head_key = choose_head_key(head_keys)

    # 【解决 Gemma 的最主要问题】：如果 config.json 没写 tie 状态，而且提取出来的所有权重里根本没有 head_keys，说明必定是 tied
    if is_tied is None:
        is_tied = (len(head_keys) == 0)

    tensor_info = {}
    for k, t in extracted_tensors.items():
        tensor_info[k] = list(t.shape)

    # 【解决整体标准化问题】：无论它物理上怎么储存，这里补齐统一的双维度字段，供下游无脑读取防报错
    standardized_dims = {}
    standardized_sources = {}
    if embed_key:
        standardized_dims["embed"] = list(extracted_tensors[embed_key].shape)
        standardized_sources["embed"] = embed_key

    if head_key:
        standardized_dims["lm_head"] = list(extracted_tensors[head_key].shape)
        standardized_sources["lm_head"] = head_key
    elif is_tied and embed_key:
        # 如果是 tied，且物理文件/键值确实少了 lm_head 这一项，则主动映射过去补全！
        standardized_dims["lm_head"] = standardized_dims["embed"]
        standardized_sources["lm_head"] = embed_key
    # 【解决 Gemma 缺 lm_head 的主要问题】如果不是 tied，且少了 lm_head 这一项，说明漏提了，从 embed 复制过去（这种通常也是权重相等的）
    elif not is_tied and embed_key:
        standardized_dims["lm_head"] = standardized_dims["embed"]
        standardized_sources["lm_head"] = embed_key

    metadata = {
        "model_name": model_name,
        "tie_word_embeddings": bool(is_tied),
        "standardized_dims": standardized_dims,
        "standardized_sources": standardized_sources,
        "raw_embed_keys": embed_keys,
        "raw_lm_head_keys": head_keys,
        "raw_tensors_dimensions": tensor_info
    }

    info_path = os.path.join(model_dir, EXTRACTED_INFO_FILENAME)
    try:
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        print(f"✅  元数据信息已保存至: {info_path}")
    except Exception as e:
        print(f"❌  保存元数据 JSON 失败: {e}")

    output_path = os.path.join(model_dir, EXTRACTED_WEIGHTS_FILENAME)
    try:
        save_file(extracted_tensors, output_path)
        print(f"✅  提取成功！文件已保存至: {output_path} (大小: {os.path.getsize(output_path)/1024/1024:.2f} MiB)")
        return True
    except Exception as e:
        print(f"❌  保存失败: {e}")
        return False


def verify_extracted_safetensors(model_dir: str, model_name: str) -> bool:
    """
    重新打开提取后的 safetensors，逐键加载并对照 info.json 中记录的 shape，
    任何缺键、形状不匹配或读取异常都视作校验失败，不允许后续删除原始权重。
    """
    extract_path = os.path.join(model_dir, EXTRACTED_WEIGHTS_FILENAME)
    info_path = os.path.join(model_dir, EXTRACTED_INFO_FILENAME)
    if not os.path.exists(extract_path) or not os.path.exists(info_path):
        print(f"❌ [校验] {model_name} 缺少提取文件或元信息：{extract_path} / {info_path}")
        return False

    try:
        with open(info_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        expected_shapes = metadata.get("raw_tensors_dimensions", {}) or {}
        if not expected_shapes:
            print(f"❌ [校验] {model_name} 元信息里没有 raw_tensors_dimensions，无法对照。")
            return False

        with safe_open(extract_path, framework="pt", device="cpu") as f:
            actual_keys = set(f.keys())
            missing = [k for k in expected_shapes if k not in actual_keys]
            if missing:
                print(f"❌ [校验] {model_name} 缺少张量: {missing}")
                return False
            for k, expected in expected_shapes.items():
                tensor = f.get_tensor(k)
                if list(tensor.shape) != list(expected):
                    print(
                        f"❌ [校验] {model_name} 张量 {k} 形状不一致："
                        f"实际 {list(tensor.shape)} vs 预期 {list(expected)}"
                    )
                    return False
        print(f"✅ [校验] {model_name} 提取文件可正常加载，所有张量形状匹配。")
        return True
    except Exception as e:
        print(f"❌ [校验] {model_name} 加载失败: {e}")
        return False


def backup_extracted_files(model_dir: str, model_name: str, backup_root: str) -> bool:
    """
    把 extracted_embeddings.safetensors 和 extracted_embeddings_info.json
    拷贝到统一目录，文件名带模型名以便集中浏览：
      <backup_root>/<model_name>.safetensors
      <backup_root>/<model_name>.info.json
    """
    extract_path = os.path.join(model_dir, EXTRACTED_WEIGHTS_FILENAME)
    info_path = os.path.join(model_dir, EXTRACTED_INFO_FILENAME)
    if not (os.path.exists(extract_path) and os.path.exists(info_path)):
        print(f"❌ [备份] {model_name} 提取产物不完整，跳过。")
        return False

    try:
        os.makedirs(backup_root, exist_ok=True)
        weight_dst = os.path.join(backup_root, f"{model_name}.safetensors")
        info_dst = os.path.join(backup_root, f"{model_name}.info.json")
        shutil.copy2(extract_path, weight_dst)
        shutil.copy2(info_path, info_dst)
        print(f"📦 [备份] {model_name} -> {weight_dst}")
        return True
    except Exception as e:
        print(f"❌ [备份] {model_name} 复制失败: {e}")
        return False


def cleanup_original_weights(model_dir: str, model_name: str) -> int:
    """
    删除模型目录下的原始下载权重以节省空间，但保留：
      - 我们刚提取的 EXTRACTED_WEIGHTS_FILENAME
      - tokenizer / config / 索引等小型 JSON / 文本文件
    返回释放的字节数。
    """
    if not os.path.isdir(model_dir):
        return 0

    freed = 0
    for root, _, files in os.walk(model_dir):
        for name in files:
            if name == EXTRACTED_WEIGHTS_FILENAME:
                continue
            if not name.endswith(WEIGHT_FILE_SUFFIXES):
                continue
            path = os.path.join(root, name)
            try:
                size = os.path.getsize(path)
                os.remove(path)
                freed += size
            except Exception as e:
                print(f"⚠️ [清理] 删除 {path} 失败: {e}")

    if freed:
        print(f"🧹 [清理] {model_name} 释放原始权重 {freed/1024/1024:.2f} MiB")
    else:
        print(f"ℹ️  [清理] {model_name} 没有可删除的原始权重文件。")
    return freed

def download_emb_only_with_retry(
    model_name: str,
    repo_id: str,
    cache_dir: str,
    max_retries: int,
    *,
    backup_root: str | None = None,
    verify: bool = True,
    cleanup_originals: bool = False,
) -> tuple[bool, str]:
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
                
                # 用与提取阶段完全一致的精确匹配，避免命中视觉塔的
                # patch_embed / pos_embed、MTP 模块的 norm_embedding、
                # MoE 共享专家的 shared_head.norm 等无关张量，
                # 进而避免下载多余的权重分片。
                for param_name, shard_file in weight_map.items():
                    if is_target_tensor_key(param_name):
                        shard_set.add(shard_file)
                
                print(f"🎯 成功定位到首尾特征矩阵，分布在分块: {list(shard_set)}")
                target_files.extend(list(shard_set))
            else:
                print(f"ℹ️  {model_name} 可能是小模型或本身未分块，将下载基础 safetensors。")
                target_files.append('*.safetensors')
                target_files.append('*.bin')
                target_files.append('*.pth')

            # 第二步：精准下载对应的巨大的权重文件
            # ignore_patterns 屏蔽 Meta 原生格式的 original/ 子目录
            # （Llama 系列里的 consolidated.*.pth 等，与 HF 格式重复且很大）
            print(f"⬇️ 开始只下载指定的 Embedding/Head 文件: {target_files}")
            final_dir = snapshot_download(
                model_id=repo_id, 
                cache_dir=cache_dir,
                allow_patterns=target_files,
                ignore_patterns=['original/*'],
            )
            
            # 第三步：下载完成后直接提取文件中的 emb/unemb
            extract_ok = extract_and_save_emb_matrices(final_dir, model_name)
            if not extract_ok:
                return False, f"❌ 提取失败: {model_name} -> {final_dir}"

            verified = True
            if verify:
                verified = verify_extracted_safetensors(final_dir, model_name)
                if not verified:
                    return False, (
                        f"❌ 校验失败: {model_name} -> {final_dir}（已保留原始权重不清理）"
                    )

            backed_up = False
            if backup_root and verified:
                backed_up = backup_extracted_files(final_dir, model_name, backup_root)

            if cleanup_originals:
                if not verified:
                    print(f"⏭ [清理] {model_name} 未通过校验，跳过删除原始权重。")
                elif backup_root and not backed_up:
                    print(f"⏭ [清理] {model_name} 备份未成功，跳过删除原始权重。")
                else:
                    cleanup_original_weights(final_dir, model_name)

            return True, f"✅ 成功完成全流程: {model_name} -> {final_dir}"
            
        except Exception as e:
            if attempt < max_retries:
                wait_time = 10 * attempt 
                print(f"⚠️ 异常: {model_name} (第 {attempt}/{max_retries} 次尝试)。等待 {wait_time}s... 原因: {e}")
                time.sleep(wait_time)
            else:
                return False, f"❌ 彻底失败: {model_name}。最终错误: {e}"

def main():
    script_dir = str(repository_root(__file__))
    yaml_path = os.path.join(script_dir, "configs", "models.yaml")
    if not os.path.exists(yaml_path):
        print(f"❌ 找不到配置文件: {yaml_path}")
        return

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
        
    config_block = config_data.get("config", {}) or {}
    base_cache_dir = config_block.get("cache_dir", "./downloaded_models")
    if not os.path.isabs(base_cache_dir):
        base_cache_dir = os.path.normpath(os.path.join(script_dir, base_cache_dir))
    max_retries = config_block.get("max_retries", 3)
    max_workers = config_block.get("max_workers", 3)
    repo_mappings = config_data.get("model_repo_ids", {})

    extracts_dir_cfg = config_block.get("extracts_dir", "./extracts")
    extracts_dir = (
        extracts_dir_cfg
        if os.path.isabs(extracts_dir_cfg)
        else os.path.normpath(os.path.join(script_dir, extracts_dir_cfg))
    )
    verify_extracts = bool(config_block.get("verify_extracts", True))
    backup_extracts = bool(config_block.get("backup_extracts", True))
    cleanup_originals = bool(config_block.get("cleanup_originals", False))
    backup_root = extracts_dir if backup_extracts else None

    print(
        f"🚀 开始批量 [下载并提取] 模型 (缓存于: {base_cache_dir}, 并发线程数: {max_workers})"
        f" | 校验={verify_extracts} 备份={backup_extracts} 清理原始权重={cleanup_originals}"
    )
    if backup_root:
        print(f"📦 备份目录: {backup_root}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {
            executor.submit(
                download_emb_only_with_retry,
                model_name,
                repo_mappings[model_name],
                base_cache_dir,
                max_retries,
                backup_root=backup_root,
                verify=verify_extracts,
                cleanup_originals=cleanup_originals,
            ): model_name for model_name in repo_mappings
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
                        "standardized_sources": data.get("standardized_sources", {}),
                        "raw_embed_keys": data.get("raw_embed_keys", []),
                        "raw_lm_head_keys": data.get("raw_lm_head_keys", []),
                        "raw_tensors_dimensions": data.get("raw_tensors_dimensions", data.get("tensors_dimensions"))
                    }
            except Exception as e:
                print(f"⚠️ 读取 {info_path} 失败: {e}")

    summary_dir = os.path.join(script_dir, "cross_model_geometry", "audits")
    summary_path = os.path.join(summary_dir, "all_models_summary.json")
    try:
        os.makedirs(summary_dir, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
        print(f"✅ 全局汇总文件已生成: {summary_path}")
    except Exception as e:
        print(f"❌ 生成全局汇总文件失败: {e}")

if __name__ == "__main__":
    main()