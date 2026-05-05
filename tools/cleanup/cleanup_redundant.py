"""
删除冗余文件。删除前对每个 .safetensors 重新读取 header 确认：
  - 不含任何严格意义的 emb/head/wte/word_embeddings 张量
对 .pth（Meta 原生 consolidated 权重）直接确认其在 original/ 子目录后删除。

用法：
    python3 cleanup_redundant.py            # dry-run，仅列出
    python3 cleanup_redundant.py --apply    # 实际删除
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from tools.paths import repository_root  # noqa: E402

CACHE = repository_root(__file__) / "downloaded_models"

# 冗余的 .safetensors 分片（来自审计结果）
REDUNDANT_SHARDS = [
    "ZhipuAI/GLM-5/model-00274-of-00282.safetensors",
    "ZhipuAI/GLM-5___1/model-00274-of-00282.safetensors",
    "qwen/Qwen3___5-9B/model.safetensors-00004-of-00004.safetensors",
    "qwen/Qwen3___5-9B-Base/model.safetensors-00004-of-00004.safetensors",
    "qwen/Qwen3___5-4B/model.safetensors-00002-of-00002.safetensors",
    "qwen/Qwen3___5-4B-Base/model.safetensors-00002-of-00002.safetensors",
    "qwen/Qwen3___5-27B/model.safetensors-00011-of-00011.safetensors",
    "qwen/Qwen3___5-35B-A3B/model.safetensors-00014-of-00014.safetensors",
    "qwen/Qwen3___5-35B-A3B-Base/model.safetensors-00014-of-00014.safetensors",
    "qwen/Qwen3___5-122B-A10B/model.safetensors-00039-of-00039.safetensors",
    "qwen/Qwen3___5-397B-A17B/model.safetensors-00094-of-00094.safetensors",
    "qwen/Qwen3___6-35B-A3B/model-00002-of-00026.safetensors",
]

# Meta 原生的 consolidated 权重（与 HF 格式重复，但很大）
REDUNDANT_PTH = [
    "LLM-Research/Llama-3___2-1B/original/consolidated.00.pth",
    "LLM-Research/Llama-3___2-1B-Instruct/original/consolidated.00.pth",
]

STRICT_KEYWORDS = ("embed_tokens", "word_embeddings", "lm_head",
                   "transformer.wte", "head.weight", "output.weight")


def fmt(n: float) -> str:
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} PiB"


def is_strict_emb_or_head(key: str) -> bool:
    k = key.lower()
    return any(t in k for t in STRICT_KEYWORDS)


def safetensors_header_keys(path: Path) -> list[str]:
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        h = json.loads(f.read(n).decode("utf-8"))
    return [k for k in h if k != "__metadata__"]


def verify_safe_to_delete_safetensors(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "文件不存在"
    try:
        keys = safetensors_header_keys(path)
    except Exception as e:
        return False, f"读取 header 失败: {e}"
    hits = [k for k in keys if is_strict_emb_or_head(k)]
    if hits:
        return False, f"⚠ 含严格 emb/head 张量: {hits[:3]}"
    return True, f"OK ({len(keys)} 个张量, 无严格 emb/head)"


def verify_safe_to_delete_pth(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "文件不存在"
    if "original" not in path.parts:
        return False, "⚠ 不在 original/ 子目录，拒绝删除"
    sibling = path.parent.parent / "model.safetensors"
    sib_idx = path.parent.parent / "model.safetensors.index.json"
    if not sibling.exists() and not sib_idx.exists():
        return False, "⚠ 同级目录没有 HF safetensors，可能并非冗余"
    return True, f"OK (HF 格式同时存在)"


def main(apply: bool):
    print(f"🔎 模式: {'APPLY 删除' if apply else 'DRY-RUN 只列出'}")
    print(f"📂 缓存根目录: {CACHE}\n")

    total_freed = 0
    deleted = 0
    skipped = 0

    print("=" * 90)
    print("【A】冗余 safetensors 分片")
    print("=" * 90)
    for rel in REDUNDANT_SHARDS:
        p = CACHE / rel
        ok, reason = verify_safe_to_delete_safetensors(p)
        size = p.stat().st_size if p.exists() else 0
        marker = "✅" if ok else "⏭"
        print(f"  {marker} {fmt(size):>10s}  {rel}")
        print(f"     └ {reason}")
        if ok:
            if apply:
                try:
                    p.unlink()
                    deleted += 1
                    total_freed += size
                except Exception as e:
                    print(f"     ❌ 删除失败: {e}")
            else:
                total_freed += size
                deleted += 1
        else:
            skipped += 1

    print("\n" + "=" * 90)
    print("【B】Llama 原生 consolidated 权重")
    print("=" * 90)
    for rel in REDUNDANT_PTH:
        p = CACHE / rel
        ok, reason = verify_safe_to_delete_pth(p)
        size = p.stat().st_size if p.exists() else 0
        marker = "✅" if ok else "⏭"
        print(f"  {marker} {fmt(size):>10s}  {rel}")
        print(f"     └ {reason}")
        if ok:
            if apply:
                try:
                    p.unlink()
                    deleted += 1
                    total_freed += size
                except Exception as e:
                    print(f"     ❌ 删除失败: {e}")
            else:
                total_freed += size
                deleted += 1
        else:
            skipped += 1

    print("\n" + "=" * 90)
    if apply:
        print(f"🗑 实际删除文件数: {deleted}, 释放空间: {fmt(total_freed)}, 跳过: {skipped}")
    else:
        print(f"📋 计划删除文件数: {deleted}, 预计释放: {fmt(total_freed)}, 跳过: {skipped}")
        print("   再加 --apply 才会真正删除")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
