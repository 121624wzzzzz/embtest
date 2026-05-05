"""
审计脚本：
  1) 比对 models.yaml 中声明的所有模型 vs downloaded_models/ 中实际下载的目录，
     列出 [缺失 / 已下但未提取 / 已完整提取] 三类。
  2) 对每个已提取模型，根据 config.json 推断权重 dtype，结合
     standardized_dims (vocab x hidden) 和 tie_word_embeddings 计算
     emb/unemb 理论占用，并与 extracted_embeddings.safetensors 实际文件大小对比，
     标出异常。
"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from typing import Tuple, Dict

import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from tools.paths import repository_root  # noqa: E402

ROOT = repository_root(__file__)
YAML = ROOT / "models.yaml"
CACHE = ROOT / "downloaded_models"

DTYPE_BYTES = {
    "float32": 4, "fp32": 4, "torch.float32": 4,
    "float16": 2, "fp16": 2, "half": 2, "torch.float16": 2,
    "bfloat16": 2, "bf16": 2, "torch.bfloat16": 2,
    "int8": 1, "fp8": 1, "float8_e4m3fn": 1, "float8_e5m2": 1,
}

def detect_dtype_bytes(model_dir: Path) -> Tuple[int, str]:
    """
    返回 emb/head 张量的字节数。
    重要：DeepSeek-V3 / MiniMax 等 fp8 量化模型，主干虽然是 fp8，但
    embedding 和 lm_head 通常仍然以 bf16 存储，所以这里不看 quantization_config，
    只用 torch_dtype 来决定，缺省按 bf16(2B) 计算。
    """
    cfg = model_dir / "config.json"
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        td = (data.get("torch_dtype") or "").lower().strip()
        if td in DTYPE_BYTES:
            return DTYPE_BYTES[td], td
    return 2, "assumed-bf16"

def fmt(n: int) -> str:
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} PiB"

def main():
    cfg = yaml.safe_load(YAML.read_text(encoding="utf-8"))
    repo_ids: Dict[str, str] = cfg["model_repo_ids"]

    rows = []
    for model_name, repo_id in sorted(repo_ids.items()):
        owner, repo = repo_id.split("/", 1)
        # 同时尝试原名和把 . 替换为 ___ 的落盘名
        candidates = [CACHE / owner / repo, CACHE / owner / repo.replace(".", "___")]
        model_dir = next((p for p in candidates if p.exists()), None)

        if model_dir is None:
            rows.append({"name": model_name, "repo": repo_id, "status": "❌ 未下载",
                         "dir": "-", "info": None})
            continue

        info_path = model_dir / "extracted_embeddings_info.json"
        out_path = model_dir / "extracted_embeddings.safetensors"
        if not info_path.exists() or not out_path.exists():
            rows.append({"name": model_name, "repo": repo_id,
                         "status": "⚠️ 已下但未提取", "dir": str(model_dir),
                         "info": None})
            continue

        info = json.loads(info_path.read_text(encoding="utf-8"))
        dims = info.get("standardized_dims", {})
        emb = dims.get("embed")
        tied = bool(info.get("tie_word_embeddings"))
        sources = info.get("standardized_sources", {})
        has_phys_head = sources.get("lm_head") and sources["lm_head"] != sources.get("embed")

        # 关键修正：用 raw_tensors_dimensions 里所有张量的元素数总和算理论值
        raw = info.get("raw_tensors_dimensions", {}) or {}
        # 物理上真正写进 extracted_embeddings.safetensors 的就是 raw 里的全部张量
        total_elems = 0
        for shape in raw.values():
            n = 1
            for d in shape:
                n *= d
            total_elems += n
        n_matrices = len(raw)

        bytes_per, dtype_label = detect_dtype_bytes(model_dir)
        expected = total_elems * bytes_per
        actual = out_path.stat().st_size
        diff = actual - expected
        ratio = actual / expected if expected else 0.0
        status = "✅ 正常"
        if expected == 0:
            status = "❓ 维度缺失"
        elif ratio < 0.98 or ratio > 1.05:
            status = f"❗ 异常 ({ratio:.2f}x)"

        rows.append({
            "name": model_name, "repo": repo_id, "status": status,
            "dir": str(model_dir),
            "tied": tied, "phys_head": has_phys_head,
            "emb": emb, "dtype": dtype_label,
            "n_mat": n_matrices, "expected": expected, "actual": actual,
            "diff": diff,
        })

    # ---- 输出 ----
    miss = [r for r in rows if r["status"].startswith("❌")]
    half = [r for r in rows if r["status"].startswith("⚠️")]
    abn  = [r for r in rows if r["status"].startswith("❗") or r["status"].startswith("❓")]
    ok   = [r for r in rows if r["status"].startswith("✅")]

    print(f"📊 共 {len(rows)} 个模型 | ✅ 正常 {len(ok)} | ❗ 异常 {len(abn)} | ⚠️ 未提取 {len(half)} | ❌ 未下载 {len(miss)}")
    print()

    if miss:
        print("=== ❌ 未下载（yaml 中声明但本地无目录） ===")
        for r in miss:
            print(f"  - {r['name']:32s} -> {r['repo']}")
        print()
    if half:
        print("=== ⚠️ 已下但未提取（缺 extracted_embeddings.* 文件） ===")
        for r in half:
            print(f"  - {r['name']:32s} -> {r['dir']}")
        print()

    print("=== 占用核对 ===")
    print(f"{'模型':32s} {'tied':5s} {'phys_head':9s} {'emb形状':>20s} {'dtype':>14s} {'矩阵数':>5s} {'理论':>10s} {'实际':>10s} {'比值':>6s} {'状态'}")
    for r in sorted(rows, key=lambda x: x["name"]):
        if r["status"].startswith("❌") or r["status"].startswith("⚠️"):
            continue
        emb_s = f"{r['emb'][0]}x{r['emb'][1]}" if r["emb"] else "-"
        ratio = r["actual"] / r["expected"] if r["expected"] else 0
        print(f"{r['name']:32s} {str(r['tied']):5s} {str(r['phys_head']):9s} {emb_s:>20s} {r['dtype']:>14s} {r['n_mat']:>5d} {fmt(r['expected']):>10s} {fmt(r['actual']):>10s} {ratio:>6.2f} {r['status']}")

if __name__ == "__main__":
    main()
