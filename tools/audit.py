"""统一审计入口：下载完整性、矩阵大小、tied/untied 与冗余缓存检查。

用法：
    python3 tools/audit.py downloads  # 下载/抽取完整性与大小核对
    python3 tools/audit.py tied       # tied/untied 与冗余文件检查
    python3 tools/audit.py all        # 依次执行全部审计
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from tools.paths import repository_root  # noqa: E402

ROOT = repository_root(__file__)
YAML = ROOT / "configs" / "models.yaml"
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

def audit_downloads():
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


WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth")
EMB_HEAD_KEYWORDS = ("embed_tokens", "word_embeddings", "lm_head", "wte")


def fmt(n: float) -> str:
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} PiB"


def is_emb_or_head_key(key: str) -> bool:
    k = key.lower()
    return any(t in k for t in EMB_HEAD_KEYWORDS) or k in {
        "head.weight", "output.weight"
    }


def safetensors_keys(path: Path) -> List[str]:
    """只读 safetensors header，不加载张量。"""
    try:
        with open(path, "rb") as f:
            header_size = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(header_size).decode("utf-8"))
        return [k for k in header.keys() if k != "__metadata__"]
    except Exception:
        return []


def directory_size(path: Path) -> int:
    """物理大小（不跟随软链）。"""
    total = 0
    for root, _, files in os.walk(path, followlinks=False):
        for fn in files:
            p = Path(root) / fn
            try:
                total += p.lstat().st_size
            except OSError:
                pass
    return total


def find_model_dir(repo_id: str) -> Optional[Path]:
    owner, repo = repo_id.split("/", 1)
    for cand in (CACHE / owner / repo, CACHE / owner / repo.replace(".", "___")):
        if cand.exists() and not cand.is_symlink():
            return cand
        if cand.exists() and cand.is_symlink():
            target = cand.resolve()
            if target.exists():
                return target
    return None


def check_tied(model_name: str, repo_id: str) -> dict:
    model_dir = find_model_dir(repo_id)
    if model_dir is None:
        return {"name": model_name, "status": "missing"}

    cfg_path = model_dir / "config.json"
    info_path = model_dir / "extracted_embeddings_info.json"
    out_path = model_dir / "extracted_embeddings.safetensors"

    cfg_tied = None
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg_tied = cfg.get("tie_word_embeddings")
        except Exception:
            pass

    if not info_path.exists():
        return {"name": model_name, "status": "no_info", "cfg_tied": cfg_tied}

    info = json.loads(info_path.read_text(encoding="utf-8"))
    info_tied = info.get("tie_word_embeddings")
    raw_head_keys = info.get("raw_lm_head_keys", [])
    raw_embed_keys = info.get("raw_embed_keys", [])

    # 物理证据：extracted 文件里有没有独立的 lm_head 张量？
    extracted_keys = safetensors_keys(out_path) if out_path.exists() else []
    has_phys_head = any("lm_head" in k.lower() or k.lower() in {"head.weight", "output.weight"} for k in extracted_keys)
    has_phys_embed = any("embed_tokens" in k.lower() or "word_embeddings" in k.lower() or "wte" in k.lower() for k in extracted_keys)

    # info.json 自动推断逻辑：是否与 config 真值一致
    inference = "from_config" if cfg_tied is not None else "inferred"

    # 综合判定可信状态
    issues = []
    # 物理上 head 存在却被标 tied，意味着实际是 untied，info.json 错了
    if info_tied and has_phys_head:
        issues.append("⚠ info=tied 但物理上有 lm_head 张量")
    # 物理上没有 head 却被标 untied
    if (info_tied is False) and (not has_phys_head):
        issues.append("⚠ info=untied 但物理上没有 lm_head 张量")
    # info 和 config 不一致
    if cfg_tied is not None and info_tied is not None and cfg_tied != info_tied:
        issues.append(f"⚠ info({info_tied}) ≠ config({cfg_tied})")

    return {
        "name": model_name,
        "dir": str(model_dir),
        "status": "ok",
        "cfg_tied": cfg_tied,
        "info_tied": info_tied,
        "inference": inference,
        "raw_embed_keys": raw_embed_keys,
        "raw_head_keys": raw_head_keys,
        "phys_embed": has_phys_embed,
        "phys_head": has_phys_head,
        "issues": issues,
        "extracted_keys": extracted_keys,
    }


def check_redundancy(model_name: str, repo_id: str) -> dict:
    model_dir = find_model_dir(repo_id)
    if model_dir is None:
        return {"name": model_name, "status": "missing"}

    # 1) 看 index 文件，理论上 *应该* 下载哪些 shard
    expected_shards: set = set()
    index_paths = [
        model_dir / "model.safetensors.index.json",
        model_dir / "pytorch_model.bin.index.json",
    ]
    for ip in index_paths:
        if ip.exists():
            try:
                idx = json.loads(ip.read_text(encoding="utf-8"))
                wmap = idx.get("weight_map", {})
                for param, shard in wmap.items():
                    if is_emb_or_head_key(param):
                        expected_shards.add(shard)
            except Exception:
                pass
            break

    # 2) 列出实际下载的所有权重文件
    actual_files: List[Path] = []
    for root, _, files in os.walk(model_dir, followlinks=False):
        for fn in files:
            if fn.endswith(WEIGHT_EXTS) and fn != "extracted_embeddings.safetensors":
                actual_files.append(Path(root) / fn)

    # 3) 判定每个实际下载文件是否真的含 emb/head
    redundant_files: List[Tuple[Path, int]] = []
    useful_files: List[Tuple[Path, int]] = []
    for fp in actual_files:
        keys = safetensors_keys(fp) if fp.suffix == ".safetensors" else []
        contains_target = any(is_emb_or_head_key(k) for k in keys)
        size = fp.stat().st_size
        # 如果 index 显示这个 shard 应该不含 emb/head，或者 header 也确实没有，标冗余
        if expected_shards and fp.name not in expected_shards and not contains_target:
            redundant_files.append((fp, size))
        elif (not expected_shards) and (not contains_target) and fp.suffix == ".safetensors":
            redundant_files.append((fp, size))
        else:
            useful_files.append((fp, size))

    # 4) 多余的元目录
    extra_dirs = []
    for sub in ("original", ".cache"):
        p = model_dir / sub
        if p.exists():
            extra_dirs.append((p, directory_size(p)))

    return {
        "name": model_name,
        "dir": str(model_dir),
        "status": "ok",
        "expected_shards": sorted(expected_shards),
        "actual_weight_files": [p.name for p in actual_files],
        "useful_files": [(p.name, s) for p, s in useful_files],
        "redundant_files": [(str(p), s) for p, s in redundant_files],
        "extra_dirs": [(str(p), s) for p, s in extra_dirs],
    }


def audit_tied_redundancy():
    cfg = yaml.safe_load(YAML.read_text(encoding="utf-8"))
    repo_ids: Dict[str, str] = cfg["model_repo_ids"]

    # ============= 第一部分：tied/untied 核对 =============
    tied_rows = []
    for name, repo in sorted(repo_ids.items()):
        tied_rows.append(check_tied(name, repo))

    print("=" * 92)
    print("【1】Tied / Untied 判断核对")
    print("=" * 92)
    print(f"{'模型':32s} {'cfg':>6s} {'info':>6s} {'inf?':>6s} {'phEmb':>6s} {'phHead':>6s}  问题")
    bad_tie = []
    for r in tied_rows:
        if r.get("status") != "ok":
            print(f"{r['name']:32s}   状态: {r['status']}")
            continue
        cfg_v = "—" if r["cfg_tied"] is None else ("T" if r["cfg_tied"] else "F")
        info_v = "T" if r["info_tied"] else "F"
        inf = "auto" if r["inference"] == "inferred" else "cfg"
        phE = "Y" if r["phys_embed"] else "N"
        phH = "Y" if r["phys_head"] else "N"
        issues = "; ".join(r["issues"]) if r["issues"] else "✅"
        if r["issues"]:
            bad_tie.append(r)
        print(f"{r['name']:32s} {cfg_v:>6s} {info_v:>6s} {inf:>6s} {phE:>6s} {phH:>6s}  {issues}")

    if bad_tie:
        print("\n>>> tied/untied 异常清单：")
        for r in bad_tie:
            print(f"  - {r['name']:32s} cfg={r['cfg_tied']} info={r['info_tied']} "
                  f"raw_embed={r['raw_embed_keys']} raw_head={r['raw_head_keys']} "
                  f"phys_head={r['phys_head']} | {r['issues']}")
    else:
        print("\n>>> 所有模型 tied/untied 判断与物理存储一致 ✅")

    # ============= 第二部分：冗余检查 =============
    print("\n" + "=" * 92)
    print("【2】冗余文件检查（不含 emb/head 的多余 shard / 多余目录）")
    print("=" * 92)
    red_rows = [check_redundancy(n, r) for n, r in sorted(repo_ids.items())]

    total_redundant = 0
    n_with_red = 0
    for r in red_rows:
        if r.get("status") != "ok":
            continue
        red = r["redundant_files"]
        extra = r["extra_dirs"]
        size_red = sum(s for _, s in red)
        size_extra = sum(s for _, s in extra)
        if red or extra:
            n_with_red += 1
            total_redundant += size_red + size_extra
            print(f"\n  ❗ {r['name']}  (理论 emb/head shards: {r['expected_shards']})")
            print(f"     实际下载: {r['actual_weight_files']}")
            for p, s in red:
                print(f"     ↳ 冗余权重 {fmt(s):>10s}  {p}")
            for p, s in extra:
                print(f"     ↳ 多余目录 {fmt(s):>10s}  {p}")

    if n_with_red == 0:
        print("\n>>> 所有模型都没有多余下载，干干净净 ✅")
    else:
        print(f"\n>>> 共 {n_with_red} 个模型有多余文件/目录，合计可释放: {fmt(total_redundant)}")

    # ============= 第三部分：软链/重复目录 =============
    print("\n" + "=" * 92)
    print("【3】软链重复目录（owner/Name vs owner/Name___）")
    print("=" * 92)
    duplicate_pairs = []
    for owner_dir in CACHE.iterdir():
        if not owner_dir.is_dir() or owner_dir.name.startswith("."):
            continue
        names = list(owner_dir.iterdir())
        for n in names:
            if "___" not in n.name:
                continue
            sibling_name = n.name.replace("___", ".")
            sibling = owner_dir / sibling_name
            if sibling.exists():
                duplicate_pairs.append((sibling, n))
    print(f"  发现 {len(duplicate_pairs)} 组成对的同义目录（"
          f"\".\" 名 vs \"___\" 名）")
    real_dup_bytes = 0
    for a, b in duplicate_pairs:
        a_link = a.is_symlink()
        b_link = b.is_symlink()
        a_size = directory_size(a)
        b_size = directory_size(b)
        # 软链 lstat 大小很小，本身不占空间；如果两个都不是软链且都很大，那才是真复制
        is_real_dup = (not a_link) and (not b_link)
        marker = "🔁 真复制" if is_real_dup else ("软链 →" if a_link or b_link else "?")
        if is_real_dup:
            real_dup_bytes += min(a_size, b_size)
        print(f"  {marker:10s} {a.name:34s} ({fmt(a_size)})  +  {b.name:34s} ({fmt(b_size)})")
    if real_dup_bytes:
        print(f"\n>>> 真复制空间: {fmt(real_dup_bytes)} 可释放")
    else:
        print("\n>>> 没有真复制，全部是软链，无浪费 ✅")

    # ============= 第四部分：临时/锁目录 =============
    print("\n" + "=" * 92)
    print("【4】顶层 .lock / 临时目录")
    print("=" * 92)
    for sub in CACHE.iterdir():
        if sub.name.startswith(".") or "temp" in sub.name.lower() or sub.name.startswith("_"):
            sz = directory_size(sub)
            print(f"  {sub.name:20s}  {fmt(sz):>10s}  {sub}")


def main() -> None:
    parser = argparse.ArgumentParser(description="get_useful cache/extract audit tools")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("downloads", "tied", "all"),
        default="all",
        help="审计类型：downloads=下载/抽取完整性；tied=tied/untied 与冗余；all=全部",
    )
    args = parser.parse_args()

    if args.command in ("downloads", "all"):
        audit_downloads()
    if args.command == "all":
        print("\n" + "#" * 92 + "\n")
    if args.command in ("tied", "all"):
        audit_tied_redundancy()


if __name__ == "__main__":
    main()
