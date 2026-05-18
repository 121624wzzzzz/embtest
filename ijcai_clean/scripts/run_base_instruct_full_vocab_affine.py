#!/usr/bin/env python3
"""Full-vocabulary Base-Instruct affine, A, and low-rank diagnostics.

Unlike Task5's default row subsampling, this runner uses every token id row for
Base-Instruct pairs whose vocabularies match exactly. It writes one consolidated
CSV/Markdown report with fit quality, compact diagnostics of the fitted affine
A matrix, and SVD/low-rank energy diagnostics for both E_instruct - E_base and
A - I.
"""
from __future__ import annotations

import csv
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

from _cli import bootstrap_repo

_REPO_ROOT = bootstrap_repo(__file__)

from ijcai_clean.data import actual_tied, load_E_U_matrices, load_info_json  # noqa: E402


BATCH_ROWS = 8192
SOURCE_TASK = "task1_base_instruct"
ENERGY_THRESHOLDS = (0.5, 0.8, 0.9, 0.95, 0.99)
ENERGY_AT_K = (1, 5, 10, 20, 50, 100, 200, 500, 1000)
RELATIVE_ENERGY_FRACTIONS = (0.01, 0.05, 0.10)


def _as_tensor(arr: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(
        device=device, dtype=dtype
    )


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else float("nan")


def _fmt(value: Any) -> str:
    return f"{float(value):.4f}"


def _rank_for(cumulative: torch.Tensor, threshold: float) -> int:
    idx = torch.searchsorted(cumulative, torch.tensor(threshold, device=cumulative.device))
    return int(idx.item()) + 1


def svd_energy_from_gram(
    gram: torch.Tensor, *, prefix: str, top_values: int = 10
) -> Dict[str, Any]:
    """Return singular-value energy diagnostics from M.T @ M."""
    eigvals = torch.linalg.eigvalsh(gram)
    eigvals = torch.clamp(eigvals, min=0.0).flip(0)
    total = float(eigvals.sum().item())
    out: Dict[str, Any] = {
        f"{prefix}_total_energy": total,
        f"{prefix}_fro_norm": math.sqrt(total),
        f"{prefix}_numerical_rank": int((eigvals > (eigvals.max() * 1e-6)).sum().item())
        if eigvals.numel()
        else 0,
    }
    if total <= 0:
        for threshold in ENERGY_THRESHOLDS:
            out[f"{prefix}_rank_{int(threshold * 100)}"] = 0
        for k in ENERGY_AT_K:
            out[f"{prefix}_energy_at_{k}"] = 0.0
        for fraction in RELATIVE_ENERGY_FRACTIONS:
            out[f"{prefix}_energy_at_{int(fraction * 100)}pct_h"] = 0.0
        out[f"{prefix}_effective_rank"] = 0.0
        out[f"{prefix}_top_singular_values"] = ""
        return out

    energy = eigvals / total
    cumulative = torch.cumsum(energy, dim=0)
    for threshold in ENERGY_THRESHOLDS:
        out[f"{prefix}_rank_{int(threshold * 100)}"] = _rank_for(cumulative, threshold)
    for k in ENERGY_AT_K:
        kk = min(k, int(cumulative.numel()))
        out[f"{prefix}_energy_at_{k}"] = float(cumulative[kk - 1].item()) if kk else 0.0
    for fraction in RELATIVE_ENERGY_FRACTIONS:
        kk = min(math.ceil(float(cumulative.numel()) * fraction), int(cumulative.numel()))
        out[f"{prefix}_energy_at_{int(fraction * 100)}pct_h"] = (
            float(cumulative[kk - 1].item()) if kk else 0.0
        )

    nonzero = energy[energy > 0]
    entropy = -torch.sum(nonzero * torch.log(nonzero)).item()
    out[f"{prefix}_effective_rank"] = float(math.exp(entropy))
    singular = torch.sqrt(eigvals[:top_values]).detach().cpu().numpy()
    out[f"{prefix}_top_singular_values"] = ";".join(f"{x:.6g}" for x in singular)
    return out


def gram_of_row_delta(
    X_np: np.ndarray,
    Y_np: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_rows: int = BATCH_ROWS,
) -> torch.Tensor:
    """Compute (Y-X).T @ (Y-X) by streaming rows."""
    if X_np.shape != Y_np.shape:
        raise ValueError(f"shape mismatch: {X_np.shape} vs {Y_np.shape}")
    _, d = X_np.shape
    gram = torch.zeros((d, d), device=device, dtype=dtype)
    for start in range(0, X_np.shape[0], batch_rows):
        end = min(X_np.shape[0], start + batch_rows)
        delta = _as_tensor(Y_np[start:end], device, dtype) - _as_tensor(
            X_np[start:end], device, dtype
        )
        gram.add_(delta.T @ delta)
        del delta
    return gram


def _a_diagnostics(A: torch.Tensor) -> Dict[str, float]:
    d = int(A.shape[0])
    eye = torch.eye(d, device=A.device, dtype=A.dtype)
    diag = torch.diagonal(A)
    off = A - torch.diag(diag)
    norm_a = torch.linalg.matrix_norm(A, ord="fro")
    norm_i = float(d) ** 0.5
    diff_i = torch.linalg.matrix_norm(A - eye, ord="fro")
    off_norm = torch.linalg.matrix_norm(off, ord="fro")
    trace = torch.trace(A)
    # A rotation/reflection would have A.T @ A close to I.
    orth_diff = torch.linalg.matrix_norm(A.T @ A - eye, ord="fro")
    return {
        "trace_A": float(trace.item()),
        "trace_A_over_d": float((trace / d).item()),
        "norm_A": float(norm_a.item()),
        "norm_A_minus_I": float(diff_i.item()),
        "rel_A_minus_I_over_I": float(diff_i.item() / (norm_i + 1e-20)),
        "rel_A_minus_I_over_A": float(diff_i.item() / (float(norm_a.item()) + 1e-20)),
        "identity_cosine": float((trace / ((norm_a * norm_i) + 1e-20)).item()),
        "diag_mean": float(diag.mean().item()),
        "diag_std": float(diag.std(unbiased=False).item()),
        "diag_min": float(diag.min().item()),
        "diag_max": float(diag.max().item()),
        "offdiag_norm": float(off_norm.item()),
        "offdiag_norm_over_A": float(off_norm.item() / (float(norm_a.item()) + 1e-20)),
        "orthogonality_error": float(orth_diff.item()),
        "rel_orthogonality_error_over_I": float(orth_diff.item() / (norm_i + 1e-20)),
    }


def full_affine_stream(
    X_np: np.ndarray,
    Y_np: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_rows: int = BATCH_ROWS,
) -> Dict[str, Any]:
    """Fit Y ~= X A + b on all rows using centered normal equations."""
    n, dx = X_np.shape
    ny, dy = Y_np.shape
    if n != ny or dx != dy:
        raise ValueError(f"shape mismatch: {X_np.shape} vs {Y_np.shape}")

    sum_x = np.zeros(dx, dtype=np.float64)
    sum_y = np.zeros(dy, dtype=np.float64)
    y2_sum = 0.0
    for start in range(0, n, batch_rows):
        end = min(n, start + batch_rows)
        xb = X_np[start:end].astype(np.float64, copy=False)
        yb = Y_np[start:end].astype(np.float64, copy=False)
        sum_x += xb.sum(axis=0)
        sum_y += yb.sum(axis=0)
        y2_sum += float((yb * yb).sum())

    mean_x64 = sum_x / n
    mean_y64 = sum_y / n
    mx_np = mean_x64.astype(np.float32)
    my_np = mean_y64.astype(np.float32)
    ss_tot = float(y2_sum - n * float(mean_y64 @ mean_y64))

    G = torch.zeros((dx, dx), device=device, dtype=dtype)
    C = torch.zeros((dx, dy), device=device, dtype=dtype)
    mx = torch.from_numpy(mx_np).to(device=device, dtype=dtype)
    my = torch.from_numpy(my_np).to(device=device, dtype=dtype)

    for start in range(0, n, batch_rows):
        end = min(n, start + batch_rows)
        x = _as_tensor(X_np[start:end], device, dtype) - mx
        y = _as_tensor(Y_np[start:end], device, dtype) - my
        G.add_(x.T @ x)
        C.add_(x.T @ y)
        del x, y

    try:
        A = torch.linalg.solve(G, C)
        solver = "centered normal equations solve"
    except Exception:
        A = torch.linalg.lstsq(G, C, rcond=None).solution
        solver = "centered normal equations lstsq fallback"
    b = my - mx @ A

    ss_res = 0.0
    y_norm_sq = 0.0
    for start in range(0, n, batch_rows):
        end = min(n, start + batch_rows)
        x = _as_tensor(X_np[start:end], device, dtype)
        y = _as_tensor(Y_np[start:end], device, dtype)
        pred = x @ A + b
        diff = y - pred
        ss_res += float((diff * diff).sum().item())
        y_norm_sq += float((y * y).sum().item())
        del x, y, pred, diff

    out: Dict[str, Any] = {
        "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        "rel_err": (ss_res**0.5) / ((y_norm_sq**0.5) + 1e-20),
        "norm_b": float(torch.linalg.vector_norm(b).item()),
        "method": solver,
    }
    out.update(_a_diagnostics(A))
    d = int(A.shape[0])
    a_delta = A - torch.eye(d, device=A.device, dtype=A.dtype)
    a_delta_gram = a_delta.T @ a_delta
    out.update(svd_energy_from_gram(a_delta_gram, prefix="A_delta"))
    del a_delta, a_delta_gram

    del G, C, A, b, mx, my
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return out


def load_base_instruct_pairs(source_csv: Path) -> List[Dict[str, str]]:
    with source_csv.open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["source_tasks"] == SOURCE_TASK]


def prefixed(prefix: str, values: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in values.items()}


def write_markdown(rows: List[Dict[str, Any]], out_md: Path) -> None:
    groups = {
        "all Base-Instruct": rows,
        "Gemma only": [r for r in rows if "Gemma" in r["model_a"]],
        "Llama only": [r for r in rows if "Llama" in r["model_a"]],
        "Qwen only": [r for r in rows if "Qwen" in r["model_a"]],
    }
    lines = [
        "# Base-Instruct Full-Vocabulary Affine Results",
        "",
        "Source: `summary_pair.csv`, filtered to `task1_base_instruct` and recomputed with every vocabulary row (`0..vocab_size-1`) for each Base-Instruct pair.",
        "",
        "This run does not sample token rows. It fits the same centered affine relation `Y ~= X * A + b`, but computes it through streaming centered normal equations so full vocabularies can be handled without materializing one huge design matrix on GPU. It also records compact diagnostics for `A` instead of saving the full matrices.",
        "",
        "## Summary",
        "",
        "| group | n | R2_E mean | R2_U mean | rel_err_E mean | rel_err_U mean | E rel A-I mean | E identity cosine mean | E offdiag/A mean | E_delta rank95 mean | A-I rank95 mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, group in groups.items():
        lines.append(
            f"| {name} | {len(group)} | "
            f"{_fmt(_mean(float(r['R2_E']) for r in group))} | "
            f"{_fmt(_mean(float(r['R2_U']) for r in group))} | "
            f"{_fmt(_mean(float(r['rel_err_E']) for r in group))} | "
            f"{_fmt(_mean(float(r['rel_err_U']) for r in group))} | "
            f"{_fmt(_mean(float(r['E_rel_A_minus_I_over_I']) for r in group))} | "
            f"{_fmt(_mean(float(r['E_identity_cosine']) for r in group))} | "
            f"{_fmt(_mean(float(r['E_offdiag_norm_over_A']) for r in group))} | "
            f"{_fmt(_mean(float(r['E_delta_E_delta_rank_95']) for r in group))} | "
            f"{_fmt(_mean(float(r['A_delta_A_delta_rank_95']) for r in group))} |"
        )

    lines += [
        "",
        "## A Diagnostics",
        "",
        "- `E_rel_A_minus_I_over_I`: `||A - I||_F / ||I||_F`; lower means closer to identity.",
        "- `E_identity_cosine`: `trace(A) / (||A||_F ||I||_F)`; closer to 1 means A points in the identity direction.",
        "- `E_offdiag_norm_over_A`: off-diagonal Frobenius norm divided by `||A||_F`; higher means more coordinate mixing.",
        "- `E_rel_orthogonality_error_over_I`: `||A^T A - I||_F / ||I||_F`; lower means closer to an orthogonal rotation/reflection.",
        "- `E_delta_rank95` and `A-I_rank95`: smallest rank explaining 95% of squared singular-value energy.",
        "- `energy_at_1pct_h` / `energy_at_5pct_h` / `energy_at_10pct_h`: cumulative energy explained by the top 1% / 5% / 10% of hidden dimensions, for dimension-normalized spectrum comparison.",
        "",
        "## Details",
        "",
        "| model_a | model_b | tied A/B | vocab | hidden | R2_E | rel_err_E | R2_U | rel_err_U | E rel A-I/I | E I cosine | E offdiag/A | E orth err/I | E_delta rank95 | A-I rank95 | E_delta energy@5%h | A-I energy@5%h | elapsed sec |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: float(x["R2_E"]), reverse=True):
        lines.append(
            f"| `{r['model_a']}` | `{r['model_b']}` | "
            f"{r['actual_tied_a']}/{r['actual_tied_b']} | "
            f"{r['vocab_size_a']} | {r['hidden_dim_a']} | "
            f"{_fmt(r['R2_E'])} | {_fmt(r['rel_err_E'])} | "
            f"{_fmt(r['R2_U'])} | {_fmt(r['rel_err_U'])} | "
            f"{_fmt(r['E_rel_A_minus_I_over_I'])} | "
            f"{_fmt(r['E_identity_cosine'])} | "
            f"{_fmt(r['E_offdiag_norm_over_A'])} | "
            f"{_fmt(r['E_rel_orthogonality_error_over_I'])} | "
            f"{r['E_delta_E_delta_rank_95']} | "
            f"{r['A_delta_A_delta_rank_95']} | "
            f"{_fmt(r['E_delta_E_delta_energy_at_5pct_h'])} | "
            f"{_fmt(r['A_delta_A_delta_energy_at_5pct_h'])} | "
            f"{float(r['elapsed_sec']):.1f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = _REPO_ROOT / "ijcai_clean" / "results" / "task5_affine_relations"
    source_csv = out_dir / "summary_pair.csv"
    out_csv = out_dir / "summary_pair_base_instruct_full_vocab.csv"
    out_md = out_dir / "base_instruct_full_vocab_affine_report.md"
    meta_json = out_dir / "base_instruct_full_vocab_metadata.json"
    extracts = _REPO_ROOT / "extracts"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    pairs = load_base_instruct_pairs(source_csv)

    metric_fields = [
        "R2",
        "rel_err",
        "norm_b",
        "method",
        "trace_A",
        "trace_A_over_d",
        "norm_A",
        "norm_A_minus_I",
        "rel_A_minus_I_over_I",
        "rel_A_minus_I_over_A",
        "identity_cosine",
        "diag_mean",
        "diag_std",
        "diag_min",
        "diag_max",
        "offdiag_norm",
        "offdiag_norm_over_A",
        "orthogonality_error",
        "rel_orthogonality_error_over_I",
    ]
    svd_metric_fields = [
        "total_energy",
        "fro_norm",
        "numerical_rank",
        "rank_50",
        "rank_80",
        "rank_90",
        "rank_95",
        "rank_99",
        "effective_rank",
        "top_singular_values",
    ] + [f"energy_at_{k}" for k in ENERGY_AT_K] + [
        f"energy_at_{int(fraction * 100)}pct_h" for fraction in RELATIVE_ENERGY_FRACTIONS
    ]
    fields = [
        "model_a",
        "model_b",
        "source_tasks",
        "align_mode",
        "n_common",
        "n_fit",
        "full_vocab",
        "hidden_dim_a",
        "hidden_dim_b",
        "vocab_size_a",
        "vocab_size_b",
        "actual_tied_a",
        "actual_tied_b",
    ]
    fields += [f"E_{x}" for x in metric_fields] + [f"U_{x}" for x in metric_fields]
    fields += [f"E_delta_E_delta_{x}" for x in svd_metric_fields]
    fields += [f"A_delta_A_delta_{x}" for x in svd_metric_fields]
    fields += [
        # Backward-compatible aliases for earlier full-vocab CSV consumers.
        "R2_E",
        "rel_err_E",
        "norm_A_E",
        "norm_b_E",
        "R2_U",
        "rel_err_U",
        "norm_A_U",
        "norm_b_U",
        "method",
        "elapsed_sec",
    ]

    rows: List[Dict[str, Any]] = []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, pair in enumerate(pairs, 1):
            t0 = time.time()
            model_a = pair["model_a"]
            model_b = pair["model_b"]
            print(f"[{idx}/{len(pairs)}] {model_a} -> {model_b}", flush=True)

            info_a = load_info_json(extracts, model_a)
            info_b = load_info_json(extracts, model_b)
            E_a, U_a, info_a = load_E_U_matrices(extracts, model_a, info=info_a)
            E_b, U_b, info_b = load_E_U_matrices(extracts, model_b, info=info_b)

            vocab_a, dim_a = E_a.shape
            vocab_b, dim_b = E_b.shape
            if vocab_a != vocab_b:
                raise ValueError(f"{model_a} vs {model_b}: vocab differs {vocab_a} vs {vocab_b}")
            if dim_a != dim_b:
                raise ValueError(f"{model_a} vs {model_b}: hidden differs {dim_a} vs {dim_b}")

            tied_a = actual_tied(E_a, U_a)
            tied_b = actual_tied(E_b, U_b)
            e_delta_gram = gram_of_row_delta(E_a, E_b, device=device, dtype=dtype)
            e_delta_svd = svd_energy_from_gram(e_delta_gram, prefix="E_delta")
            del e_delta_gram

            e_diag = full_affine_stream(E_a, E_b, device=device, dtype=dtype)
            if tied_a and tied_b:
                u_diag = dict(e_diag)
                u_diag["method"] = str(u_diag["method"]) + " (copied from tied E)"
            else:
                u_diag = full_affine_stream(U_a, U_b, device=device, dtype=dtype)

            method = (
                str(e_diag["method"])
                if e_diag["method"] == u_diag["method"]
                else str(e_diag["method"]) + "; U: " + str(u_diag["method"])
            )
            row: Dict[str, Any] = {
                "model_a": model_a,
                "model_b": model_b,
                "source_tasks": SOURCE_TASK,
                "align_mode": "id_full_vocab",
                "n_common": vocab_a,
                "n_fit": vocab_a,
                "full_vocab": True,
                "hidden_dim_a": dim_a,
                "hidden_dim_b": dim_b,
                "vocab_size_a": vocab_a,
                "vocab_size_b": vocab_b,
                "actual_tied_a": tied_a,
                "actual_tied_b": tied_b,
                "R2_E": e_diag["R2"],
                "rel_err_E": e_diag["rel_err"],
                "norm_A_E": e_diag["norm_A"],
                "norm_b_E": e_diag["norm_b"],
                "R2_U": u_diag["R2"],
                "rel_err_U": u_diag["rel_err"],
                "norm_A_U": u_diag["norm_A"],
                "norm_b_U": u_diag["norm_b"],
                "method": method,
                "elapsed_sec": time.time() - t0,
            }
            row.update(
                prefixed(
                    "E",
                    {k: v for k, v in e_diag.items() if not k.startswith("A_delta_")},
                )
            )
            row.update(
                prefixed(
                    "U",
                    {k: v for k, v in u_diag.items() if not k.startswith("A_delta_")},
                )
            )
            row.update(prefixed("E_delta", e_delta_svd))
            row.update(
                prefixed(
                    "A_delta",
                    {k: v for k, v in e_diag.items() if k.startswith("A_delta_")},
                )
            )
            writer.writerow(row)
            f.flush()
            rows.append(row)
            print(
                f"  R2_E={row['R2_E']:.6f} R2_U={row['R2_U']:.6f} "
                f"rel_A-I/I={row['E_rel_A_minus_I_over_I']:.4f} "
                f"E_rank95={row['E_delta_E_delta_rank_95']} "
                f"A-I_rank95={row['A_delta_A_delta_rank_95']} "
                f"n={vocab_a} d={dim_a} elapsed={row['elapsed_sec']:.1f}s",
                flush=True,
            )

            del E_a, U_a, E_b, U_b
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_markdown(rows, out_md)
    meta_json.write_text(
        json.dumps(
            {
                "task": "base_instruct_full_vocab_affine",
                "source_csv": str(source_csv.relative_to(_REPO_ROOT)),
                "out_csv": str(out_csv.relative_to(_REPO_ROOT)),
                "out_md": str(out_md.relative_to(_REPO_ROOT)),
                "n_pairs": len(rows),
                "device": str(device),
                "batch_rows": BATCH_ROWS,
                "dtype": str(dtype),
                "method": "full id-aligned vocabulary, centered normal equations, no row sampling, A diagnostics and SVD energy included",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"DONE {out_csv}", flush=True)


if __name__ == "__main__":
    main()
