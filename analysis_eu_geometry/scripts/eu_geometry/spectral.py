"""Layer 3: GPU spectral / isotropy audit."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import (
    F_LAYER3_SPECTRAL,
    catalog_context,
    layers_dir,
    layers_file,
    row_metadata,
    write_long_csv,
)

SPECTRAL_FIELDS = [
    "model",
    "subset",
    "model_group",
    "role",
    "matrix",
    "mean_row_norm",
    "mean_vec_norm",
    "mu_over_row_norm",
    "sigma1",
    "sigma1_centered",
    "sigma1_over_mean_row",
    "sigma1_c_over_mean_row",
    "sigma1_over_mean_sqrt_n",
    "sigma1_c_over_mean_sqrt_n",
    "cos_v1_mu",
    "cos_v1_v1_centered",
    "rank1_energy_frac",
    "rank5_energy_frac",
    "rank10_energy_frac",
    "participation_ratio",
    "effective_rank",
    "isotropy_pr_over_d",
    "sigma_ratio",
    "rank1_centered_energy_frac",
    "rank5_centered_energy_frac",
    "participation_ratio_centered",
    "effective_rank_centered",
    "isotropy_pr_over_d_centered",
    "sigma_ratio_centered",
    "tied",
    "vocab_size",
    "hidden_dim",
]


def _full_spectrum_metrics(
    mat: "torch.Tensor",
    n: int,
    *,
    mu_hat: "torch.Tensor | None" = None,
) -> tuple[dict[str, float], "torch.Tensor"]:
    """All d singular values via Gram eigendecomposition (full economy SVD)."""
    import torch

    g = (mat.T @ mat) / n
    evals, evecs = torch.linalg.eigh(g)
    sigmas_sq = torch.clamp(evals * n, min=0.0)
    sigmas = torch.sqrt(sigmas_sq)
    total = sigmas_sq.sum()
    eps = 1e-12
    p = sigmas_sq / (total + eps)
    dim = int(p.numel())

    rank1_frac = float(p[-1].cpu())
    rank5_frac = float(p[-5:].sum().cpu()) if dim >= 5 else float(total.cpu())
    rank10_frac = float(p[-10:].sum().cpu()) if dim >= 10 else float(total.cpu())
    participation_ratio = float((total * total / (sigmas_sq * sigmas_sq).sum()).cpu())
    effective_rank = float(torch.exp(-(p * torch.log(p + eps)).sum()).cpu())
    sigma_max = float(sigmas[-1].cpu())
    sigma_min = float(sigmas[0].cpu())
    sigma_ratio = sigma_max / (sigma_min + eps)
    isotropy_pr_over_d = participation_ratio / dim if dim else float("nan")

    out: dict[str, float] = {
        "rank1_energy_frac": rank1_frac,
        "rank5_energy_frac": rank5_frac,
        "rank10_energy_frac": rank10_frac,
        "participation_ratio": participation_ratio,
        "effective_rank": effective_rank,
        "isotropy_pr_over_d": isotropy_pr_over_d,
        "sigma_ratio": sigma_ratio,
        "sigma1": sigma_max,
    }
    v1 = evecs[:, -1]
    if mu_hat is not None:
        out["cos_v1_mu"] = float(torch.abs(torch.dot(v1, mu_hat)).cpu())
    return out, v1


def spectral_stats_gpu(M: np.ndarray, device: str = "cuda:0") -> dict[str, float]:
    """Full economy SVD stats via d×d Gram on GPU (all d singular values)."""
    import torch

    X = torch.as_tensor(M, device=device, dtype=torch.float32)
    n = X.shape[0]
    row_norms = torch.linalg.norm(X, dim=1)
    mean_row_norm = float(row_norms.mean().cpu())
    mu = X.mean(dim=0)
    mean_vec_norm = float(torch.linalg.norm(mu).cpu())
    ratio = mean_vec_norm / mean_row_norm if mean_row_norm else float("nan")
    mu_hat = mu / (torch.linalg.norm(mu) + 1e-12)

    unc, v1_u = _full_spectrum_metrics(X, n, mu_hat=mu_hat)
    Xc = X - mu
    cen, v1_c = _full_spectrum_metrics(Xc, n)
    cos_v1_u_c = float(torch.abs(torch.dot(v1_u, v1_c)).cpu())

    sqrt_n = float(n**0.5)
    norm_denom = mean_row_norm * sqrt_n if mean_row_norm else float("nan")
    sigma1 = unc["sigma1"]
    sigma1_c = cen["sigma1"]

    del X, Xc, v1_u, v1_c
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "mean_row_norm": mean_row_norm,
        "mean_vec_norm": mean_vec_norm,
        "mu_over_row_norm": ratio,
        "sigma1": sigma1,
        "sigma1_centered": sigma1_c,
        "sigma1_over_mean_row": sigma1 / mean_row_norm if mean_row_norm else float("nan"),
        "sigma1_c_over_mean_row": sigma1_c / mean_row_norm if mean_row_norm else float("nan"),
        "sigma1_over_mean_sqrt_n": sigma1 / norm_denom,
        "sigma1_c_over_mean_sqrt_n": sigma1_c / norm_denom,
        "cos_v1_mu": unc.get("cos_v1_mu", float("nan")),
        "cos_v1_v1_centered": cos_v1_u_c,
        "rank1_energy_frac": unc["rank1_energy_frac"],
        "rank5_energy_frac": unc["rank5_energy_frac"],
        "rank10_energy_frac": unc["rank10_energy_frac"],
        "participation_ratio": unc["participation_ratio"],
        "effective_rank": unc["effective_rank"],
        "isotropy_pr_over_d": unc["isotropy_pr_over_d"],
        "sigma_ratio": unc["sigma_ratio"],
        "rank1_centered_energy_frac": cen["rank1_energy_frac"],
        "rank5_centered_energy_frac": cen["rank5_energy_frac"],
        "participation_ratio_centered": cen["participation_ratio"],
        "effective_rank_centered": cen["effective_rank"],
        "isotropy_pr_over_d_centered": cen["isotropy_pr_over_d"],
        "sigma_ratio_centered": cen["sigma_ratio"],
    }


def run_spectral_audit(
    repo: Path,
    *,
    models_yaml: Path,
    pairs_file: Path,
    extracts_dir: Path,
    device: str = "cuda:0",
) -> Path:
    import torch
    from ijcai_clean.data import load_E_U_matrices, load_info_json

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable, requested device={device}")
    if device.startswith("cuda"):
        torch.cuda.set_device(device)

    model_groups, bi_set, models = catalog_context(models_yaml, pairs_file)

    layers_dir(repo).mkdir(parents=True, exist_ok=True)
    out_csv = layers_file(repo, F_LAYER3_SPECTRAL)

    all_rows: list[dict[str, object]] = []
    for i, name in enumerate(models, 1):
        extra = row_metadata(name, bi_set=bi_set, model_groups=model_groups)
        info = load_info_json(extracts_dir, name)
        E, U, info = load_E_U_matrices(extracts_dir, name, info=info)
        emb_shape = info["standardized_dims"]["embed"]
        vocab_size, hidden_dim = int(emb_shape[0]), int(emb_shape[1])
        tied = bool(info.get("tie_word_embeddings"))
        matrices = [("E", E)] if tied else [("E", E), ("U", U)]

        for matrix_name, M in matrices:
            stats = spectral_stats_gpu(M, device=device)
            row = {
                "model": name,
                "matrix": matrix_name,
                "tied": tied,
                "vocab_size": vocab_size,
                "hidden_dim": hidden_dim,
                **extra,
                **stats,
            }
            all_rows.append(row)
            print(
                f"[{i}/{len(models)}] {name:40} {matrix_name} "
                f"PR/d={stats['isotropy_pr_over_d']:.3f} "
                f"PR/d(c)={stats['isotropy_pr_over_d_centered']:.3f} "
                f"rank1={stats['rank1_energy_frac']:.3f} "
                f"rank1(c)={stats['rank1_centered_energy_frac']:.3f}",
                flush=True,
            )

    all_rows.sort(key=lambda r: (str(r["model"]), str(r["matrix"])))
    write_long_csv(out_csv, all_rows, SPECTRAL_FIELDS)
    print(f"\nWrote {len(all_rows)} rows ({len(models)} models) -> {out_csv}", flush=True)

    from .features import merge_all_models_features

    features_csv = merge_all_models_features(repo)
    if features_csv is not None:
        print(f"Merged feature table -> {features_csv}", flush=True)
    return out_csv
