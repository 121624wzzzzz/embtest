#!/usr/bin/env python3
"""Build bi_analysis 30-pair Aff/LoRA budget table (main 26 + extended 4).

Reads:
  - __tep/affine/tables/final/model_level_e_u_affine_lora_summary.csv (main 26)
  - bi_analysis/tables/w_rank_budget_extended_{e,u}.csv (extended sweep)
  - __tep/affine/tables/{e,u}/*task6_decomposition_svd.csv
  - bi_pairs.yaml

Writes:
  - bi_analysis/tables/affine_lora_budget_summary.csv (30 rows)
  - bi_analysis/tables/affine_lora_by_tied_summary.csv
  - bi_analysis/tables/affine_lora_by_family_size_summary.csv
"""

from __future__ import annotations

import csv
import statistics as stats
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
BI = ROOT / "bi_analysis"
TEP = ROOT / "__tep" / "affine" / "tables"

MAIN_CSV = TEP / "final" / "model_level_e_u_affine_lora_summary.csv"
EXT_E = BI / "tables" / "w_rank_budget_extended_e.csv"
EXT_U = BI / "tables" / "w_rank_budget_extended_u.csv"
TASK6_E = TEP / "e" / "affine_task6_decomposition_svd.csv"
TASK6_U = TEP / "u" / "unembed_task6_decomposition_svd.csv"
PAIRS_YAML = BI / "bi_pairs.yaml"

OUT_MAIN = BI / "tables" / "affine_lora_budget_summary.csv"
OUT_TIED = BI / "tables" / "affine_lora_by_tied_summary.csv"
OUT_FAMILY = BI / "tables" / "affine_lora_by_family_size_summary.csv"

EXTENDED_MODELS = {
    "DeepSeek-V3-Base",
    "DeepSeek-V3.1-Base",
    "Qwen3-30B-A3B-Base",
    "Qwen3.5-35B-A3B-Base",
}

W_RANKS = (1, 2, 4, 8, 16, 32, 64)
HYBRID_RANKS = (1, 2, 4, 8)


def load_yaml_pairs() -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(PAIRS_YAML.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for family, block in data["families"].items():
        for pair in block["pairs"]:
            if pair.get("analysis_tier") == "excluded":
                continue
            base = pair["base"]
            out[base] = {
                "family": family,
                "model_a": base,
                "model_b": pair["instruct"],
                "tied": bool(pair["tied"]),
                "hidden_dim": int(pair["hidden_dim"]),
                "analysis_tier": pair["analysis_tier"],
            }
    return out


def index_csv(path: Path, key: str = "model_a") -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {r[key]: r for r in csv.DictReader(f)}


def sweep_by_model(path: Path) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["model_a"], []).append(row)
    for rows in out.values():
        rows.sort(key=lambda r: int(r["rank_w"]))
    return out


def boundary_rank_w_first_3_losses(rows: list[dict[str, str]]) -> int | None:
    """First rW where affine centered gain < W gain for 3 consecutive ranks."""
    for i in range(len(rows) - 2):
        if all(
            float(rows[i + j]["rank_affine_update_gain"])
            < float(rows[i + j]["w_rank_update_gain"])
            for j in range(3)
        ):
            return int(rows[i]["rank_w"])
    return None


def hybrid_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    sub = [r for r in rows if int(r["rank_w"]) in HYBRID_RANKS]
    ratios_raw = []
    for r in sub:
        w = float(r["w_rank_update_gain_over_raw"])
        a = float(r["rank_affine_update_gain_over_raw"])
        if w > 0:
            ratios_raw.append(a / w)
    wins = sum(
        1
        for r in sub
        if float(r["rank_affine_update_gain"]) > float(r["w_rank_update_gain"])
    )
    stable = wins == len(sub) and len(sub) == len(HYBRID_RANKS)
    return {
        "hybrid_stable": stable,
        "hybrid_min_gain_ratio": min(ratios_raw) if ratios_raw else "",
        "hybrid_median_gain_ratio": stats.median(ratios_raw) if ratios_raw else "",
        "hybrid_median_match_param_ratio": stats.median(
            [float(r["affine_over_w_params"]) for r in sub]
        )
        if sub
        else "",
    }


def side_from_task6_e(row: dict[str, str]) -> dict[str, float | str]:
    centered = float(row["Centered_delta_total_energy"])
    pred = float(row["Pred_delta_total_energy"])
    resid = float(row["Residual_total_energy"])
    p_over_d = pred / centered if centered > 0 else 0.0
    r_over_d = resid / centered if centered > 0 else 0.0
    full_affine_r2 = float(row["E_R2"])
    mean_shift = float(row["Mean_shift_over_raw_delta_energy"])
    raw = centered + float(row["Mean_shift_total_energy"])
    identity_r2 = 1.0 - centered / (centered + float(row["Mean_shift_total_energy"])) if raw > 0 else full_affine_r2
    return {
        "identity_R2": identity_r2,
        "full_affine_R2": full_affine_r2,
        "delta_R2": full_affine_r2 - identity_r2,
        "delta_over_Yc": centered / (centered + float(row["Mean_shift_total_energy"])) if raw > 0 else 0.0,
        "P_over_D": p_over_d,
        "R_over_D": r_over_d,
        "mean_shift_over_raw_delta": mean_shift,
        "D_rank95_over_h": float(row["E_delta_rank95_over_h"]),
        "P_rank95_over_h": float(row["Pred_delta_rank95_over_h"]),
        "AminusI_rank95_over_h": float(row["A_delta_rank95_over_h"]),
        "R_rank95_over_h": float(row["Residual_rank95_over_h"]),
        "D_eff_over_h": float(row["E_delta_eff_over_h"]),
        "P_eff_over_h": float(row["Pred_delta_eff_over_h"]),
        "AminusI_eff_over_h": float(row["A_delta_eff_over_h"]),
        "R_eff_over_h": float(row["Residual_eff_over_h"]),
        "D_energy_5pct_h": float(row["E_delta_energy_5pct_h"]),
        "P_energy_5pct_h": float(row["Pred_delta_energy_5pct_h"]),
        "AminusI_energy_5pct_h": float(row["A_delta_energy_5pct_h"]),
        "R_energy_5pct_h": float(row["Residual_energy_5pct_h"]),
    }


def side_from_task6_u(row: dict[str, str]) -> dict[str, float | str]:
    centered = float(row["Centered_delta_total_energy"])
    pred = float(row["Pred_delta_total_energy"])
    resid = float(row["Residual_total_energy"])
    p_over_d = pred / centered if centered > 0 else 0.0
    r_over_d = resid / centered if centered > 0 else 0.0
    full_affine_r2 = float(row["reported_affine_R2"])
    mean_shift = float(row["Mean_shift_over_raw_delta_energy"])
    raw = float(row["Delta_total_energy"])
    identity_r2 = 1.0 - centered / raw if raw > 0 else full_affine_r2
    return {
        "identity_R2": identity_r2,
        "full_affine_R2": full_affine_r2,
        "delta_R2": full_affine_r2 - identity_r2,
        "delta_over_Yc": centered / raw if raw > 0 else 0.0,
        "P_over_D": p_over_d,
        "R_over_D": r_over_d,
        "mean_shift_over_raw_delta": mean_shift,
        "D_rank95_over_h": float(row["Delta_rank95_over_h"]),
        "P_rank95_over_h": float(row["Pred_delta_rank95_over_h"]),
        "AminusI_rank95_over_h": float(row["A_delta_rank95_over_h"]),
        "R_rank95_over_h": float(row["Residual_rank95_over_h"]),
        "D_eff_over_h": float(row["Delta_eff_over_h"]),
        "P_eff_over_h": float(row["Pred_delta_eff_over_h"]),
        "AminusI_eff_over_h": float(row["A_delta_eff_over_h"]),
        "R_eff_over_h": float(row["Residual_eff_over_h"]),
        "D_energy_5pct_h": float(row["Delta_energy_5pct_h"]),
        "P_energy_5pct_h": float(row["Pred_delta_energy_5pct_h"]),
        "AminusI_energy_5pct_h": float(row["A_delta_energy_5pct_h"]),
        "R_energy_5pct_h": float(row["Residual_energy_5pct_h"]),
    }


def side_from_sweep(rows: list[dict[str, str]], prefix: str) -> dict[str, Any]:
    by_rw = {int(r["rank_w"]): r for r in rows}
    out: dict[str, Any] = {}
    for rw in W_RANKS:
        r = by_rw[rw]
        out[f"{prefix}_aff_vs_W_ratio_r{rw}"] = float(r["rank_affine_gain_over_w_gain"])
    out[f"{prefix}_aff_rank_budget_r1"] = int(by_rw[1]["rank_affine_budgeted"])
    out[f"{prefix}_aff_rank_budget_r8"] = int(by_rw[8]["rank_affine_budgeted"])
    b = boundary_rank_w_first_3_losses(rows)
    out[f"{prefix}_boundary_rank_w_first_3_losses"] = b if b is not None else ""
    h = hybrid_metrics(rows)
    out[f"{prefix}_hybrid_stable_small_budget_both"] = h["hybrid_stable"]
    out[f"{prefix}_hybrid_min_gain_ratio_1_2_4_8"] = h["hybrid_min_gain_ratio"]
    out[f"{prefix}_hybrid_median_gain_ratio_1_2_4_8"] = h["hybrid_median_gain_ratio"]
    out[f"{prefix}_hybrid_median_match_param_ratio_1_2_4_8"] = h["hybrid_median_match_param_ratio"]
    return out


def build_extended_row(
    meta: dict[str, Any],
    task6_e: dict[str, str],
    task6_u: dict[str, str],
    sweep_e: list[dict[str, str]],
    sweep_u: list[dict[str, str]],
) -> dict[str, Any]:
    e = side_from_task6_e(task6_e)
    u = side_from_task6_u(task6_u)
    row: dict[str, Any] = {
        "family": meta["family"],
        "size": "extended",
        "model_a": meta["model_a"],
        "model_b": meta["model_b"],
        "tie_word_embeddings": meta["tied"],
        "hidden_dim": meta["hidden_dim"],
        "vocab": int(sweep_e[0]["vocab"]),
        "E_U_same_matrix": meta["tied"],
        "analysis_tier": meta["analysis_tier"],
    }
    for side, data, prefix in (("E", e, "E"), ("U", u, "U")):
        row[f"{prefix}_identity_R2"] = data["identity_R2"]
        row[f"{prefix}_full_affine_R2"] = data["full_affine_R2"]
        row[f"{prefix}_delta_R2"] = data["delta_R2"]
        row[f"{prefix}_delta_over_Yc"] = data["delta_over_Yc"]
        row[f"{prefix}_P_over_D_full_affine_gain"] = data["P_over_D"]
        row[f"{prefix}_R_over_D"] = data["R_over_D"]
        row[f"{prefix}_mean_shift_over_raw_delta"] = data["mean_shift_over_raw_delta"]
        for k in (
            "D_rank95_over_h",
            "P_rank95_over_h",
            "AminusI_rank95_over_h",
            "R_rank95_over_h",
            "D_eff_over_h",
            "P_eff_over_h",
            "AminusI_eff_over_h",
            "R_eff_over_h",
            "D_energy_5pct_h",
            "P_energy_5pct_h",
            "AminusI_energy_5pct_h",
            "R_energy_5pct_h",
        ):
            row[f"{prefix}_{k}"] = data[k]
    row.update(side_from_sweep(sweep_e, "E"))
    row.update(side_from_sweep(sweep_u, "U"))
    u_p = float(row["U_P_over_D_full_affine_gain"])
    e_p = float(row["E_P_over_D_full_affine_gain"])
    row["U_minus_E_full_affine_gain"] = u_p - e_p
    row["U_over_E_full_affine_gain"] = u_p / e_p if e_p > 0 else ""
    row["U_minus_E_aff_vs_W_ratio_r1"] = float(row["U_aff_vs_W_ratio_r1"]) - float(
        row["E_aff_vs_W_ratio_r1"]
    )
    row["U_minus_E_hybrid_min_gain_ratio"] = (
        float(row["U_hybrid_min_gain_ratio_1_2_4_8"])
        - float(row["E_hybrid_min_gain_ratio_1_2_4_8"])
        if row["U_hybrid_min_gain_ratio_1_2_4_8"] != ""
        and row["E_hybrid_min_gain_ratio_1_2_4_8"] != ""
        else ""
    )
    return row


def summarize_group(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    from itertools import groupby

    def f(row: dict[str, Any], col: str) -> float:
        v = row[col]
        return float(v) if v != "" else float("nan")

    skip = {
        "E_U_same_matrix",
        "E_hybrid_stable_small_budget_both",
        "U_hybrid_stable_small_budget_both",
    }
    numeric_cols = [
        c
        for c in rows[0].keys()
        if c.startswith(("E_", "U_")) and c not in skip
    ]
    win_cols = [
        ("E_aff_wins_r1", "E_aff_vs_W_ratio_r1"),
        ("E_aff_wins_r8", "E_aff_vs_W_ratio_r8"),
        ("U_aff_wins_r1", "U_aff_vs_W_ratio_r1"),
        ("U_aff_wins_r8", "U_aff_vs_W_ratio_r8"),
        ("E_hybrid_stable_count", "E_hybrid_stable_small_budget_both"),
        ("U_hybrid_stable_count", "U_hybrid_stable_small_budget_both"),
    ]
    out = []
    for key, group in groupby(sorted(rows, key=lambda r: r[group_key]), key=lambda r: r[group_key]):
        g = list(group)
        summary: dict[str, Any] = {"group": key, "n": len(g)}
        for col in numeric_cols:
            vals = [f(r, col) for r in g]
            vals = [v for v in vals if v == v]
            if not vals:
                continue
            summary[f"{col}_mean"] = stats.mean(vals)
            summary[f"{col}_median"] = stats.median(vals)
        for win_name, src in win_cols:
            if src.endswith("_both"):
                summary[win_name] = sum(1 for r in g if r[src] in (True, "True", "true", 1))
            else:
                summary[win_name] = sum(1 for r in g if f(r, src) > 1.0)
        out.append(summary)
    return out


def main() -> None:
    pairs = load_yaml_pairs()
    main_rows = index_csv(MAIN_CSV)
    task6_e = index_csv(TASK6_E)
    task6_u = index_csv(TASK6_U)
    ext_e = sweep_by_model(EXT_E)
    ext_u = sweep_by_model(EXT_U)

    fieldnames = csv.DictReader(MAIN_CSV.open(newline="", encoding="utf-8")).fieldnames
    assert fieldnames is not None

    merged: list[dict[str, Any]] = []
    for model_a in sorted(pairs.keys()):
        meta = pairs[model_a]
        if model_a in EXTENDED_MODELS:
            row = build_extended_row(
                meta,
                task6_e[model_a],
                task6_u[model_a],
                ext_e[model_a],
                ext_u[model_a],
            )
        else:
            row = dict(main_rows[model_a])
            row["analysis_tier"] = meta["analysis_tier"]
        merged.append(row)

    assert len(merged) == 30, f"expected 30 rows, got {len(merged)}"

    extra = {"analysis_tier"}
    out_fields = list(fieldnames) + [c for c in extra if c not in fieldnames]

    with OUT_MAIN.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)
    print(f"WROTE {OUT_MAIN} ({len(merged)} rows)")

    tied_rows = []
    for r in merged:
        rr = dict(r)
        rr["group"] = f"tied={r['tie_word_embeddings']}"
        tied_rows.append(rr)
    tied_summary = summarize_group(tied_rows, "group")
    with OUT_TIED.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tied_summary[0].keys()))
        w.writeheader()
        w.writerows(tied_summary)
    print(f"WROTE {OUT_TIED}")

    fam_rows = []
    for r in merged:
        rr = dict(r)
        rr["group"] = f"{r['family']}|{r.get('size', '')}"
        fam_rows.append(rr)
    fam_summary = summarize_group(fam_rows, "group")
    with OUT_FAMILY.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fam_summary[0].keys()))
        w.writeheader()
        w.writerows(fam_summary)
    print(f"WROTE {OUT_FAMILY}")


if __name__ == "__main__":
    main()
