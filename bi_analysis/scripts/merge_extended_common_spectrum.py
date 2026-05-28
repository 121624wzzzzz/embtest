#!/usr/bin/env python3
"""Merge extended-4 common-k D/P/R spectra and decomposition SVD into __tep tables."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEP = REPO / "__tep" / "affine" / "tables"
BI = REPO / "bi_analysis" / "tables"

EXTENDED = [
    "DeepSeek-V3-Base",
    "DeepSeek-V3.1-Base",
    "Qwen3-30B-A3B-Base",
    "Qwen3.5-35B-A3B-Base",
]

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

E_SPEC = TEP / "e" / "affine_pred_delta_common_spectrum.csv"
U_SPEC = TEP / "u" / "unembed_pred_delta_common_spectrum.csv"
E_DECOMP = TEP / "e" / "affine_task6_decomposition_svd.csv"
U_DECOMP = TEP / "u" / "unembed_task6_decomposition_svd.csv"

TMP_E_DPR = Path("/tmp/extended_e_dpr.csv")
TMP_U_DPR = Path("/tmp/extended_u_dpr.csv")
TMP_E_DECOMP = Path("/tmp/extended_e_decomp.csv")
TMP_U_DECOMP = Path("/tmp/extended_u_decomp.csv")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return (list(rows[0].keys()) if rows else []), rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def merge_by_model(
    existing_path: Path,
    new_path: Path,
    *,
    drop_cols: set[str] | None = None,
    transform=None,
    strip_prefixes: tuple[str, ...] = (),
) -> tuple[list[str], list[dict[str, str]]]:
    fields, existing = read_csv(existing_path)
    _, new_rows = read_csv(new_path)
    drop_cols = drop_cols or set()
    by_model = {r["model_a"]: r for r in existing if r["model_a"] not in EXTENDED}
    for r in new_rows:
        row = dict(r)
        if transform:
            row = transform(row)
        for c in drop_cols:
            row.pop(c, None)
        for pref in strip_prefixes:
            for k in list(row):
                if k.startswith(pref):
                    row.pop(k, None)
        by_model[r["model_a"]] = row
    merged = [by_model[k] for k in sorted(by_model, key=lambda x: (x not in EXTENDED, x))]
    out_fields = [c for c in fields if c not in drop_cols]
    for c in merged[0]:
        if c not in out_fields:
            out_fields.append(c)
    return out_fields, merged


def e_spec_transform(row: dict[str, str]) -> dict[str, str]:
    row = dict(row)
    row.pop("matrix_kind", None)
    return row


def e_decomp_transform(row: dict[str, str]) -> dict[str, str]:
    mapping = {
        "E_R2": "reported_affine_R2",
        "E_delta_rank95_over_h": "Delta_rank95_over_h",
        "E_delta_eff_over_h": "Delta_eff_over_h",
        "E_delta_energy_5pct_h": "Delta_energy_5pct_h",
    }
    out = {
        "model_a": row["model_a"],
        "model_b": row["model_b"],
        "series": row["series"],
        "hidden_dim": row["hidden_dim"],
        "is_anomaly": row["is_anomaly"],
        "Centered_delta_rank95_over_h": row["Centered_delta_rank95_over_h"],
        "A_delta_rank95_over_h": row["A_delta_rank95_over_h"],
        "Pred_delta_rank95_over_h": row["Pred_delta_rank95_over_h"],
        "Residual_rank95_over_h": row["Residual_rank95_over_h"],
        "Centered_delta_eff_over_h": row["Centered_delta_eff_over_h"],
        "A_delta_eff_over_h": row["A_delta_eff_over_h"],
        "Pred_delta_eff_over_h": row["Pred_delta_eff_over_h"],
        "Residual_eff_over_h": row["Residual_eff_over_h"],
        "Centered_delta_energy_5pct_h": row["Centered_delta_energy_5pct_h"],
        "A_delta_energy_5pct_h": row["A_delta_energy_5pct_h"],
        "Pred_delta_energy_5pct_h": row["Pred_delta_energy_5pct_h"],
        "Residual_energy_5pct_h": row["Residual_energy_5pct_h"],
        "Mean_shift_energy_5pct_h": row["Mean_shift_energy_5pct_h"],
        "Pred_delta_rank_95": row["Pred_delta_rank_95"],
        "Pred_delta_effective_rank": row["Pred_delta_effective_rank"],
        "Pred_delta_total_energy": row["Pred_delta_total_energy"],
        "Centered_delta_total_energy": row["Centered_delta_total_energy"],
        "Residual_total_energy": row["Residual_total_energy"],
        "Mean_shift_total_energy": row["Mean_shift_total_energy"],
        "Residual_over_centered_delta_energy": row["Residual_over_centered_delta_energy"],
        "Mean_shift_over_raw_delta_energy": row["Mean_shift_over_raw_delta_energy"],
        "solver": row["solver"],
    }
    for dst, src in mapping.items():
        out[dst] = row[src]
    return out


def write_summary(spec_path: Path, summary_path: Path) -> None:
    _, rows = read_csv(spec_path)
    summary_rows = []
    for k in RANKS:
        d = [float(r[f"D_energy_at_{k}"]) for r in rows]
        p = [float(r[f"P_energy_at_{k}"]) for r in rows]
        pa = [float(r[f"P_abs_delta_energy_at_{k}"]) for r in rows]
        summary_rows.append(
            {
                "k": k,
                "D_median": statistics.median(d),
                "P_median": statistics.median(p),
                "P_abs_delta_median": statistics.median(pa),
                "D_mean": statistics.mean(d),
                "P_mean": statistics.mean(p),
                "P_abs_delta_mean": statistics.mean(pa),
            }
        )
    write_csv(
        summary_path,
        ["k", "D_median", "P_median", "P_abs_delta_median", "D_mean", "P_mean", "P_abs_delta_mean"],
        summary_rows,
    )


def write_bi_extended_dpr() -> None:
    _, e_rows = read_csv(TMP_E_DPR)
    _, u_rows = read_csv(TMP_U_DPR)
    u_by = {r["model_a"]: r for r in u_rows}
    out_rows = []
    for er in e_rows:
        ma = er["model_a"]
        ur = u_by[ma]
        base = {
            "model_a": ma,
            "model_b": er["model_b"],
            "analysis_tier": "extended",
            "hidden_dim": er["hidden_dim"],
            "identity_R2_E": er["identity_R2"],
            "identity_R2_U": ur["identity_R2"],
            "full_affine_gain_over_delta_E": er["full_affine_gain_over_delta"],
            "full_affine_gain_over_delta_U": ur["full_affine_gain_over_delta"],
        }
        for side, row in [("E", er), ("U", ur)]:
            rec = dict(base)
            rec["side"] = side
            for k in RANKS:
                rec[f"D_at_{k}"] = row[f"D_energy_at_{k}"]
                rec[f"P_at_{k}"] = row[f"P_energy_at_{k}"]
                rec[f"R_at_{k}"] = row[f"R_energy_at_{k}"]
            out_rows.append(rec)

    fields = list(out_rows[0].keys())
    write_csv(BI / "extended_untied_dpr_common_spectrum.csv", fields, out_rows)

    md = [
        "# Extended 4 对 untied：D/P/R common-k 功率谱",
        "",
        "与 `__tep/scripts/evaluate_pred_delta_common_spectrum.py` 同口径（centered ΔW，`P=X_c(A-I)`，`R=Y_c-X_cA`）。",
        "已合并入主表 `__tep/affine/tables/e|u/*_pred_delta_common_spectrum.csv`（D/P/P_abs）；本表额外含 **R** 列。",
        "",
    ]
    for ma in EXTENDED:
        md.append(f"## {ma}")
        md.append("")
        md.append("| k | E: C_D | E: C_P | E: C_R | U: C_D | U: C_P | U: C_R |")
        md.append("|---:|---:|---:|---:|---:|---:|---:|")
        er = next(r for r in e_rows if r["model_a"] == ma)
        ur = u_by[ma]
        for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
            if k > int(er["hidden_dim"]):
                continue
            md.append(
                f"| {k} | {float(er[f'D_energy_at_{k}']):.4f} | {float(er[f'P_energy_at_{k}']):.4f} | "
                f"{float(er[f'R_energy_at_{k}']):.4f} | {float(ur[f'D_energy_at_{k}']):.4f} | "
                f"{float(ur[f'P_energy_at_{k}']):.4f} | {float(ur[f'R_energy_at_{k}']):.4f} |"
            )
        md.append("")
    (BI / "EXTENDED_DPR_COMMON_SPECTRUM.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    for p in (TMP_E_DPR, TMP_U_DPR, TMP_E_DECOMP, TMP_U_DECOMP):
        if not p.exists():
            raise SystemExit(f"missing {p}; run evaluate/compute scripts first")

    e_fields, e_merged = merge_by_model(
        E_SPEC,
        TMP_E_DPR,
        transform=e_spec_transform,
        strip_prefixes=("R_energy_at_",),
    )
    u_fields, u_merged = merge_by_model(
        U_SPEC, TMP_U_DPR, strip_prefixes=("R_energy_at_",)
    )
    write_csv(E_SPEC, e_fields, e_merged)
    write_csv(U_SPEC, u_fields, u_merged)
    write_summary(E_SPEC, TEP / "e" / "affine_pred_delta_common_spectrum_summary.csv")
    write_summary(U_SPEC, TEP / "u" / "unembed_pred_delta_common_spectrum_summary.csv")

    de_fields, de_merged = merge_by_model(
        E_DECOMP, TMP_E_DECOMP, transform=e_decomp_transform
    )
    du_fields, du_merged = merge_by_model(U_DECOMP, TMP_U_DECOMP)
    write_csv(E_DECOMP, de_fields, de_merged)
    write_csv(U_DECOMP, du_fields, du_merged)

    write_bi_extended_dpr()
    print(f"merged E spec -> {E_SPEC} ({len(e_merged)} rows)")
    print(f"merged U spec -> {U_SPEC} ({len(u_merged)} rows)")
    print(f"merged E decomp -> {E_DECOMP} ({len(de_merged)} rows)")
    print(f"merged U decomp -> {U_DECOMP} ({len(du_merged)} rows)")
    print(f"wrote {BI / 'extended_untied_dpr_common_spectrum.csv'}")
    print(f"wrote {BI / 'EXTENDED_DPR_COMMON_SPECTRUM.md'}")


if __name__ == "__main__":
    main()
