#!/usr/bin/env python3
"""Generate LaTeX longtable for 297 model comparison pairs from exp1_global_v4_complete.csv"""

import pandas as pd
import os

# Same 19 Base/Instruct pairs as in run_exp1_v4.py; use tuple(sorted(pair)) for row matching
_base_pairs = [
    ("Qwen3-0.6B-Base", "Qwen3-0.6B"), ("Qwen3-1.7B-Base", "Qwen3-1.7B"),
    ("Qwen3-4B-Base", "Qwen3-4B"), ("Qwen3-8B-Base", "Qwen3-8B"), ("Qwen3-14B-Base", "Qwen3-14B"),
    ("Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct"), ("Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct"),
    ("Qwen2.5-3B", "Qwen2.5-3B-Instruct"), ("Qwen2.5-7B", "Qwen2.5-7B-Instruct"),
    ("Qwen2.5-14B", "Qwen2.5-14B-Instruct"), ("Qwen2.5-32B", "Qwen2.5-32B-Instruct"),
    ("Qwen2.5-72B-Base", "Qwen2.5-72B-Instruct"),
    ("Llama-3.2-1B", "Llama-3.2-1B-Instruct"), ("Llama-3.2-3B", "Llama-3.2-3B-Instruct"),
    ("Llama-3.1-8B", "Llama-3.1-8B-Instruct"), ("Llama-3.1-70B-Base", "Llama-3.1-70B-Instruct"),
    ("Gemma-2-2B", "Gemma-2-2B-Instruct"), ("Gemma-2-9B", "Gemma-2-9B-Instruct"),
    ("Gemma-2-27B", "Gemma-2-27B-Instruct"),
]
BASE_INSTRUCT_PAIRS_SET = {tuple(sorted(p)) for p in _base_pairs}

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ijcai_clean/
    csv_path = os.path.join(base_dir, "results", "exp1_global_v4_complete.csv")
    df = pd.read_csv(csv_path)

    def is_base_instruct(row):
        pair = tuple(sorted([row["model_a"], row["model_b"]]))
        return pair in BASE_INSTRUCT_PAIRS_SET

    def assign_group(row):
        if is_base_instruct(row):
            return "Base/Instruct"
        if row["same_family"]:
            fa = row["family_a"]
            return f"Intra-{fa}"
        return "Cross-Family"

    df["group"] = df.apply(assign_group, axis=1)
    order = ["Base/Instruct", "Intra-Qwen3", "Intra-Qwen2.5", "Intra-Llama", "Intra-Gemma2", "Cross-Family"]
    df["_ord"] = df["group"].map(lambda g: order.index(g) if g in order else 99)
    df = df.sort_values(["_ord", "family_a", "family_b", "model_a", "model_b"]).reset_index(drop=True)

    def latex_esc(s):
        return s.replace("_", "\\_").replace("&", "\\&")

    def fmt(x):
        return f"{float(x):.3f}"

    # Compute run lengths per group
    prev_group = None
    run = 0
    group_runs = []
    for _, row in df.iterrows():
        g = row["group"]
        if g != prev_group:
            if run > 0:
                group_runs.append((prev_group, run))
            run = 1
            prev_group = g
        else:
            run += 1
    group_runs.append((prev_group, run))

    # Build data rows
    data_lines = []
    idx = 0
    for grp, cnt in group_runs:
        for j in range(cnt):
            row = df.iloc[idx]
            idx += 1
            pair = f"{latex_esc(row['model_a'])} vs.\\ {latex_esc(row['model_b'])}"
            ec = fmt(row["gcorr_E_cos_mean"])
            ee = fmt(row["gcorr_E_euc_mean"])
            uc = fmt(row["gcorr_U_cos_mean"])
            ue = fmt(row["gcorr_U_euc_mean"])
            if j == 0:
                data_lines.append(f"\\multirow{{{cnt}}}{{*}}{{{grp}}} & {pair} & {ec} & {ee} & {uc} & {ue} \\\\")
            else:
                data_lines.append(f" & {pair} & {ec} & {ee} & {uc} & {ue} \\\\")
        data_lines.append("\\midrule")
        data_lines.append("")
    if data_lines and data_lines[-2] == "\\midrule":
        data_lines = data_lines[:-2]

    # Full LaTeX snippet (body only) for inclusion
    out_path = os.path.join(base_dir, "results", "exp1_full_table_body.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("% ▼▼▼ 297 rows: paste below \\endfirsthead into your longtable ▼▼▼\n\n")
        f.write("\n".join(data_lines))
        f.write("\n\n% ▲▲▲ end data ▲▲▲\n")
    print(f"Wrote table body ({len(df)} rows) to {out_path}")

    # Full subsection + longtable
    full_path = os.path.join(base_dir, "results", "exp1_appendix_full_table.tex")
    header = r"""\subsection{Complete List of Model Comparisons}
\label{app:full_list}

Due to space constraints in the main text, we present the full breakdown of the 297 comparison pairs here. Table \ref{tab:full_results} details the geometric correlation metrics for both Input ($E$) and Output ($U$) spaces across all studied model pairs.

% --- 长表开始 ---
{
\small
\begin{longtable}{llcccc}
\caption{Complete empirical results for all 297 model pairs. We report the Pearson correlation ($\rho$) for both Euclidean (Euc) and Cosine (Cos) distances.}
\label{tab:full_results} \\

\toprule
\textbf{Group} & \textbf{Model Pair (A vs.\ B)} & \textbf{$E_{\mathrm{cos}}$} & \textbf{$E_{\mathrm{euc}}$} & \textbf{$U_{\mathrm{cos}}$} & \textbf{$U_{\mathrm{euc}}$} \\
\midrule
\endfirsthead

\caption[]{Continued from previous page} \\
\toprule
\textbf{Group} & \textbf{Model Pair (A vs.\ B)} & \textbf{$E_{\mathrm{cos}}$} & \textbf{$E_{\mathrm{euc}}$} & \textbf{$U_{\mathrm{cos}}$} & \textbf{$U_{\mathrm{euc}}$} \\
\midrule
\endhead

\midrule
\multicolumn{6}{r}{\textit{Continued on next page...}} \\
\endfoot

\bottomrule
\endlastfoot

"""
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(data_lines))
        f.write("\n\n\\end{longtable}\n}\n")
    print(f"Wrote full appendix table to {full_path}")


if __name__ == "__main__":
    main()
