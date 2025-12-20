# analysis_persona_bias.py
# Run: python analysis_persona_bias.py --input your_data.csv --outdir results

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


# -----------------------------
# 1) Persona dimension mapping
# -----------------------------
def add_persona_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes persona dimensions based on persona_id definitions you provided.
    """
    def training(pid: int) -> str:
        if 1 <= pid <= 9:
            return "US"
        if 10 <= pid <= 18:
            return "Pakistan"
        if 19 <= pid <= 26:
            return "Neutral"
        return "Unknown"

    # From your list:
    advanced_ids = {2, 4, 11, 13, 20, 22, 26, 8, 18}  # includes "all resources" (26)
    limited_ids  = {5, 9, 14, 17, 23}

    step_ids     = {3, 4, 12, 13, 21, 22}
    epi_ids      = {6, 8, 9, 15, 17, 18, 24, 26}      # includes global epidemiology (24,26)
    socio_ids    = {7, 8, 9, 16, 17, 18, 25, 26}
    guide_ids    = {8, 9, 17, 18, 26}

    # Optional: location specificity
    # US working in US: 6,8,9
    # Pakistan working in Pakistan: 15,17,18
    # Global/unknown: 24
    def location_specific(pid: int) -> str:
        if pid in {6, 8, 9}:
            return "US"
        if pid in {15, 17, 18}:
            return "Pakistan"
        if pid == 24:
            return "Global"
        if pid == 19:
            return "Unknown"
        return "None"

    df = df.copy()
    df["training"] = df["persona_id"].astype(int).apply(training)

    def resources(pid: int) -> str:
        if pid in advanced_ids:
            return "Advanced"
        if pid in limited_ids:
            return "Limited"
        return "Unspecified"

    df["resources"] = df["persona_id"].astype(int).apply(resources)
    df["reasoning"] = np.where(df["persona_id"].isin(step_ids), "StepByStep", "NoStep")
    df["epi_context"] = np.where(df["persona_id"].isin(epi_ids), "Yes", "No")
    df["socioeconomic"] = np.where(df["persona_id"].isin(socio_ids), "Yes", "No")
    df["guidelines"] = np.where(df["persona_id"].isin(guide_ids), "Yes", "No")
    df["location_specific"] = df["persona_id"].astype(int).apply(location_specific)

    # Make sure categorical ordering is stable
    df["training"] = pd.Categorical(df["training"], categories=["US", "Neutral", "Pakistan"], ordered=True)
    df["resources"] = pd.Categorical(df["resources"], categories=["Unspecified", "Advanced", "Limited"], ordered=False)

    return df


# -----------------------------------
# 2) Mixed model helper (robust-ish)
def fit_mixedlm(formula: str, df: pd.DataFrame):
    outcome = formula.split("~")[0].strip()

    must_have = [
        outcome, "persona_id", "clinical_id", "run",
        "training", "resources", "reasoning", "epi_context", "socioeconomic", "guidelines"
    ]
    must_have = [c for c in must_have if c in df.columns]

    # IMPORTANT: drop NA + reset index (prevents out-of-bounds crash)
    df2 = df.dropna(subset=must_have).copy().reset_index(drop=True)

    vc = {"clinical_id": "0 + C(clinical_id)", "run": "0 + C(run)"}
    model = smf.mixedlm(formula, df2, groups=df2["persona_id"], vc_formula=vc, re_formula="1")

    # try a few optimizers (reduces convergence failures)
    for method in ["lbfgs", "powell", "nm"]:
        try:
            return model.fit(method=method, reml=False, maxiter=500, disp=False)
        except Exception:
            pass

    # final fallback
    return model.fit(method="lbfgs", reml=False, maxiter=500, disp=False)



def coef_table(res, label: str) -> pd.DataFrame:
    """
    Create a tidy coefficient table with 95% CI.
    """
    params = res.params
    bse = res.bse
    out = pd.DataFrame({
        "term": params.index,
        "estimate": params.values,
        "std_error": bse.values,
    })
    out["ci_low"] = out["estimate"] - 1.96 * out["std_error"]
    out["ci_high"] = out["estimate"] + 1.96 * out["std_error"]
    out["p_value"] = res.pvalues.values
    out["model"] = label
    return out


# -----------------------------
# 3) Plots
# -----------------------------
def mean_ci(series: pd.Series):
    n = series.dropna().shape[0]
    m = series.mean()
    s = series.std(ddof=1)
    if n <= 1 or np.isnan(s):
        return m, np.nan, np.nan
    se = s / np.sqrt(n)
    ci = 1.96 * se
    return m, m - ci, m + ci


def barplot_means_with_ci(df: pd.DataFrame, group_col: str, value_col: str, title: str, outpath: str):
    g = (
        df.groupby(group_col, observed=True)[value_col]
          .apply(lambda x: pd.Series(mean_ci(x), index=["mean", "ci_low", "ci_high"]))
          .unstack()              # <-- THIS is the key fix
          .reset_index()
    )

    plt.figure()
    plt.bar(g[group_col].astype(str), g["mean"])

    yerr = np.vstack([g["mean"] - g["ci_low"], g["ci_high"] - g["mean"]])
    plt.errorbar(g[group_col].astype(str), g["mean"], yerr=yerr, fmt="none", capsize=4)

    plt.title(title)
    plt.ylabel(value_col)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()



def run_stability_plot(df: pd.DataFrame, outpath: str):
    """
    Line plot: mean total_score across run (1-3) for each training group, with 95% CI.
    """
    plt.figure()
    for tr in ["US", "Neutral", "Pakistan"]:
        sub = df[df["training"] == tr]
        stats = []
        for r in sorted(sub["run"].unique()):
            m, lo, hi = mean_ci(sub[sub["run"] == r]["total_score"])
            stats.append((r, m, lo, hi))
        stats = pd.DataFrame(stats, columns=["run", "mean", "lo", "hi"]).sort_values("run")
        plt.plot(stats["run"], stats["mean"], marker="o", label=tr)
        plt.fill_between(stats["run"], stats["lo"], stats["hi"], alpha=0.2)

    plt.title("Run stability: total_score by training")
    plt.xlabel("run")
    plt.ylabel("total_score")
    plt.xticks(sorted(df["run"].unique()))
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def forest_plot(coefs: pd.DataFrame, title: str, outpath: str, max_terms: int = 25):
    """
    Forest plot for coefficient estimates; keeps the most informative terms.
    """
    # Drop intercept and random-effect variance params if any weird terms appear
    c = coefs.copy()
    c = c[~c["term"].str.contains("Intercept", case=False, na=False)]
    c = c[~c["term"].str.contains("Group Var|run Var|clinical_id Var", case=False, na=False)]

    # Keep up to max_terms by absolute estimate
    c["abs_est"] = c["estimate"].abs()
    c = c.sort_values("abs_est", ascending=False).head(max_terms)
    c = c.sort_values("estimate")

    plt.figure(figsize=(7, max(3, 0.3 * len(c))))
    y = np.arange(len(c))
    plt.errorbar(c["estimate"], y, xerr=[c["estimate"] - c["ci_low"], c["ci_high"] - c["estimate"]],
                 fmt="o", capsize=4)
    plt.yticks(y, c["term"])
    plt.axvline(0, linestyle="--")
    plt.title(title)
    plt.xlabel("Effect on score (points)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def heatmap(matrix: pd.DataFrame, title: str, outpath: str):
    """
    Simple heatmap (no seaborn).
    """
    plt.figure(figsize=(10, 4))
    vals = matrix.values
    plt.imshow(vals, aspect="auto")
    plt.colorbar(label="Standardized effect size (approx)")
    plt.yticks(range(matrix.shape[0]), matrix.index.tolist())
    plt.xticks(range(matrix.shape[1]), matrix.columns.tolist(), rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -----------------------------
# 4) Main analysis (Q1–Q4)
# -----------------------------
def main(input_path: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(input_path)

    required = ["clinical_id", "persona_id", "run", "q1_score", "q2_score", "q3_score", "q4_score", "q5_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure numeric
    for c in ["clinical_id", "persona_id", "run", "q1_score", "q2_score", "q3_score", "q4_score", "q5_score"]:
        df[c] = pd.to_numeric(df[c], errors="raise")

    if "total_score" not in df.columns:
        df["total_score"] = df[["q1_score", "q2_score", "q3_score", "q4_score", "q5_score"]].sum(axis=1)

    df = add_persona_dimensions(df)

    # Save enriched dataset
    df.to_csv(os.path.join(outdir, "data_with_dimensions.csv"), index=False)

    # ---------------- Q1: training effect ----------------
    desc = df.groupby("training")["total_score"].agg(["count", "mean", "std"]).reset_index()
    desc.to_csv(os.path.join(outdir, "Q1_descriptives_by_training.csv"), index=False)

    barplot_means_with_ci(
        df, "training", "total_score",
        "Q1: total_score by training (mean ± 95% CI)",
        os.path.join(outdir, "Q1_total_by_training_bar.png")
    )

    m1 = fit_mixedlm("total_score ~ C(training)", df)
    q1_table = coef_table(m1, "Q1_total_training_only")
    q1_table.to_csv(os.path.join(outdir, "Q1_mixedlm_coefficients.csv"), index=False)

    # ---------------- Q2: dimension drivers ----------------
    # NOTE: keep it readable; add interactions most relevant to your hypothesis
    formula_q2 = (
        "total_score ~ C(training) + C(resources) + C(reasoning) + "
        "C(epi_context) + C(socioeconomic) + C(guidelines) + "
        "C(training):C(resources) + C(training):C(epi_context)"
    )
    m2 = fit_mixedlm(formula_q2, df)
    q2_table = coef_table(m2, "Q2_total_dimension_model")
    q2_table.to_csv(os.path.join(outdir, "Q2_mixedlm_coefficients.csv"), index=False)

    forest_plot(
        q2_table,
        "Q2: Dimension-level effects on total_score (top terms)",
        os.path.join(outdir, "Q2_dimension_effects_forest.png")
    )

    # ---------------- Q3: run consistency ----------------
    # A) training * run interaction
    m3 = fit_mixedlm("total_score ~ C(training) * C(run)", df)
    q3_table = coef_table(m3, "Q3_total_training_by_run")
    q3_table.to_csv(os.path.join(outdir, "Q3_mixedlm_training_x_run.csv"), index=False)

    # B) run stability plot
    run_stability_plot(df, os.path.join(outdir, "Q3_run_stability_lines.png"))

    # Simple run-to-run correlations (descriptive)
    pivot = df.pivot_table(
        index=["clinical_id", "persona_id"], columns="run", values="total_score", aggfunc="mean"
    )
    corr = pivot.corr()
    corr.to_csv(os.path.join(outdir, "Q3_run_correlations.csv"))

    # ---------------- Q4: rubric sensitivity (q1–q5) ----------------
    q_tables = []
    pvals = []
    rows = []  # for heatmap: standardized effects per q

    # We'll focus on training (Pakistan vs US) and optionally other dimensions
    # Standardized effect approx: estimate / SD(outcome)
    for q in ["q1_score", "q2_score", "q3_score", "q4_score", "q5_score"]:
        res_q = fit_mixedlm(
            f"{q} ~ C(training) + C(resources) + C(reasoning) + C(epi_context) + C(socioeconomic) + C(guidelines)",
            df
        )
        tab = coef_table(res_q, f"Q4_{q}")
        tab["outcome"] = q
        q_tables.append(tab)

        # Grab the training Pakistan term if present (depends on baseline coding)
        # In statsmodels with C(training), baseline is first category ("US" given our ordering)
        pk_term = "C(training)[T.Pakistan]"
        if pk_term in tab["term"].values:
            est = float(tab.loc[tab["term"] == pk_term, "estimate"].iloc[0])
            pv = float(tab.loc[tab["term"] == pk_term, "p_value"].iloc[0])
            pvals.append(pv)

            sd = df[q].std(ddof=1)
            std_eff = est / sd if sd and not np.isnan(sd) else np.nan
        else:
            est, pv, std_eff = np.nan, np.nan, np.nan
            pvals.append(np.nan)

        rows.append({"rubric": q, "Pakistan_vs_US_std_effect": std_eff})

    q4_all = pd.concat(q_tables, ignore_index=True)
    q4_all.to_csv(os.path.join(outdir, "Q4_mixedlm_coefficients_all_qs.csv"), index=False)

    # Multiple testing correction (for the 5 Pakistan-vs-US tests)
    # (If some missing, we handle it)
    pvals_clean = [p for p in pvals if not np.isnan(p)]
    if len(pvals_clean) > 0:
        reject, p_adj, _, _ = multipletests(pvals_clean, alpha=0.05, method="fdr_bh")
        # Map back
        adj_rows = []
        j = 0
        for q, pv in zip(["q1_score", "q2_score", "q3_score", "q4_score", "q5_score"], pvals):
            if np.isnan(pv):
                adj_rows.append({"rubric": q, "p_value": np.nan, "p_value_fdr_bh": np.nan, "reject_0.05": np.nan})
            else:
                adj_rows.append({"rubric": q, "p_value": pv, "p_value_fdr_bh": p_adj[j], "reject_0.05": reject[j]})
                j += 1
        pd.DataFrame(adj_rows).to_csv(os.path.join(outdir, "Q4_training_Pakistan_vs_US_pvalues_FDR.csv"), index=False)

    # Bar plots for each rubric score by training (easy to interpret)
    for q in ["q1_score", "q2_score", "q3_score", "q4_score", "q5_score"]:
        barplot_means_with_ci(
            df, "training", q,
            f"Q4: {q} by training (mean ± 95% CI)",
            os.path.join(outdir, f"Q4_{q}_by_training_bar.png")
        )

    # Heatmap: just 1 column (Pakistan vs US standardized effect) but still helpful
    heat_df = pd.DataFrame(rows).set_index("rubric")
    heatmap(
        heat_df,
        "Q4: Rubric sensitivity (Pakistan vs US standardized effect)",
        os.path.join(outdir, "Q4_rubric_sensitivity_heatmap.png")
    )

    # ---------------- Summary print ----------------
    summary_path = os.path.join(outdir, "README_results_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Outputs saved in this folder:\n")
        f.write("- data_with_dimensions.csv\n")
        f.write("- Q1_descriptives_by_training.csv, Q1_mixedlm_coefficients.csv, Q1_total_by_training_bar.png\n")
        f.write("- Q2_mixedlm_coefficients.csv, Q2_dimension_effects_forest.png\n")
        f.write("- Q3_mixedlm_training_x_run.csv, Q3_run_stability_lines.png, Q3_run_correlations.csv\n")
        f.write("- Q4_mixedlm_coefficients_all_qs.csv, Q4_training_Pakistan_vs_US_pvalues_FDR.csv\n")
        f.write("- Q4_qk_by_training_bar.png (for k=1..5), Q4_rubric_sensitivity_heatmap.png\n")
        f.write("\nKey models:\n")
        f.write("Q1: total_score ~ training + (random: persona_id) + vc(clinical_id, run)\n")
        f.write("Q2: total_score ~ training + resources + reasoning + epi + socioeconomic + guidelines + interactions\n")
        f.write("Q3: total_score ~ training * run + random persona + vc(clinical_id, run)\n")
        f.write("Q4: qk ~ training + dimensions + random persona + vc(clinical_id, run)\n")

    print(f"Done. Results saved to: {outdir}")
    print(f"See: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CSV with scores.")
    parser.add_argument("--outdir", default="results", help="Output directory.")
    args = parser.parse_args()

    main(args.input, args.outdir)
