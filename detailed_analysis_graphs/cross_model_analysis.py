import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# -----------------------------
# Persona dimension mapping
# -----------------------------
def add_persona_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    def training(pid: int) -> str:
        if 1 <= pid <= 9: return "US"
        if 10 <= pid <= 18: return "Pakistan"
        if 19 <= pid <= 26: return "Neutral"
        return "Unknown"

    advanced_ids = {2, 4, 11, 13, 20, 22, 26, 8, 18}
    limited_ids  = {5, 9, 14, 17, 23}
    step_ids     = {3, 4, 12, 13, 21, 22}
    epi_ids      = {6, 8, 9, 15, 17, 18, 24, 26}
    socio_ids    = {7, 8, 9, 16, 17, 18, 25, 26}
    guide_ids    = {8, 9, 17, 18, 26}

    def location_specific(pid: int) -> str:
        if pid in {6, 8, 9}: return "US"
        if pid in {15, 17, 18}: return "Pakistan"
        if pid == 24: return "Global"
        if pid == 19: return "Unknown"
        return "None"

    df = df.copy()
    df["training"] = df["persona_id"].astype(int).apply(training)

    def resources(pid: int) -> str:
        if pid in advanced_ids: return "Advanced"
        if pid in limited_ids:  return "Limited"
        return "Unspecified"

    df["resources"] = df["persona_id"].astype(int).apply(resources)
    df["reasoning"] = np.where(df["persona_id"].isin(step_ids), "StepByStep", "NoStep")
    df["epi_context"] = np.where(df["persona_id"].isin(epi_ids), "Yes", "No")
    df["socioeconomic"] = np.where(df["persona_id"].isin(socio_ids), "Yes", "No")
    df["guidelines"] = np.where(df["persona_id"].isin(guide_ids), "Yes", "No")
    df["location_specific"] = df["persona_id"].astype(int).apply(location_specific)

    df["training"] = pd.Categorical(df["training"], categories=["US","Neutral","Pakistan"], ordered=True)
    df["resources"] = pd.Categorical(df["resources"], categories=["Unspecified","Advanced","Limited"])
    return df


# -----------------------------
# Robust MixedLM
# -----------------------------
def fit_mixedlm(formula: str, df: pd.DataFrame):
    outcome = formula.split("~")[0].strip()

    must_have = [outcome, "persona_id", "clinical_id", "run", "model_name", "training"]
    must_have = [c for c in must_have if c in df.columns]
    df2 = df.dropna(subset=must_have).copy().reset_index(drop=True)

    # Random intercept by persona_id; add variance components for clinical_id and run
    vc = {"clinical_id": "0 + C(clinical_id)", "run": "0 + C(run)"}

    # NOTE: We keep persona_id as grouping; model_name stays fixed-effect
    model = smf.mixedlm(formula, df2, groups=df2["persona_id"], vc_formula=vc, re_formula="1")

    for method in ["lbfgs", "powell", "nm"]:
        try:
            return model.fit(method=method, reml=False, maxiter=500, disp=False)
        except Exception:
            pass
    return model.fit(method="lbfgs", reml=False, maxiter=500, disp=False)


def coef_table(res, label: str) -> pd.DataFrame:
    params = res.params
    bse = res.bse
    out = pd.DataFrame({"term": params.index, "estimate": params.values, "std_error": bse.values})
    out["ci_low"] = out["estimate"] - 1.96*out["std_error"]
    out["ci_high"] = out["estimate"] + 1.96*out["std_error"]
    out["p_value"] = res.pvalues.values
    out["model"] = label
    return out


# -----------------------------
# Plot helpers
# -----------------------------
def mean_ci(x: pd.Series):
    x = x.dropna()
    n = len(x)
    m = x.mean()
    s = x.std(ddof=1)
    if n <= 1 or np.isnan(s):
        return m, np.nan, np.nan
    se = s/np.sqrt(n)
    ci = 1.96*se
    return m, m-ci, m+ci


def barplot_group(df, xcol, ycol, title, outpath, by=None):
    """
    Barplot with 95% CI **with value labels inside bars**.
    If by is provided, x becomes (by,xcol).
    """
    if by is None:
        g = (df.groupby(xcol, observed=True)[ycol]
             .apply(lambda s: pd.Series(mean_ci(s), index=["mean","lo","hi"]))
             .unstack().reset_index())

        plt.figure(figsize=(8,4))
        bars = plt.bar(g[xcol].astype(str), g["mean"], color="skyblue")

        yerr = np.vstack([g["mean"]-g["lo"], g["hi"]-g["mean"]])
        plt.errorbar(g[xcol].astype(str), g["mean"], yerr=yerr, fmt="none", capsize=4, color="black")

        # -------- ADD LABELS INSIDE BAR --------
        for bar, value in zip(bars, g["mean"]):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height()/2,
                f"{value:.1f}",
                ha="center", va="center", fontsize=10, color="black", fontweight="bold"
            )

        plt.title(title)
        plt.ylabel(ycol)
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()

    else:
        g = (df.groupby([by, xcol], observed=True)[ycol]
             .apply(lambda s: pd.Series(mean_ci(s), index=["mean","lo","hi"]))
             .unstack().reset_index())

        models = g[by].unique().tolist()
        levels = g[xcol].astype(str).unique().tolist()

        plt.figure(figsize=(10,5))
        width = 0.8/len(models) if len(models) else 0.2
        xs = np.arange(len(levels))

        for i, m in enumerate(models):
            sub = g[g[by]==m].set_index(xcol).reindex(levels).reset_index()
            bars = plt.bar(xs + i*width, sub["mean"], width=width, label=str(m))

            # -------- LABELS INSIDE GROUPED BARS --------
            for bar, value in zip(bars, sub["mean"]):
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height()/2,
                    f"{value:.1f}",
                    ha="center", va="center", fontsize=9, color="black", fontweight="bold"
                )

        plt.xticks(xs + (len(models)-1)*width/2, levels)
        plt.title(title)
        plt.ylabel(ycol)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
def plot_model_training_scores(df, outpath):
    """
    One graph:
    X-axis = model
    Bars = US / Neutral / Pakistan
    Y = mean total_score
    """

    # Explicit pivot guarantees alignment
    pivot = (
        df.groupby(["model_name", "training"], observed=True)["total_score"]
          .mean()
          .unstack("training")
          .reindex(columns=["US", "Neutral", "Pakistan"])
    )

    models = pivot.index.tolist()
    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(12,5))

    for i, tr in enumerate(["US", "Neutral", "Pakistan"]):
        y = pivot[tr].values
        bars = plt.bar(x + i*width, y, width, label=tr)

        # Labels inside bars
        for bar, value in zip(bars, y):
            if not np.isnan(value):
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height()/2,
                    f"{value:.1f}",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold"
                )

    plt.xticks(x + width, models)
    plt.ylabel("Mean Total Diagnostic Score")
    plt.title("Model Performance by Persona Training Background")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_pooled_prompt_effects(df, outpath):
    """
    Figure 2 (Q2):
    Pooled mean effects of prompt dimensions on total_score
    """

    dims = {
        "Advanced resources": df[df["resources"] == "Advanced"]["total_score"],
        "Limited resources": df[df["resources"] == "Limited"]["total_score"],
        "Step-by-step reasoning": df[df["reasoning"] == "StepByStep"]["total_score"],
        "Guidelines included": df[df["guidelines"] == "Yes"]["total_score"],
        "Epidemiologic context": df[df["epi_context"] == "Yes"]["total_score"],
        "Socioeconomic context": df[df["socioeconomic"] == "Yes"]["total_score"],
    }

    rows = []
    for k, v in dims.items():
        m, lo, hi = mean_ci(v)
        rows.append((k, m, lo, hi))

    eff = pd.DataFrame(rows, columns=["dimension","mean","lo","hi"])
    eff.to_csv(outpath.replace(".png", ".csv"), index=False)

    # Okabe–Ito palette
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]

    plt.figure(figsize=(9,4))
    bars = plt.bar(
        eff["dimension"],
        eff["mean"],
        color=colors,
        edgecolor="black",
        linewidth=0.8
    )

    yerr = np.vstack([eff["mean"] - eff["lo"], eff["hi"] - eff["mean"]])
    plt.errorbar(
        eff["dimension"],
        eff["mean"],
        yerr=yerr,
        fmt="none",
        capsize=4,
        color="black",
        linewidth=1
    )

    for bar, val in zip(bars, eff["mean"]):
        plt.text(
            bar.get_x()+bar.get_width()/2,
            bar.get_height()/2,
            f"{val:.2f}",
            ha="center", va="center",
            fontsize=9, fontweight="bold"
        )

    plt.ylabel("Mean Total Diagnostic Score")
    plt.title("Pooled Effects of Prompt Dimensions (All Models)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_average_run_stability(df, outpath):
    """
    Figure 3 (Q3):
    Average run-to-run variability per model
    """

    rows = []
    for mn in df["model_name"].unique():
        sub = df[df["model_name"] == mn]
        run_means = sub.groupby(["run","persona_id"])["total_score"].mean().reset_index()
        sd = run_means.groupby("persona_id")["total_score"].std().mean()
        rows.append((mn, sd))

    stab = pd.DataFrame(rows, columns=["model","avg_run_sd"])
    stab.to_csv(outpath.replace(".png", ".csv"), index=False)

    plt.figure(figsize=(8,4))
    bars = plt.bar(
        stab["model"],
        stab["avg_run_sd"],
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.8
    )

    for bar, val in zip(bars, stab["avg_run_sd"]):
        plt.text(
            bar.get_x()+bar.get_width()/2,
            bar.get_height()/2,
            f"{val:.2f}",
            ha="center", va="center",
            fontsize=9, fontweight="bold"
        )

    plt.ylabel("Average SD Across Runs")
    plt.title("Average Run-to-Run Stability by Model")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

# -----------------------------
# Load & normalize
# -----------------------------
def load_one(model_name: str, path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    rename_map = {"Clinical_id":"clinical_id", "Persona_id":"persona_id", "Run":"run"}
    df = df.rename(columns=rename_map)
    df.columns = [c.lower() for c in df.columns]

    req = ["clinical_id","persona_id","run","q1_score","q2_score","q3_score","q4_score","q5_score"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"{model_name}: Missing columns {missing}. Present columns: {list(df.columns)}")

    for c in req:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "total_score" not in df.columns:
        df["total_score"] = df[["q1_score","q2_score","q3_score","q4_score","q5_score"]].sum(axis=1)

    df["model_name"] = model_name
    df = add_persona_dimensions(df)
    return df


# -----------------------------
# Main
# -----------------------------
def main(files, outdir):
    os.makedirs(outdir, exist_ok=True)

    # Load all and stack
    dfs = []
    for item in files:
        model_name, path = item.split("=", 1)
        dfs.append(load_one(model_name.strip(), path.strip()))
    all_df = pd.concat(dfs, ignore_index=True)

    all_df.to_csv(os.path.join(outdir, "all_models_stacked.csv"), index=False)

    # -----------------------------
    # A) Descriptives (overall + by training per model)
    # -----------------------------
    overall = (all_df.groupby("model_name", observed=True)["total_score"]
               .agg(["count","mean","std"]).reset_index())
    overall.to_csv(os.path.join(outdir, "A_overall_by_model.csv"), index=False)

    by_tr = (all_df.groupby(["model_name","training"], observed=True)["total_score"]
             .agg(["count","mean","std"]).reset_index())
    by_tr.to_csv(os.path.join(outdir, "A_total_by_model_and_training.csv"), index=False)

    barplot_group(all_df, "model_name", "total_score",
                  "Overall total_score by model (mean ± 95% CI)",
                  os.path.join(outdir, "A_overall_total_by_model.png"))

    barplot_group(all_df, "training", "total_score",
                  "total_score by training, separated by model",
                  os.path.join(outdir, "A_training_by_model_bars.png"),
                  by="model_name")

        # Cross-model training persona comparison (MAIN FIGURE FOR Q1)
    plot_model_training_scores(
        all_df,
        os.path.join(outdir, "A_onegraph_model_training.png")
    )

    # -----------------------------
    # B) Key inferential: WHICH MODEL HAS BIGGEST Pakistan vs US drop?
    # Mixed model with interaction: total ~ model * training
    # -----------------------------
    m_main = fit_mixedlm("total_score ~ C(model_name) * C(training)", all_df)
    main_tab = coef_table(m_main, "B_total_model_x_training")
    main_tab.to_csv(os.path.join(outdir, "B_mixedlm_total_model_x_training.csv"), index=False)

    # Compute "Pakistan vs US" marginal drop per model using predicted means
    # (Simple approach: use group means as an interpretable summary)
    drops = (all_df.groupby(["model_name","training"], observed=True)["total_score"]
             .mean().unstack())
    drops["Pakistan_minus_US"] = drops["Pakistan"] - drops["US"]
    drops["Neutral_minus_US"] = drops["Neutral"] - drops["US"]
    drops = drops.reset_index()
    drops.to_csv(os.path.join(outdir, "B_drop_summary_means.csv"), index=False)

    # Plot drops
    plt.figure(figsize=(9,4))
    plt.bar(drops["model_name"].astype(str), drops["Pakistan_minus_US"])
    plt.axhline(0, linestyle="--")
    plt.title("Pakistan vs US drop (mean total_score difference) by model")
    plt.ylabel("Pakistan - US (points)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "B_pakistan_minus_us_drop_by_model.png"), dpi=200)
    plt.close()

        # -----------------------------
    # Figure 2 (Q2): Pooled prompt effects
    # -----------------------------
    plot_pooled_prompt_effects(
        all_df,
        os.path.join(outdir, "Figure2_pooled_prompt_effects.png")
    )


    # -----------------------------
    # C) Persona-level comparison across models (which persona is worst/best per model)
    # -----------------------------
    persona_means = (all_df.groupby(["model_name","persona_id"], observed=True)["total_score"]
                     .mean().reset_index())
    persona_means.to_csv(os.path.join(outdir, "C_persona_means_by_model.csv"), index=False)

    # Optional plot: top 10 worst personas per model (by mean)
    # (Saves one plot per model)
    for mn in all_df["model_name"].unique():
        sub = persona_means[persona_means["model_name"]==mn].sort_values("total_score").head(10)
        plt.figure(figsize=(8,4))
        plt.bar(sub["persona_id"].astype(str), sub["total_score"])
        plt.title(f"Bottom-10 personas by mean total_score ({mn})")
        plt.xlabel("persona_id"); plt.ylabel("mean total_score")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"C_bottom10_personas_{mn}.png"), dpi=200)
        plt.close()
    # -----------------------------
    # Figure 3 (Q3): Average run stability
    # -----------------------------
    plot_average_run_stability(
        all_df,
        os.path.join(outdir, "Figure3_average_run_stability.png")
    )

    # -----------------------------
    # D) Rubric sensitivity across models (q1..q5): model * training
    # -----------------------------
    rubric_results = []
    pvals = []
    for q in ["q1_score","q2_score","q3_score","q4_score","q5_score"]:
        res = fit_mixedlm(f"{q} ~ C(model_name) * C(training)", all_df)
        tab = coef_table(res, f"D_{q}_model_x_training")
        tab["outcome"] = q
        rubric_results.append(tab)

        # collect Pakistan interaction-ish p-values is messy across coding,
        # so instead we do a clean summary via mean-difference per model per q:
        # (saved below)
    rubric_all = pd.concat(rubric_results, ignore_index=True)
    rubric_all.to_csv(os.path.join(outdir, "D_mixedlm_rubric_model_x_training_all.csv"), index=False)

    # Mean-based rubric drop summaries (very interpretable)
    rows = []
    for q in ["q1_score","q2_score","q3_score","q4_score","q5_score"]:
        tmp = all_df.groupby(["model_name","training"], observed=True)[q].mean().unstack()
        tmp["Pakistan_minus_US"] = tmp["Pakistan"] - tmp["US"]
        tmp["Neutral_minus_US"] = tmp["Neutral"] - tmp["US"]
        tmp = tmp.reset_index()
        tmp["rubric"] = q
        rows.append(tmp)
    rubric_drop = pd.concat(rows, ignore_index=True)
    rubric_drop.to_csv(os.path.join(outdir, "D_rubric_drop_summary_means.csv"), index=False)

    #heatmap
    pivot = rubric_drop.pivot(
        index="rubric",
        columns="model_name",
        values="Pakistan_minus_US"
    )

    plt.figure(figsize=(10,4))

    vmax = np.nanmax(np.abs(pivot.values))

    im = plt.imshow(
        pivot.values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax
    )

    plt.colorbar(im, label="Pakistan − US (points)")
    plt.yticks(range(pivot.shape[0]), pivot.index.tolist())
    plt.xticks(
        range(pivot.shape[1]),
        pivot.columns.tolist(),
        rotation=45,
        ha="right"
    )

    plt.title("Rubric-Level Sensitivity to Training Background")
    plt.tight_layout()
    plt.savefig(
        os.path.join(outdir, "Figure4_rubric_sensitivity_heatmap.png"),
        dpi=300
    )
    plt.close()


    # -----------------------------
    # E) Run stability across models
    # -----------------------------
    run_means = (all_df.groupby(["model_name","run","training"], observed=True)["total_score"]
                 .mean().reset_index())
    run_means.to_csv(os.path.join(outdir, "E_run_means_by_model_training.csv"), index=False)

    # Plot run stability: one plot per model
    for mn in all_df["model_name"].unique():
        plt.figure(figsize=(7,4))
        sub = all_df[all_df["model_name"]==mn]
        for tr in ["US","Neutral","Pakistan"]:
            s2 = sub[sub["training"]==tr]
            pts = []
            for r in sorted(s2["run"].unique()):
                m, lo, hi = mean_ci(s2[s2["run"]==r]["total_score"])
                pts.append((r,m,lo,hi))
            pts = pd.DataFrame(pts, columns=["run","mean","lo","hi"])
            plt.plot(pts["run"], pts["mean"], marker="o", label=tr)
            plt.fill_between(pts["run"], pts["lo"], pts["hi"], alpha=0.2)
        plt.title(f"Run stability by training ({mn})")
        plt.xlabel("run"); plt.ylabel("total_score")
        plt.xticks(sorted(sub["run"].unique()))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"E_run_stability_{mn}.png"), dpi=200)
        plt.close()

        

    # README summary
    with open(os.path.join(outdir, "README_cross_model.txt"), "w", encoding="utf-8") as f:
        f.write("Cross-model outputs:\n")
        f.write("- all_models_stacked.csv\n")
        f.write("- A_overall_by_model.csv, A_total_by_model_and_training.csv\n")
        f.write("- A_overall_total_by_model.png, A_training_by_model_bars.png\n")
        f.write("- B_mixedlm_total_model_x_training.csv\n")
        f.write("- B_drop_summary_means.csv, B_pakistan_minus_us_drop_by_model.png\n")
        f.write("- C_persona_means_by_model.csv + C_bottom10_personas_<model>.png\n")
        f.write("- D_mixedlm_rubric_model_x_training_all.csv\n")
        f.write("- D_rubric_drop_summary_means.csv, D_heatmap_pakistan_minus_us_by_model_and_rubric.png\n")
        f.write("- E_run_means_by_model_training.csv + E_run_stability_<model>.png\n")

    print(f"Done. Results saved to: {outdir}")

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="cross_model_results", help="Output folder")
    parser.add_argument("--file", action="append", required=True,
                        help='ModelName=path.csv (repeat this flag per model)')
    args = parser.parse_args()

    main(args.file, args.outdir)
