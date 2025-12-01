"""
RQ3 Visualization Script - MONA Output Files

Generates RQ3 visualizations from the MONA output files:
- rq3_entropy_analysis.csv
- rq3_descriptive_stats.csv
- rq3_correlation_matrix.csv
- rq3_regression_models.csv
- rq3_entropy_model_residuals.csv

Creates:
1. Histograms for entropy and balanced accuracy
2. Scatter plot: balanced_accuracy vs entropy
3. Scatter plot: balanced_accuracy vs majority_prop
4. Scatter plot: balanced_accuracy vs log(n_test) 
5. Residual plots from entropy-only regression (outlier municipalities)
6. Correlation heatmap 

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# ==============================================================================
# MAP KOMMUN CODES TO NAMES
# ==============================================================================

KOMMUN_LOOKUP_FILE = Path("data") / "municipality_code.csv"


def load_kommun_lookup():
    """
    Load kommun code/name file and return a dict {code4: name}.
    Expects columns: 'Code', 'Name'.
    """
    df_k = pd.read_csv(KOMMUN_LOOKUP_FILE)  # add sep="\t" if it's tab-separated
    # make sure codes are 4-digit strings (e.g. 0114)
    df_k["Code"] = df_k["Code"].astype(str).str.strip().str.zfill(4)
    return dict(zip(df_k["Code"], df_k["Name"]))


# ==============================================================================
# CONFIGURATION
# ==============================================================================

RESULTS_DIR = Path("outputs_mona")

# Input CSVs (from mona_rq3.py)
TABLES_DIR = RESULTS_DIR / "tables"
ENTROPY_CSV = TABLES_DIR / "rq3_entropy_analysis.csv"
DESC_CSV = TABLES_DIR / "rq3_descriptive_stats.csv"
CORR_CSV = TABLES_DIR / "rq3_correlation_matrix.csv"
REG_CSV = TABLES_DIR / "rq3_regression_models.csv"
RESID_CSV = TABLES_DIR / "rq3_entropy_model_residuals.csv"

# Output plots
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

HIST_ENTROPY_PLOT = PLOTS_DIR / "rq3_entropy_histogram.png"
HIST_BALACC_PLOT = PLOTS_DIR / "rq3_balanced_accuracy_histogram.png"
SCATTER_ENTROPY_PLOT = PLOTS_DIR / "rq3_balacc_vs_entropy.png"
SCATTER_MAJPROP_PLOT = PLOTS_DIR / "rq3_balacc_vs_majority_prop.png"
SCATTER_LOGNTEST_PLOT = PLOTS_DIR / "rq3_balacc_vs_logn_test.png"
RESID_VS_ENTROPY_PLOT = PLOTS_DIR / "rq3_residuals_vs_entropy.png"
RESID_TOP_BOTTOM_PLOT = PLOTS_DIR / "rq3_residuals_top_bottom.png"
CORR_HEATMAP_PLOT = PLOTS_DIR / "rq3_correlation_heatmap.png"

# Styling
sns.set(style="whitegrid")
DPI = 300
FIGSIZE_STANDARD = (10, 6)

# Color palette
COLORS = {
    "entropy": "#2E86AB",
    "balanced_accuracy": "#A23B72",
    "majority_prop": "#F6AE2D",
    "n_test": "#33673B",
    "residuals": "#555555",
}

# Number of top/bottom outliers to show in barplot
N_OUTLIERS = 10


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def load_required_data():
    """Load the core RQ3 tables and perform basic checks."""
    print("=" * 80)
    print("RQ3 VISUALIZATION: LOADING DATA")
    print("=" * 80)

    if not ENTROPY_CSV.exists():
        raise FileNotFoundError(
            f"Entropy analysis file not found: {ENTROPY_CSV}\n"
            "Please run mona_rq3.py first."
        )

    df_entropy = pd.read_csv(ENTROPY_CSV)
    print(f"Loaded entropy analysis: {len(df_entropy)} municipalities")

    df_desc = pd.read_csv(DESC_CSV) if DESC_CSV.exists() else None
    if df_desc is not None:
        print(f"Loaded descriptive stats: {len(df_desc)} rows")

    df_corr = pd.read_csv(CORR_CSV, index_col=0) if CORR_CSV.exists() else None
    if df_corr is not None:
        print("Loaded correlation matrix")

    df_reg = pd.read_csv(REG_CSV) if REG_CSV.exists() else None
    if df_reg is not None:
        print(f"Loaded regression models: {len(df_reg)} models")

    df_resid = pd.read_csv(RESID_CSV) if RESID_CSV.exists() else None
    if df_resid is not None:
        print(f"Loaded residuals file: {len(df_resid)} municipalities")

    print("\nColumns in entropy analysis CSV:")
    print(list(df_entropy.columns))

    return df_entropy, df_desc, df_corr, df_reg, df_resid


# ==============================================================================
# PLOT 1: HISTOGRAMS
# ==============================================================================


def plot_entropy_histogram(df):
    """Histogram of Shannon entropy across municipalities."""
    print("\n1. Creating entropy histogram...")

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    sns.histplot(
        data=df,
        x="entropy",
        bins=30,
        kde=True,
        color=COLORS["entropy"],
        ax=ax,
    )

    ax.set_xlabel("Shannon Entropy (Y distribution)", fontsize=12)
    ax.set_ylabel("Number of Municipalities", fontsize=12)
    ax.set_title(
        "Distribution of Outcome Diversity Across Municipalities",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(HIST_ENTROPY_PLOT, dpi=DPI, bbox_inches="tight")
    print(f"   Saved: {HIST_ENTROPY_PLOT}")
    plt.close()


def plot_balacc_histogram(df):
    """Histogram of balanced accuracy across municipalities."""
    print("\n2. Creating balanced accuracy histogram...")

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    sns.histplot(
        data=df,
        x="balanced_accuracy",
        bins=30,
        kde=True,
        color=COLORS["balanced_accuracy"],
        ax=ax,
    )

    ax.set_xlabel("Balanced Accuracy", fontsize=12)
    ax.set_ylabel("Number of Municipalities", fontsize=12)
    ax.set_title(
        "Distribution of Predictive Performance Across Municipalities",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(HIST_BALACC_PLOT, dpi=DPI, bbox_inches="tight")
    print(f"   Saved: {HIST_BALACC_PLOT}")
    plt.close()


# ==============================================================================
# PLOT 2: SCATTER PLOTS
# ==============================================================================

def plot_balacc_vs_entropy(df, df_reg=None):
    """
    Scatter: balanced_accuracy vs entropy with regression line and 95% CI.

    - Point size reflects test set size (n_test), if available.
    - Regression line and 95% CI taken from rq3_regression_models.csv
      (model with predictors == 'entropy').
    - R², slope, and intercept are shown in the legend entry for the regression line
      (similar style as plot_balacc_vs_logn_test).
    """
    print("\n3. Creating scatter: balanced_accuracy vs entropy...")

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    # ------------------------------------------------------------------
    # Data and point sizes
    # ------------------------------------------------------------------
    if "n_test" in df.columns:
        df_plot = df[df["n_test"] > 0].copy()
        if df_plot.empty:
            print("   No valid rows with n_test > 0. Skipping plot.")
            return
        # rescale sizes so they don't get huge
        max_ref = df_plot["n_test"].quantile(0.95)
        max_ref = max_ref if max_ref > 0 else df_plot["n_test"].max()
        sizes = 20 + 80 * (df_plot["n_test"] / max_ref).clip(0, 1)
    else:
        df_plot = df.copy()
        if df_plot.empty:
            print("   No valid rows. Skipping plot.")
            return
        sizes = 40

    df_plot = df_plot.dropna(subset=["entropy", "balanced_accuracy"])
    if df_plot.empty:
        print("   No valid rows after dropping NA in entropy/balanced_accuracy.")
        return

    # ------------------------------------------------------------------
    # Scatter: municipalities
    # ------------------------------------------------------------------
    scatter = ax.scatter(
        df_plot["entropy"],
        df_plot["balanced_accuracy"],
        s=sizes,
        alpha=0.7,
        color=COLORS.get("entropy", "#2E86AB"),
        edgecolor="white",
        linewidth=0.5,
        label="Municipalities",
    )

    # ------------------------------------------------------------------
    # Regression line + 95% CI from regression_models.csv if available
    # ------------------------------------------------------------------
    reg_line = None
    r2 = slope = intercept = None

    if df_reg is not None:
        reg = df_reg.copy()

        # Prefer the entropy-only model (predictors == "entropy")
        mask = (
            (reg["dependent_var"] == "balanced_accuracy") &
            reg["predictors"].str.fullmatch(r"entropy", case=False)
        )
        if not mask.any():
            # Fallback: anything that contains "entropy" but not log(n_test)
            mask = (
                (reg["dependent_var"] == "balanced_accuracy") &
                reg["predictors"].str.contains("entropy", case=False, na=False) &
                ~reg["predictors"].str.contains("log", case=False, na=False)
            )

        if mask.any():
            row = reg[mask].iloc[0]

            intercept = row.get("intercept", np.nan)
            slope = row.get("coef_entropy", np.nan)
            r2 = row.get("r2", np.nan)
            ci_low = row.get("ci_entropy_low", np.nan)
            ci_high = row.get("ci_entropy_high", np.nan)

            if np.isfinite(intercept) and np.isfinite(slope):
                x_vals = np.linspace(
                    df_plot["entropy"].min(),
                    df_plot["entropy"].max(),
                    200,
                )
                y_vals = intercept + slope * x_vals

                reg_label = "Regression"
                if np.isfinite(r2):
                    reg_label = (
                        rf"Regression "
                        rf"($R^2$={r2:.3f}, slope={slope:.3f}, intercept={intercept:.3f})"
                    )

                (reg_line,) = ax.plot(
                    x_vals,
                    y_vals,
                    linestyle="--",
                    linewidth=2,
                    color="black",
                    label=reg_label,
                )

                # 95% CI band if CI for slope is available
                if np.isfinite(ci_low) and np.isfinite(ci_high):
                    y_low = intercept + ci_low * x_vals
                    y_high = intercept + ci_high * x_vals
                    ax.fill_between(
                        x_vals,
                        y_low,
                        y_high,
                        alpha=0.15,
                        color=COLORS.get("entropy", "#2E86AB"),
                    )

    # ------------------------------------------------------------------
    # Fallback: simple OLS if no regression row is found
    # (still shows R², slope, intercept in legend; but no CI)
    # ------------------------------------------------------------------
    if reg_line is None and len(df_plot) >= 2:
        x = df_plot["entropy"].values
        y = df_plot["balanced_accuracy"].values

        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = intercept + slope * x_line

        # R²
        y_hat = intercept + slope * x
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        reg_label = "Regression"
        if np.isfinite(r2):
            reg_label = (
                rf"Regression "
                rf"($R^2$={r2:.3f}, slope={slope:.3f}, intercept={intercept:.3f})"
            )

        ax.plot(
            x_line,
            y_line,
            linestyle="--",
            linewidth=2,
            color="black",
            label=reg_label,
        )

    # ------------------------------------------------------------------
    # Size legend for n_test (quantiles), if available
    # ------------------------------------------------------------------
    if "n_test" in df.columns:
        for q in [0.25, 0.5, 0.75]:
            n = df_plot["n_test"].quantile(q)
            size_q = 20 + 80 * (n / max_ref) if max_ref > 0 else 40
            ax.scatter(
                [],
                [],
                s=size_q,
                color=COLORS.get("entropy", "#2E86AB"),
                alpha=0.7,
                edgecolor="white",
                linewidth=0.5,
                label=f"n_test ≈ {int(round(n))}",
            )

    # ------------------------------------------------------------------
    # Axes, title, legend
    # ------------------------------------------------------------------
    ax.set_xlabel("Shannon entropy (Y distribution)", fontsize=12)
    ax.set_ylabel("Balanced accuracy", fontsize=12)
    ax.set_title(
        "Predictive Performance vs Outcome Diversity",
        fontsize=14,
        fontweight="bold",
    )

    ax.grid(True, alpha=0.3)

    # Legend: includes municipalities (points), size quantiles, regression+stats
    ax.legend(fontsize=10, loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(SCATTER_ENTROPY_PLOT, dpi=DPI, bbox_inches="tight")
    print(f"   Saved: {SCATTER_ENTROPY_PLOT}")
    plt.close()


def plot_balacc_vs_majority_prop(df):
    """Scatter: balanced_accuracy vs majority_prop."""
    print("\n4. Creating scatter: balanced_accuracy vs majority_prop...")

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    ax.scatter(
        df["majority_prop"],
        df["balanced_accuracy"],
        s=40,
        alpha=0.7,
        color=COLORS["majority_prop"],
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xlabel("Majority Class Proportion", fontsize=12)
    ax.set_ylabel("Balanced Accuracy", fontsize=12)
    ax.set_title(
        "Predictive Performance vs Class Imbalance",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SCATTER_MAJPROP_PLOT, dpi=DPI, bbox_inches="tight")
    print(f"   Saved: {SCATTER_MAJPROP_PLOT}")
    plt.close()


def plot_balacc_vs_logn_test(df, df_reg=None):
    """
    Scatter: balanced_accuracy vs log(n_test) with regression line and 95% CI.

    - Uses municipality-level test set size (n_test).
    - Regression line and 95% CI taken from rq3_regression_models.csv
      (model with predictors == 'log(n_test)').
    - R², slope, and intercept are shown in the legend entry for the regression line.
    """
    if "n_test" not in df.columns:
        print("\n5. No n_test column found. Skipping log(n_test) plot.")
        return

    print("\n5. Creating scatter: balanced_accuracy vs log(n_test)...")

    # Keep only positive test sizes and compute log
    df_plot = df[df["n_test"] > 0].copy()
    df_plot["log_n_test"] = np.log(df_plot["n_test"])

    # Drop rows with missing balanced_accuracy
    df_plot = df_plot.dropna(subset=["log_n_test", "balanced_accuracy"])

    if df_plot.empty:
        print("   No valid rows after filtering n_test > 0 and dropping NA. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    # ------------------------------------------------------------------
    # Scatter
    # ------------------------------------------------------------------
    scatter = ax.scatter(
        df_plot["log_n_test"],
        df_plot["balanced_accuracy"],
        s=40,
        alpha=0.7,
        color=COLORS.get("n_test", "#33673B"),
        edgecolor="white",
        linewidth=0.5,
        label="Municipalities",
    )

    # ------------------------------------------------------------------
    # Regression line + 95% CI from regression_models.csv if available
    # ------------------------------------------------------------------
    r2 = slope = intercept = None
    reg_line = None

    if df_reg is not None:
        reg = df_reg.copy()

        # Prefer the log(n_test)-only model
        mask = (
            (reg["dependent_var"] == "balanced_accuracy") &
            reg["predictors"].str.fullmatch(r"log\(n_test\)", case=False)
        )
        if not mask.any():
            # Fallback: any model that has "log(n_test)" as predictor
            mask = (
                (reg["dependent_var"] == "balanced_accuracy") &
                reg["predictors"].str.contains("log(n_test)", case=False, na=False) &
                ~reg["predictors"].str.contains("\+", na=False)   # avoid entropy + log(n_test)
            )

        if mask.any():
            row = reg[mask].iloc[0]

            intercept = row.get("intercept", np.nan)
            slope = row.get("coef_log_n_test", np.nan)
            r2 = row.get("r2", np.nan)
            ci_low = row.get("ci_log_low", np.nan)
            ci_high = row.get("ci_log_high", np.nan)

            if np.isfinite(intercept) and np.isfinite(slope):
                x_vals = np.linspace(
                    df_plot["log_n_test"].min(),
                    df_plot["log_n_test"].max(),
                    200,
                )
                y_vals = intercept + slope * x_vals

                # Regression line
                reg_label = "Regression"
                if np.isfinite(r2):
                    reg_label = (
                        rf"Regression "
                        rf"($R^2$={r2:.3f}, slope={slope:.3f}, intercept={intercept:.3f})"
                    )

                reg_line, = ax.plot(
                    x_vals,
                    y_vals,
                    linestyle="--",
                    linewidth=2,
                    color="black",
                    label=reg_label,
                )

                # 95% CI band if CI for slope is available
                if np.isfinite(ci_low) and np.isfinite(ci_high):
                    y_low = intercept + ci_low * x_vals
                    y_high = intercept + ci_high * x_vals
                    ax.fill_between(
                        x_vals,
                        y_low,
                        y_high,
                        alpha=0.15,
                        color=COLORS.get("n_test", "#33673B"),
                    )

    # ------------------------------------------------------------------
    # Fallback: simple OLS if no regression row is found
    # (still shows R², slope, intercept in legend; but no CI)
    # ------------------------------------------------------------------
    if reg_line is None and len(df_plot) >= 2:
        x = df_plot["log_n_test"].values
        y = df_plot["balanced_accuracy"].values

        # Simple linear regression
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = intercept + slope * x_line

        # R²
        y_hat = intercept + slope * x
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        reg_label = "Regression"
        if np.isfinite(r2):
            reg_label = (
                rf"Regression "
                rf"($R^2$={r2:.3f}, slope={slope:.3f}, intercept={intercept:.3f})"
            )

        ax.plot(
            x_line,
            y_line,
            linestyle="--",
            linewidth=2,
            color="black",
            label=reg_label,
        )

    # ------------------------------------------------------------------
    # Axes, title, legend (with R² / slope / intercept in regression label)
    # ------------------------------------------------------------------
    ax.set_xlabel("log(Test set size)", fontsize=12)
    ax.set_ylabel("Balanced Accuracy", fontsize=12)
    ax.set_title(
        "Predictive Performance vs Municipality Test Size",
        fontsize=14,
        fontweight="bold",
    )

    ax.grid(True, alpha=0.3)

    # Legend: includes points + regression line + its stats
    ax.legend(fontsize=10, loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(SCATTER_LOGNTEST_PLOT, dpi=DPI, bbox_inches="tight")
    print(f"   Saved: {SCATTER_LOGNTEST_PLOT}")
    plt.close()


# ==============================================================================
# PLOT 3: RESIDUALS AND OUTLIERS
# ==============================================================================


def plot_residuals_vs_entropy(df_resid):
    """
    Scatter plot of residuals from entropy-only model vs entropy.
    Residuals are stored in 'residual_entropy_model'.
    """
    if df_resid is None or "residual_entropy_model" not in df_resid.columns:
        print("\n6. Residuals file not available or missing column. Skipping residual plot.")
        return

    print("\n6. Creating residuals vs entropy plot...")

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)

    ax.scatter(
        df_resid["entropy"],
        df_resid["residual_entropy_model"],
        s=40,
        alpha=0.7,
        color=COLORS["residuals"],
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xlabel("Shannon Entropy (Y distribution)", fontsize=12)
    ax.set_ylabel("Residual (Balanced Accuracy - Fitted)", fontsize=12)
    ax.set_title(
        "Residuals from Entropy-Only Model",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESID_VS_ENTROPY_PLOT, dpi=DPI, bbox_inches="tight")
    print(f"   Saved: {RESID_VS_ENTROPY_PLOT}")
    plt.close()


def plot_residuals_top_bottom(df_resid, n_outliers=N_OUTLIERS):
    """
    Barplot of top and bottom municipalities by residual (over-/under-performers),
    using municipality names (kommun_name) if available.
    """
    if df_resid is None or "residual_entropy_model" not in df_resid.columns:
        print("\n7. Residuals file not available or missing column. Skipping outlier barplot.")
        return

    print("\n7. Creating residuals top/bottom barplot...")

    # Prefer kommun_name if already attached in main(); otherwise fall back to a code column
    label_col = None
    if "kommun_name" in df_resid.columns:
        label_col = "kommun_name"
    else:
        for cand in ["Kommun", "kommun", "municipality", "mun_code"]:
            if cand in df_resid.columns:
                label_col = cand
                break

    if label_col is None:
        raise ValueError("No municipality name or code column found in residuals dataframe.")

    # Get top / bottom outliers
    top = df_resid.nlargest(n_outliers, "residual_entropy_model").copy()
    bottom = df_resid.nsmallest(n_outliers, "residual_entropy_model").copy()

    top["type"] = "Over-performers"
    bottom["type"] = "Under-performers"

    plot_df = pd.concat([top, bottom], axis=0)
    plot_df = plot_df.sort_values("residual_entropy_model")

    # Final label
    plot_df["kommun_label"] = plot_df[label_col]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.barplot(
        data=plot_df,
        x="residual_entropy_model",
        y="kommun_label",
        hue="type",
        dodge=False,
        palette={"Over-performers": "#2E86AB", "Under-performers": "#A23B72"},
        ax=ax,
    )

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual from Entropy-Only Model", fontsize=12)
    ax.set_ylabel("Municipality", fontsize=12)
    ax.set_title(
        f"Top {n_outliers} Over- and Under-performing Municipalities",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(RESID_TOP_BOTTOM_PLOT, dpi=DPI, bbox_inches="tight")
    print(f"   Saved: {RESID_TOP_BOTTOM_PLOT}")
    plt.close()


# ==============================================================================
# PLOT 4: CORRELATION HEATMAP
# ==============================================================================


def plot_correlation_heatmap(df_entropy):
    """
    Cleaner correlation heatmap using the entropy analysis data directly.

    Shows the lower triangle of correlations between:
    - balanced_accuracy
    - entropy
    - majority_prop
    - hhi (if available)
    - n_test (if available)
    """
    print("\nCreating correlation heatmap...")

    # Choose variables to include (keep only those that exist)
    cols = ["balanced_accuracy", "entropy", "majority_prop", "hhi", "n_test"]
    cols = [c for c in cols if c in df_entropy.columns]

    if len(cols) < 2:
        print("  Not enough variables found for correlation heatmap. Skipping.")
        return

    corr = df_entropy[cols].corr()

    # Mask the upper triangle so we only see the lower one
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title(
        "Correlation Matrix: Predictability, Entropy, Size and Imbalance",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(CORR_HEATMAP_PLOT, dpi=DPI, bbox_inches="tight")
    print(f"   Saved: {CORR_HEATMAP_PLOT}")
    plt.close()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


def main():
    print("=" * 80)
    print("RQ3 VISUALIZATION GENERATION")
    print("=" * 80)

    # Load kommun code -> name lookup and main data tables
    code_to_name = load_kommun_lookup()
    df_entropy, df_desc, df_corr, df_reg, df_resid = load_required_data()

    # Attach municipality names where possible
    if "Kommun" in df_entropy.columns:
        df_entropy["kommun_name"] = (
            df_entropy["Kommun"].astype(str).str.strip().str.zfill(4).map(code_to_name)
        )
    if df_resid is not None and "Kommun" in df_resid.columns:
        df_resid["kommun_name"] = (
            df_resid["Kommun"].astype(str).str.strip().str.zfill(4).map(code_to_name)
        )

    print("\nGenerating visualizations...")
    print("-" * 80)

    # 1–2: histograms
    plot_entropy_histogram(df_entropy)
    plot_balacc_histogram(df_entropy)

    # 3–5: scatter plots
    plot_balacc_vs_entropy(df_entropy, df_reg)
    plot_balacc_vs_majority_prop(df_entropy)
    plot_balacc_vs_logn_test(df_entropy, df_reg)

    # 6–7: residuals / outliers
    plot_residuals_vs_entropy(df_resid)
    plot_residuals_top_bottom(df_resid, n_outliers=N_OUTLIERS)

    # 8: correlation heatmap
    plot_correlation_heatmap(df_entropy)

    print("\n" + "=" * 80)
    print("RQ3 VISUALIZATION GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
