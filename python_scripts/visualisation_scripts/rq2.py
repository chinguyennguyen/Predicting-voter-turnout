"""
RQ2 Visualization: Geographic Predictability Analysis

Generates RQ1 visualizations from the MONA output files:
- rq2_kommun_predictability.csv
- rq2_summary_statistics.csv

This script should be run AFTER RQ2 analysis is complete and
the results CSV has been copied back from MONA.

Creates:
1. Distribution of predictability across municipalities
2. Municipality size vs predictability scatter plot
3. Metrics comparison across municipalities
4. Summary statistics table visualization
5. Choropleth maps of Sweden by municipality for multiple metrics

"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

# Set publication-ready style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# ============================================================================
# Municipality name lookup
# ============================================================================

KOMMUN_LOOKUP_FILE = Path("data") / "municipality_code.csv"

def load_kommun_lookup():
    """
    Load municipality_code.csv and return a dict {code4: name}.
    Expects columns: 'Code', 'Name'.
    """
    df_k = pd.read_csv(KOMMUN_LOOKUP_FILE)  # add sep="\t" if file is tab-separated
    df_k["Code"] = df_k["Code"].astype(str).str.strip().str.zfill(4)
    return dict(zip(df_k["Code"], df_k["Name"]))

# ============================================================================
# Configuration
# ============================================================================

# Input paths
RESULTS_DIR = Path("outputs_mona")
INPUT_CSV = RESULTS_DIR / "tables" / "rq2_kommun_predictability.csv"
SUMMARY_CSV = RESULTS_DIR / "tables" / "rq2_summary_statistics.csv"

# Shapefile path (Sweden municipalities)
# Folder structure:
#   data/
#       alla_kommuner/
#           alla_kommuner.shp
#           alla_kommuner.shx
#           alla_kommuner.dbf
#           alla_kommuner.prj
SHAPEFILE_PATH = Path("data") / "alla_kommuner" / "alla_kommuner.shp"

# Column name for municipality in your RQ2 CSV
RQ2_MUNI_COL = "Kommun"  # typically 4-digit string municipality code

# Output directory
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Visualization Functions (non-map)
# ============================================================================

def plot_predictability_distribution(df, output_path):
    """
    Plot distribution of balanced accuracy across municipalities.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1 = axes[0]
    ax1.hist(df["balanced_accuracy"], bins=30, edgecolor="black", alpha=0.7)
    ax1.axvline(
        df["balanced_accuracy"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {df['balanced_accuracy'].mean():.3f}",
    )
    ax1.axvline(
        df["balanced_accuracy"].median(),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {df['balanced_accuracy'].median():.3f}",
    )
    ax1.set_xlabel("Balanced Accuracy", fontsize=12)
    ax1.set_ylabel("Number of Municipalities", fontsize=12)
    ax1.set_title(
        "Distribution of Predictability Across Municipalities",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2 = axes[1]
    box = ax2.boxplot(
        [df["balanced_accuracy"]],
        vert=True,
        patch_artist=True,
        labels=["Balanced Accuracy"],
    )
    box["boxes"][0].set_facecolor("lightblue")
    box["boxes"][0].set_alpha(0.7)
    ax2.set_ylabel("Balanced Accuracy", fontsize=12)
    ax2.set_title("Predictability Range", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    # Add statistics text
    stats_text = (
        f"Mean: {df['balanced_accuracy'].mean():.3f}\n"
        f"Std: {df['balanced_accuracy'].std():.3f}\n"
        f"Min: {df['balanced_accuracy'].min():.3f}\n"
        f"Max: {df['balanced_accuracy'].max():.3f}"
    )
    ax2.text(
        0.95,
        0.05,
        stats_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   Saved: {output_path}")
    plt.close()


def plot_size_vs_predictability(df, output_path):
    """
    Scatter plot: Municipality size vs predictability.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        df["n_observations"],
        df["balanced_accuracy"],
        alpha=0.6,
        s=50,
        c=df["balanced_accuracy"],
        cmap="RdYlGn",
        edgecolors="black",
        linewidth=0.5,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Balanced Accuracy", fontsize=12)

    # Add trend line
    z = np.polyfit(df["n_observations"], df["balanced_accuracy"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["n_observations"].min(), df["n_observations"].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label="Trend")

    # Calculate correlation
    corr = df["n_observations"].corr(df["balanced_accuracy"])

    ax.set_xlabel("Number of Test Observations", fontsize=12)
    ax.set_ylabel("Balanced Accuracy", fontsize=12)
    ax.set_title("Municipality Size vs Predictability", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add correlation text
    ax.text(
        0.05,
        0.95,
        f"Correlation: {corr:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   Saved: {output_path}")
    plt.close()


def plot_metrics_comparison(df, output_path):
    """
    Compare different performance metrics across municipalities.
    """
    metrics = ["balanced_accuracy", "macro_f1", "weighted_f1", "accuracy"]
    metric_labels = ["Balanced\nAccuracy", "Macro\nF1", "Weighted\nF1", "Accuracy"]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for box plots
    data_to_plot = [df[metric] for metric in metrics]

    # Create box plot
    bp = ax.boxplot(data_to_plot, labels=metric_labels, patch_artist=True)

    # Customize colors
    colors = ["lightblue", "lightgreen", "lightyellow", "lightcoral"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Distribution of Performance Metrics Across Municipalities",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")

    # Add mean markers
    means = [df[metric].mean() for metric in metrics]
    ax.plot(range(1, len(means) + 1), means, "ro", markersize=8, label="Mean")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   Saved: {output_path}")
    plt.close()


def create_summary_table_plot(summary_df, output_path):
    """
    Create a visual table of summary statistics.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("tight")
    ax.axis("off")

    # Format numbers for display
    display_data = summary_df.copy()
    for col in ["mean", "std", "min", "q25", "median", "q75", "max"]:
        if col in display_data.columns:
            display_data[col] = display_data[col].apply(
                lambda x: f"{x:.4f}" if isinstance(x, float) else x
            )

    # Create table
    table = ax.table(
        cellText=display_data.values,
        colLabels=display_data.columns,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(display_data.columns)):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(display_data) + 1):
        for j in range(len(display_data.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")

    plt.title(
        "Summary Statistics Across Municipalities",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   Saved: {output_path}")
    plt.close()


def plot_top_bottom_municipalities(df, output_path, n=10):
    """
    Create horizontal bar chart showing top and bottom municipalities.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    LIGHT_GREEN = "#99CC97"
    LIGHT_RED   = "#C78F8F"

    # Top municipalities
    top = df.nlargest(n, 'balanced_accuracy')
    ax1.barh(range(len(top)), top['balanced_accuracy'], color=LIGHT_GREEN, alpha=0.7)
    ax1.set_yticks(range(len(top)))
    ax1.set_yticklabels(top['kommun_name'])  
    ax1.set_xlabel('Balanced Accuracy', fontsize=12)
    ax1.set_title(f'Most {n} Predictable Municipalities', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(top['balanced_accuracy']):
        ax1.text(v, i, f' {v:.3f}', va='center', fontsize=9)
    
    # Bottom municipalities
    bottom = df.nsmallest(n, 'balanced_accuracy')
    ax2.barh(range(len(bottom)), bottom['balanced_accuracy'], color=LIGHT_RED, alpha=0.7)
    ax2.set_yticks(range(len(bottom)))
    ax2.set_yticklabels(bottom['kommun_name'])  
    ax2.set_xlabel('Balanced Accuracy', fontsize=12)
    ax2.set_title(f'Least {n} Predictable Municipalities', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(bottom['balanced_accuracy']):
        ax2.text(v, i, f' {v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    plt.close()


# ============================================================================
# Choropleth Map Functions
# ============================================================================

def prepare_geo_dataframe(df, shapefile_path, rq2_col=RQ2_MUNI_COL):
    """
    Read Sweden municipality shapefile and merge with RQ2 results.

    - df[rq2_col]: municipality codes (numeric or string)
    - shapefile: contains some municipality code column, auto-detected
    """
    print(f"\n   Loading shapefile from {shapefile_path}")

    # Let GDAL try to reconstruct .shx if missing
    os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")

    # Read shapefile
    gdf = gpd.read_file(shapefile_path)

    print(f"   Loaded {len(gdf)} municipalities from shapefile")
    print(f"   Shapefile columns: {list(gdf.columns)}")

    # Try to auto-detect the municipality code column
    candidate_cols = [
        "KOMMUNKOD",  # common SCB name
        "ID",
        "KNKOD",
        "KnKod",
        "kommun_kod",
        "Kommunkod",
        "KommunNr",
    ]
    shape_col = None
    for col in candidate_cols:
        if col in gdf.columns:
            shape_col = col
            break

    if shape_col is None:
        raise ValueError(
            "Could not find municipality code column in shapefile.\n"
            f"Looked for: {candidate_cols}\n"
            f"Available columns: {list(gdf.columns)}"
        )

    print(f"   Using '{shape_col}' as municipality code column in shapefile")

    # Convert codes to 4-digit strings in both data sets
    df = df.copy()
    df[rq2_col] = df[rq2_col].astype(str).str.zfill(4)
    gdf[shape_col] = gdf[shape_col].astype(str).str.zfill(4)

    print("   Merging shapefile with RQ2 results...")
    merged = gdf.merge(df, left_on=shape_col, right_on=rq2_col, how="left")

    print(f"   Merged geodataframe has {len(merged)} rows")
    print(f"   Municipalities with RQ2 data: {merged[rq2_col].notna().sum()}")

    missing = merged[rq2_col].isna().sum()
    if missing > 0:
        print(f"   Warning: {missing} municipalities have no RQ2 data")

    return merged


def plot_metric_map(geo_df, metric, output_path, cmap="RdYlGn"):
    """
    Plot a choropleth map of Sweden by municipality for a given metric.
    """
    if metric not in geo_df.columns:
        print(f"   WARNING: metric '{metric}' not found in data. Skipping map.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    vmin = geo_df[metric].min()
    vmax = geo_df[metric].max()

    geo_df.plot(
        column=metric,
        ax=ax,
        cmap=cmap,
        linewidth=0.2,
        edgecolor="black",
        legend=True,
        vmin=vmin,
        vmax=vmax,
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "black",
            "hatch": "///",
            "label": "No data",
        },
    )

    ax.set_axis_off()
    # Choose a nicer title for the training-size map
    if metric == "n_observations":
        title = "Test Sample Size by Municipality"
    else:
        pretty_name = metric.replace("_", " ").title()
        title = f"{pretty_name} by Municipality"

    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   Saved map: {output_path}")
    plt.close()


def generate_all_metric_maps(df, shapefile_path, output_dir):
    """
    Generate choropleth maps for several metrics over Sweden.
    """
    if not shapefile_path.exists():
        print(f"\n   WARNING: Shapefile not found at {shapefile_path}. Skipping maps.")
        return

    # Prepare merged geodataframe
    geo_df = prepare_geo_dataframe(df, shapefile_path)

    # Core performance metrics
    metrics = ["balanced_accuracy", "macro_f1", "weighted_f1", "accuracy"]

    # Add training size from rq2_kommun_predictability if available
    if "n_observations" in df.columns:
        metrics.append("n_observations")
        print("   Including training size map using column 'n_observations'.")
    else:
        print("   Column 'n_observations' not found in df; skipping training size map.")

    print("\n   Generating choropleth maps for metrics:")

    for metric in metrics:
        print(f"     - {metric}")
        output_path = output_dir / f"rq2_map_{metric}.png"

        # Optional: distinct colormap for n_observations
        cmap = "Blues" if metric == "n_observations" else "RdYlGn"

        # plot_metric_map already has cmap default,
        # we just override it for n_observations.
        plot_metric_map(geo_df, metric, output_path, cmap=cmap)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Generate all visualizations."""

    print("=" * 80)
    print("RQ2 VISUALIZATION GENERATION")
    print("=" * 80)
    print()

    # Check if results exist
    if not INPUT_CSV.exists():
        print(f"ERROR: Results file not found: {INPUT_CSV}")
        print("Please run RQ2 analysis first and copy results from MONA.")
        return

    # Load results
    print(f"Loading results from {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded data for {len(df)} municipalities")
    print()

    # Load summary statistics if available
    summary_df = None
    if SUMMARY_CSV.exists():
        print(f"Loading summary statistics from {SUMMARY_CSV}")
        summary_df = pd.read_csv(SUMMARY_CSV)
        print()
    
    # Attach municipality names
    code_to_name = load_kommun_lookup()
    df[RQ2_MUNI_COL] = df[RQ2_MUNI_COL].astype(str).str.strip().str.zfill(4)
    df["kommun_name"] = df[RQ2_MUNI_COL].map(code_to_name)

    # Generate visualizations
    print("Generating visualizations...")
    print("-" * 80)

    # 1. Predictability distribution
    print("\n1. Predictability distribution...")
    plot_predictability_distribution(
        df, PLOTS_DIR / "rq2_predictability_distribution.png"
    )

    # 2. Size vs predictability
    print("\n2. Size vs predictability scatter...")
    plot_size_vs_predictability(df, PLOTS_DIR / "rq2_size_vs_predictability.png")

    # 3. Metrics comparison
    print("\n3. Metrics comparison...")
    plot_metrics_comparison(df, PLOTS_DIR / "rq2_metrics_comparison.png")

    # 4. Top/Bottom municipalities
    print("\n4. Top/Bottom municipalities bar chart...")
    plot_top_bottom_municipalities(df, PLOTS_DIR / "rq2_top_bottom_municipalities.png")

    # 5. Summary table
    if summary_df is not None:
        print("\n5. Summary statistics table...")
        create_summary_table_plot(summary_df, PLOTS_DIR / "rq2_summary_table.png")

    # 6. Choropleth maps
    print("\n6. Choropleth maps of Sweden by municipality...")
    generate_all_metric_maps(df, SHAPEFILE_PATH, PLOTS_DIR)

    print()
    print("=" * 80)
    print("VISUALIZATION GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nAll plots saved to: {PLOTS_DIR}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nTotal municipalities: {len(df)}")
    print(f"Mean balanced accuracy: {df['balanced_accuracy'].mean():.4f}")
    print(f"Std balanced accuracy: {df['balanced_accuracy'].std():.4f}")
    print(f"Min balanced accuracy: {df['balanced_accuracy'].min():.4f}")
    print(f"Max balanced accuracy: {df['balanced_accuracy'].max():.4f}")
    
    # Top 5 municipalities
    print("\nTop 5 Most Predictable Municipalities:")
    top5 = df.nlargest(5, 'balanced_accuracy')
    for _, row in top5.iterrows():
        print(f"  {row['kommun_name']}: {row['balanced_accuracy']:.4f}")
    
    # Bottom 5 municipalities
    print("\nBottom 5 Least Predictable Municipalities:")
    bottom5 = df.nsmallest(5, 'balanced_accuracy')
    for _, row in bottom5.iterrows():
        print(f"  {row['kommun_name']}: {row['balanced_accuracy']:.4f}")



if __name__ == "__main__":
    main()