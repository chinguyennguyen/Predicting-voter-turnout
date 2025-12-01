"""
RQ1 Visualization Script - Using MONA Output Files

Generates RQ1 visualizations from the MONA output files:
- rq1_all_experiments.csv (detailed results)
- rq1_pareto_frontier.csv (Pareto optimal models)

Creates:
1. Method comparison plot (stratified vs balanced in 50k-800k range)
2. Stratified scaling curve (1M-4M)
3. Two-zone plots for:
   - Balanced Accuracy
   - Macro/Weighted F1
   - Training Time
4. Pareto frontier plot (performance vs training time)

Runtime: ~30 seconds for all plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# File paths
RESULTS_DIR = Path("outputs_mona")

# Input CSVs
ALL_EXPERIMENTS_CSV = RESULTS_DIR / "tables" / "rq1_all_experiments.csv"
PARETO_CSV = RESULTS_DIR / "tables" / "rq1_pareto_frontier.csv"

# Output plots
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_PLOT = PLOTS_DIR / "rq1_method_comparison.png"
SCALING_PLOT = PLOTS_DIR / "rq1_stratified_scaling.png"

# Two-zone plots (one per metric)
COMBINED_PLOT_BAL_ACC = PLOTS_DIR / "rq1_two_zone_balanced_accuracy.png"
COMBINED_PLOT_F1 = PLOTS_DIR / "rq1_two_zone_macro_f1.png"
COMBINED_PLOT_TRAIN_TIME = PLOTS_DIR / "rq1_two_zone_train_time.png"

# For backward compatibility, you can still treat this as "the" combined plot
COMBINED_PLOT = COMBINED_PLOT_BAL_ACC

PARETO_PLOT = PLOTS_DIR / "rq1_pareto_frontier.png"

# Global styling
COLORS = {
    'stratified': '#2E86AB',  # Blue
    'balanced': '#A23B72'     # Purple
}
STRATIFIED_SCALING_COLOR = '#2E86AB'  # Same blue as comparison zone for consistency

DPI = 300
FIGSIZE_STANDARD = (10, 6)
FIGSIZE_COMBINED = (14, 7)

# Zone boundary (where balanced sampling maxes out)
BALANCED_LIMIT = 840  # in thousands

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def aggregate_results(df, group_cols):
    """
    Aggregate detailed results by group columns.
    Calculates mean and standard error for balanced_accuracy.
    """
    agg_dict = {
        'balanced_accuracy': ['mean', 'std', 'count']
    }
    
    summary = df.groupby(group_cols).agg(agg_dict).reset_index()
    summary.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" 
                      for col in summary.columns]
    
    # Calculate standard error
    summary['balanced_accuracy_se'] = (summary['balanced_accuracy_std'] / 
                                       np.sqrt(summary['balanced_accuracy_count']))
    
    return summary


def aggregate_multiple_metrics(df, group_cols):
    """
    Aggregate detailed results for multiple metrics (legacy helper).
    """
    agg_dict = {
        'balanced_accuracy': ['mean', 'std', 'count'],
        'weighted_f1': ['mean', 'std', 'count'],
        'train_time': ['mean', 'std', 'count']
    }
    
    summary = df.groupby(group_cols).agg(agg_dict).reset_index()
    summary.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" 
                      for col in summary.columns]
    
    # Calculate standard errors
    for metric in ['balanced_accuracy', 'weighted_f1', 'train_time']:
        summary[f'{metric}_se'] = (summary[f'{metric}_std'] / 
                                   np.sqrt(summary[f'{metric}_count']))
    
    return summary


def aggregate_single_metric(df, group_cols, metric_col):
    """
    Aggregate a single metric by group columns.
    Returns mean, std, count, and standard error columns for the metric.
    """
    agg = df.groupby(group_cols)[metric_col].agg(['mean', 'std', 'count']).reset_index()
    agg.rename(columns={
        'mean': f'{metric_col}_mean',
        'std': f'{metric_col}_std',
        'count': f'{metric_col}_count'
    }, inplace=True)
    agg[f'{metric_col}_se'] = agg[f'{metric_col}_std'] / np.sqrt(agg[f'{metric_col}_count'])
    return agg

# ==============================================================================
# PLOT 1: METHOD COMPARISON (50k-800k)
# ==============================================================================

def create_method_comparison_plot(df):
    """
    Plot comparing stratified vs balanced sampling in the comparison zone.
    """
    print("\n1. Creating method comparison plot...")
    
    # Filter to comparison zone (≤800k)
    comparison_df = df[df['training_size_k'] <= 800].copy()
    
    # Aggregate by training_size and method
    summary = aggregate_results(comparison_df, ['training_size_k', 'method'])
    
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    
    # Plot both methods
    for method in ['stratified', 'balanced']:
        method_data = summary[summary['method'] == method]
        
        ax.plot(method_data['training_size_k'], 
                method_data['balanced_accuracy_mean'],
                marker='o', linewidth=2, markersize=8,
                color=COLORS[method], label=method.capitalize())
        
        # Add error bars if multiple seeds
        if method_data['balanced_accuracy_count'].max() > 1:
            ax.fill_between(
                method_data['training_size_k'],
                method_data['balanced_accuracy_mean'] - method_data['balanced_accuracy_se'],
                method_data['balanced_accuracy_mean'] + method_data['balanced_accuracy_se'],
                alpha=0.2, color=COLORS[method]
            )
    
    ax.set_xlabel('Training Size (thousands)', fontsize=12)
    ax.set_ylabel('Balanced Accuracy', fontsize=12)
    ax.set_title('Method Comparison: Stratified vs Balanced Sampling\n(50k-800k Training Range)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(COMPARISON_PLOT, dpi=DPI, bbox_inches='tight')
    print(f"   Saved: {COMPARISON_PLOT}")
    plt.close()


# ==============================================================================
# PLOT 2: STRATIFIED SCALING (1M-4M)
# ==============================================================================

def create_scaling_plot(df):
    """
    Plot stratified sampling performance at large scales (1M-4M).
    """
    print("\n2. Creating stratified scaling plot...")
    
    # Filter to scaling zone (≥1M) and stratified only
    scaling_df = df[(df['training_size_k'] >= 1000) & 
                    (df['method'] == 'stratified')].copy()
    
    if len(scaling_df) == 0:
        print("   WARNING: No scaling data found. Skipping plot.")
        return
    
    # Aggregate by training_size
    summary = aggregate_results(scaling_df, ['training_size_k'])
    
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    
    ax.plot(summary['training_size_k'], 
            summary['balanced_accuracy_mean'],
            marker='s', linewidth=2.5, markersize=10,
            color=STRATIFIED_SCALING_COLOR, label='Stratified')
    
    # Add error bars if multiple seeds
    if summary['balanced_accuracy_count'].max() > 1:
        ax.fill_between(
            summary['training_size_k'],
            summary['balanced_accuracy_mean'] - summary['balanced_accuracy_se'],
            summary['balanced_accuracy_mean'] + summary['balanced_accuracy_se'],
            alpha=0.2, color=STRATIFIED_SCALING_COLOR
        )
    
    ax.set_xlabel('Training Size (thousands)', fontsize=12)
    ax.set_ylabel('Balanced Accuracy', fontsize=12)
    ax.set_title('Stratified Sampling at Scale\n(1M-4M Training Range)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(SCALING_PLOT, dpi=DPI, bbox_inches='tight')
    print(f"   Saved: {SCALING_PLOT}")
    plt.close()


# ==============================================================================
# PLOT 3: GENERIC TWO-ZONE PLOT FOR A SINGLE METRIC
# ==============================================================================

def create_two_zone_plot_for_metric(
    df,
    metric_col,
    display_name,
    output_path
):
    """
    Combined two-zone plot for a single metric:
    - Comparison zone (≤800k): stratified vs balanced
    - Scaling zone (≥1M): stratified only
    """
    print(f"\n3. Creating two-zone plot for {display_name}...")
    
    if metric_col not in df.columns:
        print(f"   WARNING: Column '{metric_col}' not found. Skipping this metric.")
        return
    
    # Aggregate by training_size and method for this metric
    summary = aggregate_single_metric(df, ['training_size_k', 'method'], metric_col)
    
    fig, ax = plt.subplots(figsize=FIGSIZE_COMBINED)
    
    # Plot comparison zone (both methods, ≤800k)
    for method in ['stratified', 'balanced']:
        method_data = summary[
            (summary['method'] == method) & 
            (summary['training_size_k'] <= 800)
        ]
        
        if len(method_data) > 0:
            ax.plot(method_data['training_size_k'], 
                    method_data[f'{metric_col}_mean'],
                    marker='o', linewidth=2, markersize=8,
                    color=COLORS[method], label=f'{method.capitalize()} (≤800k)')
            
            # Add error bars
            if method_data[f'{metric_col}_count'].max() > 1:
                ax.fill_between(
                    method_data['training_size_k'],
                    method_data[f'{metric_col}_mean'] - method_data[f'{metric_col}_se'],
                    method_data[f'{metric_col}_mean'] + method_data[f'{metric_col}_se'],
                    alpha=0.2, color=COLORS[method]
                )
    
    # Plot scaling zone (stratified only, ≥1M)
    scaling_data = summary[
        (summary['method'] == 'stratified') & 
        (summary['training_size_k'] >= 1000)
    ]
    
    if len(scaling_data) > 0:
        ax.plot(scaling_data['training_size_k'], 
                scaling_data[f'{metric_col}_mean'],
                marker='s', linewidth=2.5, markersize=10,
                color=STRATIFIED_SCALING_COLOR, 
                label='Stratified (≥1M)', linestyle='--')
        
        # Add error bars
        if scaling_data[f'{metric_col}_count'].max() > 1:
            ax.fill_between(
                scaling_data['training_size_k'],
                scaling_data[f'{metric_col}_mean'] - scaling_data[f'{metric_col}_se'],
                scaling_data[f'{metric_col}_mean'] + scaling_data[f'{metric_col}_se'],
                alpha=0.2, color=STRATIFIED_SCALING_COLOR
            )
    
    # Add vertical line at balanced limit
    ax.axvline(x=BALANCED_LIMIT, color='gray', linestyle=':', linewidth=2, 
               label=f'Balanced Limit ({BALANCED_LIMIT}k)')
    
    # Add zone labels
    ymin, ymax = ax.get_ylim()
    ax.text(400, ymin + 0.02 * (ymax - ymin),
            'Comparison Zone\n(Both Methods)', 
            ha='center', fontsize=10, style='italic', color='gray')
    ax.text(2500, ymin + 0.02 * (ymax - ymin),
            'Scaling Zone\n(Stratified Only)', 
            ha='center', fontsize=10, style='italic', color='gray')
    
    ax.set_xlabel('Training Size (thousands)', fontsize=12)
    ax.set_ylabel(display_name, fontsize=12)
 #   ax.set_title(f'RQ1: Complete Analysis - {display_name}', 
 #                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    plt.close()


def create_all_two_zone_plots(df):
    """
    Convenience wrapper to produce two-zone plots for:
    - Balanced Accuracy
    - Macro/Weighted F1
    - Training Time
    """
    # Balanced Accuracy
    create_two_zone_plot_for_metric(
        df,
        metric_col='balanced_accuracy',
        display_name='Balanced Accuracy',
        output_path=COMBINED_PLOT_BAL_ACC
    )

    # Macro F1 (or fallback to Weighted F1)
    if 'macro_f1' in df.columns:
        f1_col = 'macro_f1'
        f1_label = 'Macro F1'
    elif 'weighted_f1' in df.columns:
        print("   WARNING: 'macro_f1' not found. Using 'weighted_f1' instead.")
        f1_col = 'weighted_f1'
        f1_label = 'Weighted F1'
    else:
        print("   WARNING: No macro_f1 or weighted_f1 column found. Skipping F1 two-zone plot.")
        f1_col = None
        f1_label = None

    if f1_col is not None:
        create_two_zone_plot_for_metric(
            df,
            metric_col=f1_col,
            display_name=f1_label,
            output_path=COMBINED_PLOT_F1
        )

    # Training Time
    if 'train_time' in df.columns:
        create_two_zone_plot_for_metric(
            df,
            metric_col='train_time',
            display_name='Training Time (seconds)',
            output_path=COMBINED_PLOT_TRAIN_TIME
        )
    else:
        print("   WARNING: 'train_time' column not found. Skipping train time two-zone plot.")


# ==============================================================================
# PLOT 4: PARETO FRONTIER
# ==============================================================================

def create_pareto_plot(all_df, pareto_df):
    """
    Plot Pareto frontier showing performance vs training time trade-off.
    """
    print("\n4. Creating Pareto frontier plot...")
    
    if not PARETO_CSV.exists():
        print("   WARNING: Pareto CSV not found. Skipping plot.")
        return
    
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    
    # Plot all experiments (background)
    for method in ['stratified', 'balanced']:
        method_data = all_df[all_df['method'] == method]
        ax.scatter(method_data['train_time'], 
                  method_data['balanced_accuracy'],
                  alpha=0.3, s=50, color=COLORS[method], 
                  label=f'{method.capitalize()} (all)')
    
    # Plot Pareto frontier (highlighted)
    pareto_stratified = pareto_df[pareto_df['method'] == 'stratified']
    pareto_balanced = pareto_df[pareto_df['method'] == 'balanced']
    
    if len(pareto_stratified) > 0:
        ax.scatter(pareto_stratified['train_time'], 
                  pareto_stratified['balanced_accuracy'],
                  s=200, marker='*', edgecolors='black', linewidth=1.5,
                  color=COLORS['stratified'], label='Stratified (Pareto)', 
                  zorder=5)
    
    if len(pareto_balanced) > 0:
        ax.scatter(pareto_balanced['train_time'], 
                  pareto_balanced['balanced_accuracy'],
                  s=200, marker='*', edgecolors='black', linewidth=1.5,
                  color=COLORS['balanced'], label='Balanced (Pareto)', 
                  zorder=5)
    
    # Connect Pareto points
    pareto_sorted = pareto_df.sort_values('train_time')
    ax.plot(pareto_sorted['train_time'], 
            pareto_sorted['balanced_accuracy'],
            'k--', linewidth=1, alpha=0.5, zorder=4)
    
    ax.set_xlabel('Training Time (seconds)', fontsize=12)
    ax.set_ylabel('Balanced Accuracy', fontsize=12)
    ax.set_title('Pareto Frontier: Performance vs Training Time Trade-off', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PARETO_PLOT, dpi=DPI, bbox_inches='tight')
    print(f"   Saved: {PARETO_PLOT}")
    plt.close()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Generate all RQ1 visualizations."""
    
    print("=" * 80)
    print("RQ1 VISUALIZATION GENERATION")
    print("=" * 80)
    
    # Check if results exist
    if not ALL_EXPERIMENTS_CSV.exists():
        print(f"\nERROR: Results file not found: {ALL_EXPERIMENTS_CSV}")
        print("Please run RQ1 analysis in MONA first and copy results.")
        return
    
    # Load all experiments
    print(f"\nLoading all experiments from {ALL_EXPERIMENTS_CSV}")
    all_df = pd.read_csv(ALL_EXPERIMENTS_CSV)
    print(f"Loaded {len(all_df)} experiments")
    
    # Print column names for debugging
    print(f"\nColumns in CSV: {list(all_df.columns)}")
    
    # Load Pareto frontier if available
    pareto_df = None
    if PARETO_CSV.exists():
        print(f"Loading Pareto frontier from {PARETO_CSV}")
        pareto_df = pd.read_csv(PARETO_CSV)
        print(f"Loaded {len(pareto_df)} Pareto-optimal models")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    print("-" * 80)
    
    create_method_comparison_plot(all_df)
    create_scaling_plot(all_df)
    create_all_two_zone_plots(all_df)
    
    if pareto_df is not None:
        create_pareto_plot(all_df, pareto_df)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nAll plots saved to: {PLOTS_DIR}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Overall best performance
    best_row = all_df.loc[all_df['balanced_accuracy'].idxmax()]
    print(f"\nBest Overall Performance:")
    print(f"  Method: {best_row['method']}")
    print(f"  Training Size: {best_row['training_size_k']}k")
    print(f"  Balanced Accuracy: {best_row['balanced_accuracy']:.4f}")
    print(f"  Training Time: {best_row['train_time']:.1f}s")
    
    # Method comparison in overlap zone
    comparison_df = all_df[all_df['training_size_k'] <= 800]
    for method in ['stratified', 'balanced']:
        method_df = comparison_df[comparison_df['method'] == method]
        if len(method_df) > 0:
            print(f"\n{method.capitalize()} (50k-800k):")
            print(f"  Mean Accuracy: {method_df['balanced_accuracy'].mean():.4f}")
            print(f"  Best Accuracy: {method_df['balanced_accuracy'].max():.4f}")
    
    # Scaling zone performance
    scaling_df = all_df[(all_df['training_size_k'] >= 1000) & 
                        (all_df['method'] == 'stratified')]
    if len(scaling_df) > 0:
        print(f"\nStratified Scaling (1M-4M):")
        print(f"  Mean Accuracy: {scaling_df['balanced_accuracy'].mean():.4f}")
        print(f"  Best Accuracy: {scaling_df['balanced_accuracy'].max():.4f}")


if __name__ == "__main__":
    main()
