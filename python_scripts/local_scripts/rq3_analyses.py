"""
RQ2 ENTROPY ANALYSIS - DEVELOPMENT SCRIPT
Swedish Voter Turnout Prediction Project (Synthetic Data)

Research Question:
    Does outcome diversity (entropy of Y distribution) explain variation
    in municipality-level predictability?

Hypothesis:
    Municipalities with homogeneous voting outcomes (low entropy) are easier
    to predict than municipalities with heterogeneous outcomes (high entropy).

Approach:
    1. Load RQ2 predictability results (rq2_kommun_predictability.csv)
    2. Calculate Shannon entropy of Y distribution per municipality
    3. Calculate alternative diversity measures (majority proportion, HHI)
    4. Merge with RQ2 predictability data
    5. Run descriptive statistics, visualizations, and simple regressions

FOR LOCAL TESTING: Uses synthetic data
FOR PRODUCTION: Use rq3.py in MONA

Main Outputs:
-   outputs_synthetic/tables/rq3_entropy_analysis.csv  # Main analysis results
-   outputs_synthetic/tables/regession_analysis.csv  # Regression results

"""

# %% IMPORTS

import pandas as pd
import numpy as np
import time
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Make python_scripts/ visible for config
THIS_DIR = Path(__file__).resolve().parent
PYTHON_SCRIPTS_DIR = THIS_DIR.parent  # .../python_scripts

if str(PYTHON_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_SCRIPTS_DIR))

# Shared MONA modules
from mona_scripts.config import setup_config
from mona_scripts.data_loader import load_data
from mona_scripts.utils import get_logger

warnings.filterwarnings("ignore")

print("All imports successful!")

# %% CONFIGURATION

# Paths (local development)
DATA_PATH = "data/synthetic_data.dta"
OUTPUT_DIR = "outputs_synthetic"

# Initialize logger
logger = get_logger(__name__)
logger.info("Logger initialized")

# Setup configuration
config = setup_config(DATA_PATH, OUTPUT_DIR)
logger.info("Configuration loaded")
logger.info(f"Data path: {config.DATA_FILE}")
logger.info(f"Output directory: {config.OUTPUT_DIR}")

# RQ3-specific config
config.KOMMUN_COLUMN = 'Kommun'
config.TARGET_COLUMN = 'y_mun'  # already set in BaseConfig, but make explicit


# %% HELPER FUNCTIONS

def normalize_kommun_code(val):
    """
    Normalize municipality code to a 4-digit string (e.g., '0114'),
    consistent with mona_rq2.py (where Kommun is written as 4-digit string).

    Handles both numeric and string inputs and preserves leading zeros.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s == "":
        return np.nan
    try:
        # If it's numeric-like, cast to int first then zfill(4)
        return str(int(s)).zfill(4)
    except (ValueError, TypeError):
        # Non-numeric, just zero-pad to length 4
        return s.zfill(4)


def calculate_shannon_entropy(proportions):
    """
    Calculate Shannon entropy for a probability distribution.

    H = -Σ(p_i * log(p_i))

    Parameters
    ----------
    proportions : array-like
        Proportions for each class (should sum to 1)

    Returns
    -------
    float : Shannon entropy value
    """
    p = np.array(proportions, dtype=float)
    p = p[p > 0]  # Remove zeros to avoid log(0)
    if len(p) == 0:
        return 0.0
    entropy = -np.sum(p * np.log(p))
    return float(entropy)


def calculate_y_distributions(data, config, logger):
    """
    Calculate the distribution of voting transitions (Y) per municipality.

    Uses the raw data (before train/test split) so that all municipalities
    are represented.

    Returns
    -------
    DataFrame with columns:
        kommun        : 4-digit municipality code (string, e.g., '0114')
        n_obs         : number of observations in this municipality
        p_<class>     : proportion of each outcome class (e.g., p_VV, p_NN, ...)
        entropy       : Shannon entropy of the Y distribution
        majority_prop : proportion of the majority class
        hhi           : Herfindahl–Hirschman Index of concentration
    """
    logger.info("=" * 80)
    logger.info("CALCULATING Y DISTRIBUTIONS PER MUNICIPALITY")
    logger.info("=" * 80)
    logger.info("")

    # Normalize municipality codes to 4-digit strings
    data = data.copy()
    data['kommun_str'] = data[config.KOMMUN_COLUMN].apply(normalize_kommun_code)

    # Drop rows with missing municipality or target
    mask_valid = data['kommun_str'].notna() & data[config.TARGET_COLUMN].notna()
    data = data[mask_valid].copy()

    all_kommun = sorted(data['kommun_str'].unique())
    logger.info(f"Unique municipalities in raw data (after cleaning): {len(all_kommun)}")

    # Determine outcome classes from raw target
    unique_labels = sorted(data[config.TARGET_COLUMN].dropna().unique())
    class_names = [str(lbl) for lbl in unique_labels]
    logger.info(f"Outcome classes in raw data: {class_names}")
    logger.info("")

    # Group by municipality and compute normalized counts (proportions)
    logger.info("Grouping by municipality and calculating Y distributions.")
    kommun_dist = (
        data
        .groupby('kommun_str')[config.TARGET_COLUMN]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    # Ensure all municipalities are present
    kommun_dist = kommun_dist.reindex(all_kommun, fill_value=0)

    # Ensure all classes are columns (add missing class columns with 0)
    for lbl in unique_labels:
        if lbl not in kommun_dist.columns:
            kommun_dist[lbl] = 0.0
    kommun_dist = kommun_dist[unique_labels]  # consistent ordering

    # Count observations per municipality
    n_obs = (
        data
        .groupby('kommun_str')[config.TARGET_COLUMN]
        .size()
        .reindex(all_kommun, fill_value=0)
    )

    # Convert to proportions matrix
    probs = kommun_dist.to_numpy(dtype=float)

    # Majority proportion
    majority_prop = probs.max(axis=1)

    # Herfindahl index (sum of squared proportions)
    hhi = (probs ** 2).sum(axis=1)

    # Shannon entropy
    entropy_vals = np.array([calculate_shannon_entropy(row) for row in probs])

    # Build output DataFrame
    out = pd.DataFrame({
        'kommun': all_kommun,
        'n_obs': n_obs.values,
        'entropy': entropy_vals,
        'majority_prop': majority_prop,
        'hhi': hhi,
    })

    # Add per-class proportions with prefix p_
    for j, lbl in enumerate(unique_labels):
        class_name = str(lbl)
        col_name = f"p_{class_name}"
        out[col_name] = probs[:, j]

    logger.info(f"Y distributions calculated for {len(out)} municipalities")
    logger.info(
        f"Entropy summary: min={out['entropy'].min():.3f}, "
        f"max={out['entropy'].max():.3f}, mean={out['entropy'].mean():.3f}"
    )
    logger.info(f"Theoretical max (if 4 classes): {np.log(4):.3f}")
    logger.info("")

    return out


def descriptive_statistics(df, logger):
    """
    Print descriptive statistics for entropy and predictability and
    return them as a DataFrame for export.
    """
    logger.info("=" * 80)
    logger.info("DESCRIPTIVE STATISTICS")
    logger.info("=" * 80)
    logger.info("")

    logger.info("Shannon Entropy (Y Distribution):")
    logger.info(f"  Mean:   {df['entropy'].mean():.4f}")
    logger.info(f"  Median: {df['entropy'].median():.4f}")
    logger.info(f"  Std:    {df['entropy'].std():.4f}")
    logger.info(f"  Min:    {df['entropy'].min():.4f}")
    logger.info(f"  Max:    {df['entropy'].max():.4f}")
    logger.info(f"  Theoretical Max (4 classes): {np.log(4):.4f}")
    logger.info("")

    logger.info("Majority Class Proportion:")
    logger.info(f"  Mean:   {df['majority_prop'].mean():.4f}")
    logger.info(f"  Median: {df['majority_prop'].median():.4f}")
    logger.info(f"  Min:    {df['majority_prop'].min():.4f}")
    logger.info(f"  Max:    {df['majority_prop'].max():.4f}")
    logger.info("")

    logger.info("Herfindahl Index:")
    logger.info(f"  Mean:   {df['hhi'].mean():.4f}")
    logger.info(f"  Median: {df['hhi'].median():.4f}")
    logger.info(f"  Min:    {df['hhi'].min():.4f}")
    logger.info(f"  Max:    {df['hhi'].max():.4f}")
    logger.info("")

    logger.info("Balanced Accuracy:")
    logger.info(f"  Mean:   {df['balanced_accuracy'].mean():.4f}")
    logger.info(f"  Median: {df['balanced_accuracy'].median():.4f}")
    logger.info(f"  Min:    {df['balanced_accuracy'].min():.4f}")
    logger.info(f"  Max:    {df['balanced_accuracy'].max():.4f}")
    logger.info("")

    if 'n_test' in df.columns:
        logger.info("Population Size (test observations):")
        logger.info(f"  Mean:   {df['n_test'].mean():.0f}")
        logger.info(f"  Median: {df['n_test'].median():.0f}")
        logger.info(f"  Min:    {df['n_test'].min():.0f}")
        logger.info(f"  Max:    {df['n_test'].max():.0f}")
        logger.info("")

    # Build a DataFrame with summary statistics for export
    summary_rows = []
    var_meta = [
        ('entropy', 'Shannon entropy (Y distribution)'),
        ('majority_prop', 'Majority class proportion'),
        ('hhi', 'Herfindahl index'),
        ('balanced_accuracy', 'Balanced accuracy'),
        ('n_test', 'Test set size (municipality)'),
        ('n_obs', 'Total observations used for entropy'),
    ]

    for var, label in var_meta:
        if var in df.columns:
            s = df[var].dropna()
            summary_rows.append({
                'variable': var,
                'label': label,
                'n': int(s.shape[0]),
                'mean': float(s.mean()),
                'median': float(s.median()),
                'std': float(s.std()),
                'min': float(s.min()),
                'max': float(s.max()),
            })

    desc_df = pd.DataFrame(summary_rows)
    return desc_df


def correlation_analysis(df, logger):
    """
    Calculate and print correlation matrix.
    Returns the correlation matrix as a DataFrame for export.
    """
    logger.info("=" * 80)
    logger.info("CORRELATION ANALYSIS")
    logger.info("=" * 80)
    logger.info("")

    vars_of_interest = ['balanced_accuracy', 'entropy', 'majority_prop', 'hhi']
    if 'n_test' in df.columns:
        vars_of_interest.append('n_test')

    corr_matrix = df[vars_of_interest].corr()

    logger.info("Correlation Matrix:")
    logger.info("")
    logger.info(corr_matrix.round(3).to_string())
    logger.info("")

    logger.info("Key Correlations with Balanced Accuracy:")
    for var in ['entropy', 'majority_prop', 'hhi', 'n_test']:
        if var in corr_matrix.columns:
            logger.info(
                f"  {var:15s}: {corr_matrix.loc['balanced_accuracy', var]:>7.4f}"
            )
    logger.info("")

    return corr_matrix


def regression_analysis(df, logger):
    """
    Run regression models to test entropy–predictability relationship.

    Uses statsmodels OLS to get coefficients, p-values, and 95% confidence
    intervals, plus R-squared. Returns a DataFrame summarizing all models.
    """

    logger.info("=" * 80)
    logger.info("REGRESSION ANALYSIS (with p-values and 95% CI)")
    logger.info("=" * 80)
    logger.info("")

    rows = []

    # If n_test is available, create log_n_test once here
    if 'n_test' in df.columns:
        df = df.copy()
        df['log_n_test'] = np.log(df['n_test'])

    # ------------------------------------------------------------
    # Model 1: balanced_accuracy ~ entropy
    # ------------------------------------------------------------
    X1 = sm.add_constant(df['entropy'])
    y = df['balanced_accuracy']

    model1 = sm.OLS(y, X1).fit()

    logger.info("Model 1: balanced_accuracy ~ entropy")
    logger.info(model1.summary().as_text())

    rows.append({
        'model': 'model1_entropy',
        'dependent_var': 'balanced_accuracy',
        'predictors': 'entropy',
        'intercept': float(model1.params['const']),
        'coef_entropy': float(model1.params['entropy']),
        'coef_log_n_test': np.nan,
        'p_entropy': float(model1.pvalues['entropy']),
        'p_log_n_test': np.nan,
        'ci_entropy_low': float(model1.conf_int().loc['entropy', 0]),
        'ci_entropy_high': float(model1.conf_int().loc['entropy', 1]),
        'ci_log_low': np.nan,
        'ci_log_high': np.nan,
        'r2': float(model1.rsquared),
        'n_obs': int(model1.nobs),
    })

    # ------------------------------------------------------------
    # If n_test exists: Model 2 and Model 3
    # ------------------------------------------------------------
    if 'n_test' in df.columns:

        # -------------------------
        # Model 2: ~ log(n_test)
        # -------------------------
        X2 = sm.add_constant(df['log_n_test'])
        model2 = sm.OLS(y, X2).fit()

        logger.info("Model 2: balanced_accuracy ~ log(n_test)")
        logger.info(model2.summary().as_text())

        rows.append({
            'model': 'model2_log_n_test',
            'dependent_var': 'balanced_accuracy',
            'predictors': 'log(n_test)',
            'intercept': float(model2.params['const']),
            'coef_entropy': np.nan,
            'coef_log_n_test': float(model2.params['log_n_test']),
            'p_entropy': np.nan,
            'p_log_n_test': float(model2.pvalues['log_n_test']),
            'ci_entropy_low': np.nan,
            'ci_entropy_high': np.nan,
            'ci_log_low': float(model2.conf_int().loc['log_n_test', 0]),
            'ci_log_high': float(model2.conf_int().loc['log_n_test', 1]),
            'r2': float(model2.rsquared),
            'n_obs': int(model2.nobs),
        })

        # -----------------------------------------
        # Model 3: ~ entropy + log(n_test)
        # -----------------------------------------
        X3 = sm.add_constant(df[['entropy', 'log_n_test']])
        model3 = sm.OLS(y, X3).fit()

        logger.info("Model 3: balanced_accuracy ~ entropy + log(n_test)")
        logger.info(model3.summary().as_text())

        rows.append({
            'model': 'model3_entropy_log_n_test',
            'dependent_var': 'balanced_accuracy',
            'predictors': 'entropy + log(n_test)',
            'intercept': float(model3.params['const']),
            'coef_entropy': float(model3.params['entropy']),
            'coef_log_n_test': float(model3.params['log_n_test']),
            'p_entropy': float(model3.pvalues['entropy']),
            'p_log_n_test': float(model3.pvalues['log_n_test']),
            'ci_entropy_low': float(model3.conf_int().loc['entropy', 0]),
            'ci_entropy_high': float(model3.conf_int().loc['entropy', 1]),
            'ci_log_low': float(model3.conf_int().loc['log_n_test', 0]),
            'ci_log_high': float(model3.conf_int().loc['log_n_test', 1]),
            'r2': float(model3.rsquared),
            'n_obs': int(model3.nobs),
        })

    reg_df = pd.DataFrame(rows)
    return reg_df



def identify_outliers(df, logger, n_top=5):
    """
    Identify municipalities that are much easier or harder to predict
    than their entropy would suggest (based on residuals from Model 1).

    Returns
    -------
    df_with_residuals : DataFrame
        Original df plus a 'residual_entropy_model' column.
    """
    logger.info("=" * 80)
    logger.info("OUTLIER MUNICIPALITIES (BASED ON ENTROPY MODEL)")
    logger.info("=" * 80)
    logger.info("")

    # Simple linear model balanced_accuracy ~ entropy
    y = df['balanced_accuracy'].values.reshape(-1, 1)
    X = df['entropy'].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X).flatten()
    residuals = y.flatten() - y_pred

    df = df.copy()
    df['residual_entropy_model'] = residuals

    # Best over-performers
    top = df.nlargest(n_top, 'residual_entropy_model')
    bottom = df.nsmallest(n_top, 'residual_entropy_model')

    logger.info(f"TOP {n_top} OVER-PERFORMERS (more predictable than entropy suggests):")
    logger.info(f"{'Kommun':<8} {'BalAcc':>8} {'Entropy':>8} {'Resid':>8}")
    for _, row in top.iterrows():
        logger.info(
            f"{row['Kommun']:<8} "
            f"{row['balanced_accuracy']:>8.3f} "
            f"{row['entropy']:>8.3f} "
            f"{row['residual_entropy_model']:>8.3f}"
        )
    logger.info("")

    logger.info(f"TOP {n_top} UNDER-PERFORMERS (less predictable than entropy suggests):")
    logger.info(f"{'Kommun':<8} {'BalAcc':>8} {'Entropy':>8} {'Resid':>8}")
    for _, row in bottom.iterrows():
        logger.info(
            f"{row['Kommun']:<8} "
            f"{row['balanced_accuracy']:>8.3f} "
            f"{row['entropy']:>8.3f} "
            f"{row['residual_entropy_model']:>8.3f}"
        )
    logger.info("")

    return df


def save_results(df, config, logger):
    """Save analysis results to CSV."""
    logger.info("=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)
    logger.info("")

    # Collect class proportion columns dynamically
    class_cols = sorted([c for c in df.columns if c.startswith('p_')])

    # Select relevant columns
    base_cols = ['Kommun', 'balanced_accuracy']
    if 'n_test' in df.columns:
        base_cols.append('n_test')
    base_cols += ['entropy', 'majority_prop', 'hhi']
    if 'n_obs' in df.columns:
        base_cols.append('n_obs')

    output_cols = base_cols + class_cols

    output_path = config.TABLES_DIR / 'rq3_entropy_analysis.csv'
    df[output_cols].to_csv(output_path, index=False)

    logger.info(f"Analysis results saved: {output_path}")
    logger.info(f"  Saved {len(df)} municipalities")
    logger.info("")


# %% MAIN EXECUTION

def main():
    """Main function to run entropy analysis on MONA."""
    script_start = time.time()

    logger.info("=" * 80)
    logger.info("RQ3: ENTROPY ANALYSIS")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Analysis plan:")
    logger.info("  1. Load RQ2 municipality predictability results")
    logger.info("  2. Compute Y-distribution entropy per municipality from raw data")
    logger.info("  3. Merge entropy with predictability metrics")
    logger.info("  4. Run descriptive, correlation, and regression analyses")
    logger.info("  5. Save results to CSV (no plots here)")
    logger.info("=" * 80)
    logger.info("")

    try:
        # ------------------------------------------------------------------
        # 1. Load RQ2 municipality predictability results
        # ------------------------------------------------------------------
        logger.info("=" * 80)
        logger.info("LOADING RQ2 PREDICTABILITY RESULTS")
        logger.info("=" * 80)
        logger.info("")

        rq2_path = config.TABLES_DIR / 'rq2_kommun_predictability.csv'
        if not rq2_path.exists():
            raise FileNotFoundError(
                f"RQ2 results not found at {rq2_path}\n"
                "Please run mona_rq2.py first!"
            )

        # Ensure Kommun is read as string to preserve leading zeros
        rq2_df = pd.read_csv(rq2_path, dtype={'Kommun': str})
        logger.info(f"Loaded predictability results for {len(rq2_df)} municipalities")

        # Rename n_observations -> n_test
        if 'n_observations' in rq2_df.columns:
            rq2_df = rq2_df.rename(columns={'n_observations': 'n_test'})

        # Normalize Kommun codes to 4-digit strings
        rq2_df['Kommun'] = rq2_df['Kommun'].apply(normalize_kommun_code)
        logger.info(f"Unique municipalities in RQ2 file: {rq2_df['Kommun'].nunique()}")
        logger.info("")

        # ------------------------------------------------------------------
        # 2. Load raw data and compute Y distributions
        # ------------------------------------------------------------------
        logger.info("LOADING RAW DATA FOR Y DISTRIBUTION CALCULATION")
        data = load_data(config, logger)

        entropy_df = calculate_y_distributions(data, config, logger)
        logger.info(f"Entropy data has {len(entropy_df)} municipalities")
        logger.info("")

        # ------------------------------------------------------------------
        # 3. Merge datasets
        # ------------------------------------------------------------------
        logger.info("=" * 80)
        logger.info("MERGING PREDICTABILITY AND ENTROPY DATA")
        logger.info("=" * 80)
        logger.info("")

        # Normalize kommun codes in entropy_df (should already be 4-digit)
        entropy_df['kommun'] = entropy_df['kommun'].apply(normalize_kommun_code)

        df = rq2_df.merge(entropy_df, left_on='Kommun', right_on='kommun', how='inner')
        logger.info(f"Merged dataset: {len(df)} municipalities")
        logger.info("")

        # Optional: sanity check for any lost municipalities
        lost_from_rq2 = set(rq2_df['Kommun']) - set(df['Kommun'])
        logger.info(f"Municipalities in RQ2 but not in entropy data: {len(lost_from_rq2)}")
        logger.info("")

        # ------------------------------------------------------------------
        # 4. Analyses (all now return DataFrames we can export)
        # ------------------------------------------------------------------
        desc_df = descriptive_statistics(df, logger)
        corr_df = correlation_analysis(df, logger)
        reg_df = regression_analysis(df, logger)
        df_with_resid = identify_outliers(df, logger, n_top=min(5, len(df)))

        # ------------------------------------------------------------------
        # 5. Save main merged dataset
        # ------------------------------------------------------------------
        save_results(df, config, logger)

        # ------------------------------------------------------------------
        # 6. Save additional analysis outputs
        # ------------------------------------------------------------------
        desc_path = config.TABLES_DIR / 'rq3_descriptive_stats.csv'
        desc_df.to_csv(desc_path, index=False)
        logger.info(f"Descriptive statistics saved: {desc_path}")

        corr_path = config.TABLES_DIR / 'rq3_correlation_matrix.csv'
        corr_df.to_csv(corr_path)
        logger.info(f"Correlation matrix saved: {corr_path}")

        reg_path = config.TABLES_DIR / 'rq3_regression_models.csv'
        reg_df.to_csv(reg_path, index=False)
        logger.info(f"Regression model summaries saved: {reg_path}")

        resid_path = config.TABLES_DIR / 'rq3_entropy_model_residuals.csv'
        df_with_resid.to_csv(resid_path, index=False)
        logger.info(f"Entropy model residuals (incl. outliers) saved: {resid_path}")

        # Final summary
        script_time = time.time() - script_start

        logger.info("")
        logger.info("=" * 80)
        logger.info("RQ3 ENTROPY ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"Total runtime: {script_time:.1f} seconds")
        logger.info("")
        logger.info("Outputs saved:")
        logger.info(f"  {config.TABLES_DIR / 'rq3_entropy_analysis.csv'}")
        logger.info(f"  {desc_path}")
        logger.info(f"  {corr_path}")
        logger.info(f"  {reg_path}")
        logger.info(f"  {resid_path}")
        logger.info("")
        logger.info("Key relationships to visualize later (in separate script):")
        logger.info("  - balanced_accuracy vs entropy")
        logger.info("  - balanced_accuracy vs majority_prop")
        logger.info("  - balanced_accuracy vs log(n_test)")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("=" * 80)
        logger.error("ERROR OCCURRED")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}")
        logger.error("")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("=" * 80)
        raise


if __name__ == "__main__":
    main()
