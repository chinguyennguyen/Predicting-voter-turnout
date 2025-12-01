"""
MUNICIPALITY-LEVEL EXPLANATORY DATASET - DEVELOPMENT SCRIPT

Goal:
    Build a municipality-level explanatory dataset by aggregating
    individual-level variables to the municipal level.

Y-structure (per municipality), based on y_mun ∈ {VV, NV, VN, NN}:
    - y_share__VV, y_share__NV, y_share__VN, y_share__NN (where present)
    - class_imbalance: max class share
    - y_entropy: Shannon entropy of the class distribution
    - n_obs: sample size

X-structure (numerical features):
    For each numerical feature in config.NUMERICAL_FEATURES:
        - <col>__mean
        - <col>__var        (population variance, ddof=0)
        - <col>__gini       (Gini coefficient)

Categorical X:
    For each categorical feature in config.CATEGORICAL_FEATURES, excluding 'Kommun':
        - <col>__share__<category>  (category shares)
        - <col>__entropy            (Shannon entropy of the category distribution)

Notes:
    - Uses the same config / data path conventions as RQ2 and RQ3 dev scripts.
    - Municipality codes are stored as 4-digit strings in column 'kommun'
      (e.g., "0123"), matching RQ2/RQ3 conventions.

"""

# %% IMPORTS

import pandas as pd
import numpy as np
import time
from pathlib import Path
import warnings
import sys
import traceback

warnings.filterwarnings("ignore")

# Make python_scripts/ visible for mona_scripts
THIS_DIR = Path(__file__).resolve().parent
PYTHON_SCRIPTS_DIR = THIS_DIR.parent  # .../python_scripts

if str(PYTHON_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_SCRIPTS_DIR))

# Shared MONA modules (same pattern as rq2_analyses.py / rq3_analyses.py)
from mona_scripts.config import setup_config
from mona_scripts.data_loader import load_data
from mona_scripts.utils import get_logger


# %% CONFIGURATION

# Paths (local development, mirroring RQ2/RQ3)
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

# Municipality column name (for clarity / consistency)
config.KOMMUN_COLUMN = "Kommun"


# %% HELPER FUNCTIONS

def calculate_shannon_entropy(proportions):
    """Compute Shannon entropy given a vector of proportions."""
    p = np.array(proportions, dtype=float)
    p = p[p > 0]  # avoid log(0)
    return float(-np.sum(p * np.log(p))) if len(p) > 0 else 0.0


def gini_coefficient(x):
    """
    Compute Gini coefficient for a 1D array-like x.

    Handles non-positive values by shifting if necessary.
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan

    # Shift if there are negative values
    min_val = arr.min()
    if min_val < 0:
        arr = arr - min_val

    # All zeros -> Gini = 0
    if np.all(arr == 0):
        return 0.0

    arr = np.sort(arr)
    n = arr.size
    cum = np.cumsum(arr)
    g = (2 * np.sum((np.arange(1, n + 1) * arr)) / (n * np.sum(arr))) - (n + 1) / n
    return float(g)


# %% CORE AGGREGATION

def build_municipality_explanatory_dataset(data, config, logger):
    """
    Aggregate individual-level data to municipality-level explanatory dataset.

    Parameters
    ----------
    data : DataFrame
        Individual-level data with y_mun, X-features, and Kommun.
    config : BaseConfig
        Shared configuration with NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN.
    logger : Logger

    Returns
    -------
    DataFrame
        One row per municipality with Y-structure, numerical X-structure,
        and categorical X-structure features.
    """
    logger.info("=" * 80)
    logger.info("BUILDING MUNICIPALITY-LEVEL EXPLANATORY DATASET")
    logger.info("=" * 80)
    logger.info("")

    target_col = getattr(config, "TARGET_COLUMN", "y_mun")
    kommun_col = getattr(config, "KOMMUN_COLUMN", "Kommun")

    # Drop rows without municipality info
    data = data.copy()
    if kommun_col not in data.columns:
        raise KeyError(f"Municipality column '{kommun_col}' not found in data.")

    data = data.loc[data[kommun_col].notnull()].copy()
    logger.info(f"Data after dropping missing '{kommun_col}': {data.shape[0]:,} rows")

    kommun_values = sorted(data[kommun_col].unique())
    logger.info(f"Number of municipalities: {len(kommun_values)}")
    logger.info("")

    # Separate numerical and categorical features
    num_features = list(config.NUMERICAL_FEATURES)
    # Exclude municipality itself from categorical features
    cat_features = [c for c in config.CATEGORICAL_FEATURES if c != kommun_col]

    results = []
    start_time = time.time()

    for i, k in enumerate(kommun_values):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = (len(kommun_values) - i - 1) * avg_time
            logger.info(
                f"  Processed {i+1}/{len(kommun_values)} municipalities "
                f"(Est. remaining: {remaining/60:.1f} min)"
            )

        sub = data.loc[data[kommun_col] == k].copy()
        n_obs = len(sub)

        # Convert municipality code to 4-digit string (matches RQ2/RQ3)
        try:
            kommun_str = str(int(k)).zfill(4)
        except (ValueError, TypeError):
            kommun_str = str(k)

        row = {
            "kommun": kommun_str,
            "n_obs": n_obs,
        }

        # ---------- Y-structure: VV / NV / VN / NN distribution ----------
        if target_col not in sub.columns:
            raise KeyError(f"Target column '{target_col}' not found in data.")

        y_vals = sub[target_col].astype(str)
        y_dist = y_vals.value_counts(normalize=True)

        # Class shares: y_share__VV, etc. (for however many classes appear)
        for cls_label, share in y_dist.items():
            cls_suffix = str(cls_label).replace(" ", "_")
            row[f"y_share__{cls_suffix}"] = float(share)

        if not y_dist.empty:
            row["class_imbalance"] = float(y_dist.max())
            row["y_entropy"] = calculate_shannon_entropy(y_dist.values)
        else:
            row["class_imbalance"] = np.nan
            row["y_entropy"] = np.nan

        # ---------- Numerical X-structure ----------
        for col in num_features:
            if col not in sub.columns:
                logger.warning(
                    f"Numerical column '{col}' not found in data; skipping for municipality {kommun_str}."
                )
                continue

            x = sub[col].astype(float)
            x_valid = x[x.notnull()]

            if x_valid.size > 0:
                row[f"{col}__mean"] = float(x_valid.mean())
                row[f"{col}__var"] = float(x_valid.var(ddof=0))
                row[f"{col}__gini"] = gini_coefficient(x_valid.values)
            else:
                row[f"{col}__mean"] = np.nan
                row[f"{col}__var"] = np.nan
                row[f"{col}__gini"] = np.nan

        # ---------- Categorical X-structure ----------
        for col in cat_features:
            if col not in sub.columns:
                logger.warning(
                    f"Categorical column '{col}' not found in data; skipping for municipality {kommun_str}."
                )
                continue

            cat_series = sub[col].astype(str)
            cat_dist = cat_series.value_counts(normalize=True)

            # Category shares: <col>__share__<category>
            for cat_value, share in cat_dist.items():
                cat_suffix = (
                    str(cat_value)
                    .strip()
                    .replace(" ", "_")
                    .replace("/", "_")
                    .replace("-", "_")
                )
                row[f"{col}__share__{cat_suffix}"] = float(share)

            # Entropy of category distribution
            row[f"{col}__entropy"] = calculate_shannon_entropy(cat_dist.values)

        results.append(row)

    elapsed = time.time() - start_time
    logger.info("")
    logger.info(
        f"Completed municipality-level aggregation in {elapsed/60:.1f} minutes."
    )

    df = pd.DataFrame(results)

    # Sort by kommun code
    df = df.sort_values("kommun").reset_index(drop=True)

    logger.info(f"Final municipality-level dataset shape: {df.shape}")
    return df


def save_municipality_explanatory_dataset(df, config, logger):
    """Save municipality-level explanatory dataset to CSV."""
    logger.info("=" * 80)
    logger.info("SAVING MUNICIPALITY-LEVEL EXPLANATORY DATASET")
    logger.info("=" * 80)
    logger.info("")

    out_path = config.TABLES_DIR / "municipality_explanatory_dataset.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved municipality-level explanatory dataset to: {out_path}")
    logger.info(f"Number of municipalities: {len(df)}")
    logger.info("")


# %% MAIN SCRIPT

def main():
    script_start = time.time()

    logger.info("=" * 80)
    logger.info("MUNICIPALITY-LEVEL EXPLANATORY DATASET - START")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Steps:")
    logger.info("  1. Load individual-level data")
    logger.info("  2. Aggregate Y-structure (VV/NV/VN/NN distribution)")
    logger.info("  3. Aggregate X-structure (numerical & categorical)")
    logger.info("  4. Save dataset to CSV")
    logger.info("=" * 80)
    logger.info("")

    try:
        # 1. Load raw individual-level data (with y_mun + features)
        data = load_data(config, logger)

        # 2–3. Build explanatory dataset
        mun_df = build_municipality_explanatory_dataset(data, config, logger)

        # 4. Save to CSV
        save_municipality_explanatory_dataset(mun_df, config, logger)

        total_time = time.time() - script_start
        logger.info("=" * 80)
        logger.info("MUNICIPALITY-LEVEL EXPLANATORY DATASET - COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total runtime: {total_time/60:.1f} minutes")
        logger.info("")

    except Exception as e:
        logger.error("=" * 80)
        logger.error("ERROR OCCURRED")
        logger.error("=" * 80)
        logger.error(str(e))
        logger.error(traceback.format_exc())
        # Re-raise so failures are visible if running from CLI
        raise


if __name__ == "__main__":
    main()
