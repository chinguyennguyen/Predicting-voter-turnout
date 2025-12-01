"""
GENERATE DATA SUMMARY - MONA SCRIPT

This script generates summary statistics for the dataset.
It only needs to be run ONCE and takes approximately 2 minutes.

Purpose:
    - Load full, cleaned dataset (7.4 M observations)
    - Generate descriptive statistics for all features
    - Calculate class distributions
    - Save results to data_summary.csv

MONA SETUP INSTRUCTIONS:
1. Copy this entire script to MONA's Spyder
2. Update DATA_PATH and OUTPUT_DIR below
3. Run the script 

Expected Runtime: ~2 minutes
Expected Output: data_summary.csv (~50 KB)
"""

# %% IMPORTS

import pandas as pd
import numpy as np
import time
import logging
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("All imports successful!")
print(f"Pandas version: {pd.__version__}")

# %% MONA PATHS - UPDATE THESE!

# ==============================================================================
# CRITICAL: UPDATE THESE PATHS FOR YOUR MONA ENVIRONMENT
# ==============================================================================

# Path to your Stata data file in MONA
DATA_PATH = r"\\micro.intra\Projekt\P1401$\P1401_Gem\Chi\working_data\chapter3_kommun_election.dta"

# Output directory for results in MONA
OUTPUT_DIR = "outputs"

# ==============================================================================

# %% LOGGING SETUP

def get_logger(name):
    """Create a logger for this script."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

logger = get_logger(__name__)
logger.info("Logger initialized")

# %% CONFIGURATION

class Config:
    """Configuration for data summary."""
    
    # Paths
    DATA_FILE = Path(DATA_PATH)
    OUTPUT_DIR = Path(OUTPUT_DIR)
    TABLES_DIR = OUTPUT_DIR / "tables"
    
    # Data configuration
    TARGET_COLUMN = "y_mun"
    
    NUMERICAL_FEATURES = [
        'fgang18', 'female', 'age_2018', 'foreigner', 
        'schooling_years', 'share_sa', 'share_labor_income',
        'total_income', 'barn0_6', 'barn7_17', 'barn_above18'
    ]
    
    CATEGORICAL_FEATURES = [
        'birth_continent', 'employment_status', 'sector', 
        'marital_status', 'Kommun'
    ]
    
    ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    
    # Class names mapping (will be populated after loading data)
    CLASS_NAMES = None

# Create config instance
config = Config()

# Create output directories
config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Configuration loaded")
logger.info(f"Data path: {config.DATA_FILE}")
logger.info(f"Output directory: {config.OUTPUT_DIR}")

# %% DATA LOADING

def load_data(config, logger):
    """
    Load Stata data file with required columns only.
    """
    logger.info("=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)
    
    if not config.DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {config.DATA_FILE}")
    
    logger.info(f"Reading {config.DATA_FILE.name}...")
    logger.info(f"This may take a few minutes with 8.7M observations...")
    
    start_time = time.time()
    
    # Load only required columns to save memory
    required_cols = [config.TARGET_COLUMN] + config.ALL_FEATURES
    
    try:
        data = pd.read_stata(
            str(config.DATA_FILE),
            columns=required_cols
        )
    except Exception as e:
        logger.warning(f"Could not load with columns parameter: {e}")
        logger.info("Loading entire file instead...")
        data = pd.read_stata(str(config.DATA_FILE))
        data = data[required_cols]
    
    elapsed = time.time() - start_time
    
    logger.info(f"Data loaded successfully in {elapsed:.1f} seconds!")
    logger.info(f"Shape: {data.shape}")
    logger.info(f"Columns: {list(data.columns)}")
    logger.info(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    logger.info("")
    
    return data

# %% SETUP CLASS NAMES

def setup_class_names(data, config, logger):
    """
    Setup class name mappings from target variable.
    """
    logger.info("=" * 80)
    logger.info("SETTING UP CLASS NAMES")
    logger.info("=" * 80)
    
    y = data[config.TARGET_COLUMN]
    
    # Get unique classes and sort them
    unique_classes = sorted(y.unique())
    logger.info(f"Found {len(unique_classes)} unique classes: {unique_classes}")
    
    # Infer class names from the data
    # Assuming classes are labeled as strings or we need to map them
    if y.dtype == 'object' or y.dtype.name == 'category':
        # Classes are already named
        config.CLASS_NAMES = unique_classes
        logger.info(f"Using class names from data: {config.CLASS_NAMES}")
    else:
        # Classes are numeric - need to map to names
        # Based on your project: NN, VV, NV, VN (sorted alphabetically gives: NN, NV, VN, VV)
        # But data might be coded differently, so we'll use generic mapping
        config.CLASS_NAMES = [str(c) for c in unique_classes]
        logger.info(f"Using numeric classes as strings: {config.CLASS_NAMES}")
        logger.info("Note: If these should be NN/VV/NV/VN, update the mapping manually")
    
    logger.info("")
    return y

# %% GENERATE SUMMARY STATISTICS

def generate_summary_statistics(data, y, config, logger):
    """
    Generate comprehensive summary statistics for all features.
    """
    logger.info("=" * 80)
    logger.info("GENERATING SUMMARY STATISTICS")
    logger.info("=" * 80)
    
    summary_stats = []
    n_total = len(data)
    
    logger.info(f"Total observations: {n_total:,}")
    logger.info("")
    
    # Numerical features
    logger.info("Processing numerical features...")
    for col in config.NUMERICAL_FEATURES:
        col_data = data[col]
        n_missing = col_data.isnull().sum()
        
        summary_stats.append({
            'Variable': col,
            'Missing': n_missing,
            'Missing_Pct': (n_missing / n_total) * 100,
            'Unique': col_data.nunique(),
            'Min': col_data.min(),
            'Max': col_data.max(),
            'Median': col_data.median(),
            'Mean': col_data.mean(),
            'SD': col_data.std(),
            'Count': '',
            'Percentage': ''
        })
        
        logger.info(f"  {col}: {col_data.nunique()} unique values, {n_missing:,} missing ({n_missing/n_total*100:.2f}%)")
    
    logger.info("")
    logger.info("Processing categorical features...")
    
    # Categorical features (excluding Kommun for now)
    for col in config.CATEGORICAL_FEATURES:
        if col == 'Kommun':
            # Special handling for Kommun - just count unique
            col_data = data[col]
            n_missing = col_data.isnull().sum()
            n_unique = col_data.nunique()
            
            summary_stats.append({
                'Variable': f'{col} (total)',
                'Missing': n_missing,
                'Missing_Pct': (n_missing / n_total) * 100,
                'Unique': n_unique,
                'Min': '', 'Max': '', 'Median': '', 'Mean': '', 'SD': '',
                'Count': '',
                'Percentage': ''
            })
            
            logger.info(f"  {col}: {n_unique} municipalities, {n_missing:,} missing")
            continue
        
        # Regular categorical features - show distribution
        col_data = data[col]
        n_missing = col_data.isnull().sum()
        value_counts = col_data.value_counts().sort_index()
        
        logger.info(f"  {col}: {len(value_counts)} categories, {n_missing:,} missing")
        
        for category, count in value_counts.items():
            percentage = (count / n_total) * 100
            summary_stats.append({
                'Variable': f'{col} = {category}',
                'Missing': '' if category != value_counts.index[0] else n_missing,
                'Missing_Pct': '' if category != value_counts.index[0] else (n_missing / n_total) * 100,
                'Unique': '', 'Min': '', 'Max': '', 'Median': '', 'Mean': '', 'SD': '',
                'Count': count,
                'Percentage': percentage
            })
    
    logger.info("")
    logger.info("Processing target variable...")
    
    # Target variable distribution
    value_counts = y.value_counts().sort_index()
    
    for class_label, count in value_counts.items():
        percentage = (count / n_total) * 100
        class_name = config.CLASS_NAMES[list(value_counts.index).index(class_label)]
        
        summary_stats.append({
            'Variable': f'{config.TARGET_COLUMN} = {class_name}',
            'Missing': '', 
            'Missing_Pct': '',
            'Unique': '', 'Min': '', 'Max': '', 'Median': '', 'Mean': '', 'SD': '',
            'Count': count,
            'Percentage': percentage
        })
        
        logger.info(f"  Class {class_name}: {count:,} ({percentage:.2f}%)")
    
    logger.info("")
    logger.info("Summary statistics generation complete!")
    logger.info("")
    
    # Create DataFrame with proper column order
    summary_df = pd.DataFrame(summary_stats)
    summary_df = summary_df[[
        'Variable', 'Missing', 'Missing_Pct', 'Unique', 
        'Min', 'Max', 'Median', 'Mean', 'SD', 
        'Count', 'Percentage'
    ]]
    
    return summary_df

# %% SAVE RESULTS

def save_summary(summary_df, config, logger):
    """
    Save summary statistics to CSV.
    """
    logger.info("=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)
    
    output_path = config.TABLES_DIR / 'data_summary.csv'
    
    logger.info(f"Saving to: {output_path}")
    summary_df.to_csv(output_path, index=False)
    
    file_size = output_path.stat().st_size / 1024  # KB
    logger.info(f"File saved successfully! Size: {file_size:.1f} KB")
    logger.info("")

# %% MAIN EXECUTION

def main():
    """
    Main function to generate data summary.
    """
    script_start = time.time()
    
    logger.info("=" * 80)
    logger.info("GENERATE DATA SUMMARY - ONE-TIME OPERATION")
    logger.info("RUN STARTED AT: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 80)
    logger.info("")
    logger.info("This script generates comprehensive summary statistics.")
    logger.info("Expected runtime: ~15 minutes with 8.7M observations")
    logger.info("")
    logger.info("Output: data_summary.csv")
    logger.info("")
    logger.info("This only needs to be run ONCE!")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        # Load data
        logger.info("Step 1/4: Loading data...")
        data = load_data(config, logger)
        
        # Setup class names
        logger.info("Step 2/4: Setting up class names...")
        y = setup_class_names(data, config, logger)
        
        # Generate summary statistics
        logger.info("Step 3/4: Generating summary statistics...")
        summary_df = generate_summary_statistics(data, y, config, logger)
        
        # Save results
        logger.info("Step 4/4: Saving results...")
        save_summary(summary_df, config, logger)
        
        # Final summary
        script_elapsed = time.time() - script_start
        
        logger.info("=" * 80)
        logger.info("DATA SUMMARY GENERATION COMPLETE")
        logger.info("RUN ENDED AT: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"Total runtime: {script_elapsed/60:.1f} minutes")
        logger.info("")
        logger.info("Output saved to:")
        logger.info(f"  {config.TABLES_DIR / 'data_summary.csv'}")
        logger.info("")
        logger.info("✅ This file will be used for reference in your dissertation")
        logger.info("✅ You never need to run this script again!")
        logger.info("")
        logger.info("Next step: Run RQ1a comparison experiments")
        logger.info("  Script: run_rq1a_comparison_mona.py")
        logger.info("  Runtime: ~3-4 hours")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("ERROR OCCURRED")
        logger.error("=" * 80)
        logger.error(f"Error message: {str(e)}")
        logger.error("")
        logger.error("Please check:")
        logger.error("  1. DATA_PATH is correct")
        logger.error("  2. File exists and is accessible")
        logger.error("  3. You have read permissions")
        logger.error("=" * 80)
        raise

# %% RUN SCRIPT

if __name__ == "__main__":
    """
    Execute the main function.
    
    To run in MONA Spyder:
    1. Update DATA_PATH and OUTPUT_DIR at the top
    2. Run 
    3. Wait ~2 minutes

    """
    main()

# %% END OF SCRIPT