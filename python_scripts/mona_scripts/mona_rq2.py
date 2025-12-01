"""
RQ2 GEOGRAPHIC ANALYSIS - MONA SCRIPT

Research Question:
    Which municipalities have more predictable voting behavior?

Approach:
    1. Load best model from RQ1 (rq1_best_model.pkl)
    2. Load test data with municipality information
    3. Make predictions on test set
    4. Calculate performance metrics per municipality
    5. Save results to CSV

Expected Runtime: ~10 mins

MONA SETUP INSTRUCTIONS:
1. Ensure rq1_best_model.pkl exists in models/ directory
2. Copy ALL shared modules
3. Update DATA_PATH and OUTPUT_DIR below
4. Run the script (F5)

"""

# %% IMPORTS

import pandas as pd
import numpy as np
import time
import pickle
from pathlib import Path
from datetime import datetime
import warnings
import gc
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)

# Import shared modules
from config import setup_config
from data_loader import load_data, preprocess_data, split_data
from utils import get_logger

print("All imports successful!")
print("Shared modules loaded successfully!")

# %% MONA PATHS - UPDATE THESE!

# ==============================================================================
# CRITICAL: UPDATE THESE PATHS FOR MONA ENVIRONMENT
# ==============================================================================

DATA_PATH = r"\\micro.intra\Projekt\P1401$\P1401_Gem\Chi\working_data\chapter3_kommun_election.dta"
OUTPUT_DIR = "outputs"

# ==============================================================================

# %% SETUP

# Initialize logger
logger = get_logger(__name__)
logger.info("Logger initialized")

# Setup configuration
config = setup_config(DATA_PATH, OUTPUT_DIR)
logger.info(f"Configuration loaded")
logger.info(f"Data path: {config.DATA_FILE}")
logger.info(f"Output directory: {config.OUTPUT_DIR}")

# Add RQ2-specific config
config.KOMMUN_COLUMN = 'Kommun'

# %% MODEL LOADING

def load_best_model(config, logger):
    """Load the best model from RQ1."""
    logger.info("=" * 80)
    logger.info("LOADING BEST MODEL FROM RQ1")
    logger.info("=" * 80)
    logger.info("")
    
    model_path = config.MODELS_DIR / 'rq1_best_model.pkl'
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Best model not found at {model_path}\n"
            "Please run run_rq1_full_training_mona.py first!"
        )
    
    logger.info(f"Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Model loaded: {type(model).__name__}")
    logger.info("")
    
    return model

# %% MUNICIPALITY ANALYSIS

def calculate_kommun_metrics(y_true, y_pred, kommun_ids, config, logger):
    """Calculate performance metrics for each municipality."""
    logger.info("=" * 80)
    logger.info("CALCULATING METRICS PER MUNICIPALITY")
    logger.info("=" * 80)
    logger.info("")
    
    # Create DataFrame for grouping
    df = pd.DataFrame({
        'kommun': kommun_ids,
        'y_true': y_true,
        'y_pred': y_pred
    })
    
    results = []
    kommuner = sorted(df['kommun'].unique())
    
    logger.info(f"Processing {len(kommuner)} municipalities...")
    logger.info("")
    
    start_time = time.time()
    
    for i, kommun in enumerate(kommuner):
        # Progress update every 50 municipalities
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = (len(kommuner) - i - 1) * avg_time
            logger.info(f"  Processed {i+1}/{len(kommuner)} municipalities "
                       f"(Est. remaining: {remaining/60:.1f} min)")
        
        # Filter data for this municipality
        mask = df['kommun'] == kommun
        kommun_data = df[mask]
        
        y_true_k = kommun_data['y_true'].values
        y_pred_k = kommun_data['y_pred'].values
        
        # Basic info
        n_obs = len(y_true_k)
        
        # Overall metrics
        bal_acc = balanced_accuracy_score(y_true_k, y_pred_k)
        macro_f1 = f1_score(y_true_k, y_pred_k, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true_k, y_pred_k, average='weighted', zero_division=0)
        acc = accuracy_score(y_true_k, y_pred_k)
        
        # Per-class metrics
        classes = sorted(np.unique(y_true_k))
        
        precision_per_class = precision_score(
            y_true_k, y_pred_k, average=None, 
            labels=classes, zero_division=0
        )
        recall_per_class = recall_score(
            y_true_k, y_pred_k, average=None,
            labels=classes, zero_division=0
        )
        f1_per_class = f1_score(
            y_true_k, y_pred_k, average=None,
            labels=classes, zero_division=0
        )
        
        # Build result dictionary
        # Convert Kommun to 4-digit string to preserve leading zeros
        kommun_str = str(int(kommun)).zfill(4) if pd.notna(kommun) else kommun
        
        result = {
            'Kommun': kommun_str,
            'n_observations': n_obs,
            'balanced_accuracy': bal_acc,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'accuracy': acc,
        }
        
        # Add per-class metrics
        for i_class, class_label in enumerate(classes):
            class_name = config.CLASS_NAMES[class_label]
            result[f'f1_{class_name}'] = f1_per_class[i_class]
            result[f'precision_{class_name}'] = precision_per_class[i_class]
            result[f'recall_{class_name}'] = recall_per_class[i_class]
        
        # Add class distribution
        class_counts = pd.Series(y_true_k).value_counts()
        for class_label in range(config.NUM_CLASSES):
            class_name = config.CLASS_NAMES[class_label]
            result[f'n_{class_name}'] = class_counts.get(class_label, 0)
        
        results.append(result)
    
    elapsed = time.time() - start_time
    logger.info(f"  Completed all {len(results)} municipalities in {elapsed/60:.1f} minutes")
    logger.info("")
    
    return pd.DataFrame(results)


def generate_summary_statistics(kommun_df, logger):
    """Generate summary statistics across municipalities."""
    logger.info("=" * 80)
    logger.info("GENERATING SUMMARY STATISTICS")
    logger.info("=" * 80)
    logger.info("")
    
    summary = {
        'metric': [],
        'mean': [],
        'std': [],
        'min': [],
        'q25': [],
        'median': [],
        'q75': [],
        'max': []
    }
    
    # Metrics to summarize
    metrics = [
        'balanced_accuracy',
        'macro_f1',
        'weighted_f1',
        'accuracy',
        'n_observations'
    ]
    
    for metric in metrics:
        if metric in kommun_df.columns:
            values = kommun_df[metric]
            summary['metric'].append(metric)
            summary['mean'].append(values.mean())
            summary['std'].append(values.std())
            summary['min'].append(values.min())
            summary['q25'].append(values.quantile(0.25))
            summary['median'].append(values.median())
            summary['q75'].append(values.quantile(0.75))
            summary['max'].append(values.max())
    
    summary_df = pd.DataFrame(summary)
    
    logger.info("Summary statistics calculated")
    logger.info("")
    
    return summary_df


def print_top_bottom_kommuner(kommun_df, logger, n=10):
    """Print top and bottom municipalities by balanced accuracy."""
    logger.info("=" * 80)
    logger.info(f"TOP {n} MOST PREDICTABLE MUNICIPALITIES")
    logger.info("=" * 80)
    logger.info("")
    
    top = kommun_df.nlargest(n, 'balanced_accuracy')
    
    logger.info(f"{'Kommun':<10} {'N Obs':<10} {'Bal Acc':<12} {'Macro F1':<12}")
    logger.info("-" * 50)
    for _, row in top.iterrows():
        logger.info(f"{row['Kommun']:<10} {row['n_observations']:<10} "
                   f"{row['balanced_accuracy']:<12.4f} {row['macro_f1']:<12.4f}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"BOTTOM {n} LEAST PREDICTABLE MUNICIPALITIES")
    logger.info("=" * 80)
    logger.info("")
    
    bottom = kommun_df.nsmallest(n, 'balanced_accuracy')
    
    logger.info(f"{'Kommun':<10} {'N Obs':<10} {'Bal Acc':<12} {'Macro F1':<12}")
    logger.info("-" * 50)
    for _, row in bottom.iterrows():
        logger.info(f"{row['Kommun']:<10} {row['n_observations']:<10} "
                   f"{row['balanced_accuracy']:<12.4f} {row['macro_f1']:<12.4f}")
    
    logger.info("")


def save_results(kommun_df, summary_df, config, logger):
    """Save results to CSV files."""
    logger.info("=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)
    logger.info("")
    
    # Save municipality-level results
    kommun_path = config.TABLES_DIR / 'rq2_kommun_predictability.csv'
    logger.info(f"Saving municipality results to: {kommun_path}")
    kommun_df.to_csv(kommun_path, index=False)
    logger.info(f"  Saved {len(kommun_df)} municipalities")
    logger.info("")
    
    # Save summary statistics
    summary_path = config.TABLES_DIR / 'rq2_summary_statistics.csv'
    logger.info(f"Saving summary statistics to: {summary_path}")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"  Saved {len(summary_df)} summary metrics")
    logger.info("")

# %% MAIN EXECUTION

def main():
    """Main function to run municipality analysis."""
    script_start = time.time()
    
    logger.info("=" * 80)
    logger.info("RQ2: GEOGRAPHIC ANALYSIS (MUNICIPALITY PREDICTABILITY)")
    logger.info("RUN STARTED AT: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 80)
    logger.info("")
    logger.info("Analysis:")
    logger.info("  Load best model from RQ1")
    logger.info("  Predict on test set")
    logger.info("  Calculate metrics per municipality")
    logger.info("")
    logger.info("Expected runtime: 2-3 hours")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        # Load best model
        model = load_best_model(config, logger)
        
        # Load data
        logger.info("Loading data...")
        data = load_data(config, logger)
        
        # Store Kommun before preprocessing
        logger.info("Extracting municipality information...")
        kommun_series = data[config.KOMMUN_COLUMN].copy()
        logger.info(f"Found {len(kommun_series.unique())} unique municipalities")
        logger.info("")
        
        # Preprocess
        X, y = preprocess_data(data, config, logger)
        
        # Free memory
        del data
        gc.collect()
        logger.info("Raw data freed from memory")
        logger.info("")
        
        # Split data (must match RQ1 split)
        X_train, X_test, y_train, y_test = split_data(X, y, config, logger)
        
        # Get corresponding kommune for test set
        # The split uses the same indices, so we can use them
        test_indices = X_test.index
        kommun_test = kommun_series.loc[test_indices].values
        
        logger.info(f"Test set municipalities: {len(np.unique(kommun_test))}")
        logger.info("")
        
        # Free memory
        del X, y, X_train, y_train, kommun_series
        gc.collect()
        logger.info("Training data freed from memory")
        logger.info("")
        
        # Make predictions
        logger.info("=" * 80)
        logger.info("MAKING PREDICTIONS ON TEST SET")
        logger.info("=" * 80)
        logger.info("")
        
        pred_start = time.time()
        logger.info(f"Predicting {len(X_test):,} test observations...")
        y_pred = model.predict(X_test)
        pred_time = time.time() - pred_start
        
        logger.info(f"Predictions completed in {pred_time/60:.1f} minutes")
        logger.info("")
        
        # Calculate municipality metrics
        kommun_df = calculate_kommun_metrics(
            y_test.values, y_pred, kommun_test, config, logger
        )
        
        # Generate summary statistics
        summary_df = generate_summary_statistics(kommun_df, logger)
        
        # Print summary
        logger.info("OVERALL SUMMARY:")
        logger.info(f"  Total municipalities: {len(kommun_df)}")
        logger.info(f"  Mean balanced accuracy: {kommun_df['balanced_accuracy'].mean():.4f}")
        logger.info(f"  Std balanced accuracy: {kommun_df['balanced_accuracy'].std():.4f}")
        logger.info(f"  Min balanced accuracy: {kommun_df['balanced_accuracy'].min():.4f}")
        logger.info(f"  Max balanced accuracy: {kommun_df['balanced_accuracy'].max():.4f}")
        logger.info("")
        
        # Print top/bottom municipalities
        print_top_bottom_kommuner(kommun_df, logger, n=10)
        
        # Save results
        save_results(kommun_df, summary_df, config, logger)
        
        # Final summary
        script_time = time.time() - script_start
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("RQ2 GEOGRAPHIC ANALYSIS COMPLETE")
        logger.info("RUN ENDED AT: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"Total runtime: {script_time/60:.1f} minutes ({script_time/3600:.1f} hours)")
        logger.info("")
        logger.info("Outputs saved:")
        logger.info(f"  {config.TABLES_DIR / 'rq2_kommun_predictability.csv'}")
        logger.info(f"  {config.TABLES_DIR / 'rq2_summary_statistics.csv'}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Copy results back to local machine")
        logger.info("  2. Create choropleth maps with local visualization scripts")
        logger.info("  3. Analyze patterns: urban vs rural, homogeneous vs diverse")
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


# %% RUN SCRIPT

if __name__ == "__main__":
    """
    Execute the main function.
    
    To run in MONA Spyder:
    1. Ensure rq1_best_model.pkl exists
    2. Ensure ALL shared modules are in the same directory
    3. Update DATA_PATH and OUTPUT_DIR
    4. Run with F5
    5. Wait ~15 mins
    """
    main()

# %% END OF SCRIPT