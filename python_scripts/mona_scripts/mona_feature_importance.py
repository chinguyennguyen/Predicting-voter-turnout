"""
RQ2 FEATURE IMPORTANCE - MONA SCRIPT

Analyzes which covariates are most important for predicting voting behavior transitions. Generates confusion matrix from the best model.

Uses XGBoost built-in feature importance metrics:
- Gain: Average gain when feature is used for splitting
- Weight: Number of times feature is used in trees
- Cover: Average coverage of feature across splits

Expected Runtime: ~5-10 minutes

MONA SETUP INSTRUCTIONS:
1. Ensure rq1_best_model.pkl exists in models/ directory
2. Copy ALL shared modules
3. Update DATA_PATH and OUTPUT_DIR below
4. Run the script (F5)
5. Wait ~5 minutes
...
"""

# %% IMPORTS

import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import datetime
import warnings
import pickle
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')

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


# %% LOAD BEST MODEL

def load_best_model(config, logger):
    """Load the best model from RQ1."""
    logger.info("=" * 80)
    logger.info("LOADING BEST MODEL")
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
    
    logger.info("Model loaded successfully")
    logger.info("")
    
    return model

# %% FEATURE IMPORTANCE ANALYSIS

def extract_feature_importance(model, feature_names, logger):
    """Extract all three importance types from XGBoost model."""
    logger.info("=" * 80)
    logger.info("EXTRACTING FEATURE IMPORTANCE")
    logger.info("=" * 80)
    logger.info("")
    
    importance_types = ['gain', 'weight', 'cover']
    importance_dict = {}
    
    for imp_type in importance_types:
        logger.info(f"Computing {imp_type} importance...")
        
        # Get importance scores
        importance_scores = model.get_booster().get_score(importance_type=imp_type)
        
        # Create DataFrame
        imp_df = pd.DataFrame([
            {'feature': feat, imp_type: score}
            for feat, score in importance_scores.items()
        ])
        
        # Map feature indices to names if needed
        if imp_df['feature'].dtype != 'object':
            # Features are stored as f0, f1, etc.
            feature_map = {f'f{i}': name for i, name in enumerate(feature_names)}
            imp_df['feature'] = imp_df['feature'].map(feature_map)
        
        # Sort by importance
        imp_df = imp_df.sort_values(imp_type, ascending=False).reset_index(drop=True)
        
        importance_dict[imp_type] = imp_df
        
        logger.info(f"  Found {len(imp_df)} features with {imp_type} scores")
    
    logger.info("")
    return importance_dict


def merge_importance_scores(importance_dict, logger):
    """Merge all importance types into single DataFrame."""
    logger.info("Merging importance scores...")
    
    # Start with gain (most important metric)
    merged = importance_dict['gain'].copy()
    
    # Add weight and cover
    for imp_type in ['weight', 'cover']:
        merged = merged.merge(
            importance_dict[imp_type],
            on='feature',
            how='outer'
        )
    
    # Fill NaN with 0 (features not used have 0 importance)
    merged = merged.fillna(0)
    
    # Add normalized versions
    for imp_type in ['gain', 'weight', 'cover']:
        total = merged[imp_type].sum()
        if total > 0:
            merged[f'{imp_type}_normalized'] = merged[imp_type] / total
    
    # Sort by gain (primary metric)
    merged = merged.sort_values('gain', ascending=False).reset_index(drop=True)
    
    # Add rank
    merged.insert(0, 'rank', range(1, len(merged) + 1))
    
    logger.info(f"Merged importance for {len(merged)} features")
    logger.info("")
    
    return merged


def print_top_features(importance_df, logger, top_n=20):
    """Print top N most important features."""
    logger.info("=" * 80)
    logger.info(f"TOP {top_n} MOST IMPORTANT FEATURES (by Gain)")
    logger.info("=" * 80)
    logger.info("")
    
    logger.info(f"{'Rank':<6} {'Feature':<30} {'Gain':<12} {'Weight':<10} {'Cover':<12}")
    logger.info("-" * 80)
    
    for idx, row in importance_df.head(top_n).iterrows():
        logger.info(f"{row['rank']:<6} {row['feature']:<30} "
                   f"{row['gain']:>11.1f} {row['weight']:>9.0f} {row['cover']:>11.1f}")
    
    logger.info("")


def print_importance_summary(importance_df, logger):
    """Print summary statistics about feature importance."""
    logger.info("=" * 80)
    logger.info("FEATURE IMPORTANCE SUMMARY")
    logger.info("=" * 80)
    logger.info("")
    
    logger.info(f"Total features: {len(importance_df)}")
    logger.info("")
    
    # Top N concentration
    for n in [5, 10, 20]:
        if len(importance_df) >= n:
            top_gain = importance_df.head(n)['gain_normalized'].sum()
            logger.info(f"Top {n:>2} features account for {top_gain*100:>5.1f}% of total gain")
    
    logger.info("")
    
    # Feature usage
    used_in_model = (importance_df['weight'] > 0).sum()
    logger.info(f"Features used in model: {used_in_model} ({used_in_model/len(importance_df)*100:.1f}%)")
    
    logger.info("")


def save_importance_results(importance_df, config, logger):
    """Save feature importance results."""
    logger.info("=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)
    logger.info("")
    
    output_path = config.TABLES_DIR / 'rq2_feature_importance.csv'
    logger.info(f"Saving feature importance to: {output_path}")
    importance_df.to_csv(output_path, index=False)
    logger.info(f"  Saved {len(importance_df)} features")
    logger.info("")

def compute_and_save_confusion_matrix(model, X, y, config, logger):
    """
    Compute confusion matrix for the best model on the standard
    train/test split (same split configuration as RQ1) and save it to CSV.
    """
    logger.info("=" * 80)
    logger.info("RQ2: CONFUSION MATRIX FOR BEST MODEL")
    logger.info("=" * 80)
    logger.info("")
    
    logger.info("Creating train/test split (same configuration as RQ1)...")
    X_train, X_test, y_train, y_test = split_data(X, y, config, logger)
    logger.info(f"Test set size: {len(y_test):,} observations")
    logger.info("Predicting on test set...")
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Ensure labels are ordered consistently with CLASS_NAMES
    labels = list(range(config.NUM_CLASSES))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    # Build a nicely labeled DataFrame
    index_labels = [f"true_{cls}" for cls in config.CLASS_NAMES]
    column_labels = [f"pred_{cls}" for cls in config.CLASS_NAMES]
    
    cm_df = pd.DataFrame(cm, index=index_labels, columns=column_labels)
    
    # Save to tables directory
    output_path = config.TABLES_DIR / "rq2_best_model_confusion_matrix.csv"
    logger.info(f"Saving confusion matrix to: {output_path}")
    cm_df.to_csv(output_path)
    logger.info("Confusion matrix saved successfully")
    logger.info("")


# %% MAIN EXECUTION

def main():
    """Main function to run feature importance analysis."""
    script_start = time.time()
    
    logger.info("=" * 80)
    logger.info("RQ2: FEATURE IMPORTANCE ANALYSIS")
    logger.info("RUN STARTED AT: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 80)
    logger.info("")
    logger.info("Analysis:")
    logger.info("  Using XGBoost built-in feature importance")
    logger.info("  Metrics: Gain, Weight, Cover")
    logger.info("")
    logger.info("Expected runtime: 5-10 minutes")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        # 1. Load best model
        model = load_best_model(config, logger)
        
        # 2. Load and preprocess data
        logger.info("Loading and preprocessing data...")
        data = load_data(config, logger)
        X, y = preprocess_data(data, config, logger)
        feature_names = X.columns.tolist()
        logger.info(f"Found {len(feature_names)} features")
        logger.info("")
        
        # 3. Extract and merge feature importance
        importance_dict = extract_feature_importance(model, feature_names, logger)
        importance_df = merge_importance_scores(importance_dict, logger)
        
        # 4. Report importance results
        print_top_features(importance_df, logger, top_n=20)
        print_importance_summary(importance_df, logger)
        save_importance_results(importance_df, config, logger)
        
        # 5. Confusion matrix for the same best model (on standard test split)
        compute_and_save_confusion_matrix(model, X, y, config, logger)
        
        # 6. Final summary
        script_time = time.time() - script_start
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("RQ2 FEATURE IMPORTANCE + CONFUSION MATRIX COMPLETE")
        logger.info("RUN ENDED AT: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"Total runtime: {script_time:.1f} seconds ({script_time/60:.1f} minutes)")
        logger.info("")
        logger.info("Outputs saved:")
        logger.info(f"  {config.TABLES_DIR / 'rq2_feature_importance.csv'}")
        logger.info(f"  {config.TABLES_DIR / 'rq2_best_model_confusion_matrix.csv'}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Copy results back to local machine")
        logger.info("  2. Create visualizations (bar charts, etc.)")
        logger.info("  3. Run RQ2 municipality analysis: run_rq2_municipality_mona.py")
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

    """
    main()

# %% END OF SCRIPT