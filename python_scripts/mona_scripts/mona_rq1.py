"""
RQ1 FULL TRAINING - MONA SCRIPT
Swedish Voter Turnout Prediction Project

Single unified script that runs ALL RQ1 experiments:
- 50k-800k: both stratified & balanced (3 seeds for ≤200k, 1 seed for >200k)
- 1M-4M: stratified only (1 seed each)

Includes Pareto optimization to select best model.

Expected Runtime: ~12-14 hours

MONA SETUP INSTRUCTIONS:
1. Copy ALL shared modules (config.py, data_loader.py, sampler.py, trainer.py, evaluator.py, utils.py)
2. Update DATA_PATH and OUTPUT_DIR below
3. Run the script (F5)

Author: Chi Nguyen
Date: November 2025
"""

# %% IMPORTS

import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import datetime
import warnings
import gc
import pickle
warnings.filterwarnings('ignore')

# Import shared modules
from config import setup_config
from data_loader import load_data, preprocess_data, split_data
from sampler import sample_data
from trainer import train_model, evaluate_model
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
logger.info("Configuration loaded")
logger.info(f"Data path: {config.DATA_FILE}")
logger.info(f"Output directory: {config.OUTPUT_DIR}")

# RQ1 Full Training Configuration
# Comparison zone: 50k-800k (both methods)
config.COMPARISON_SIZES = [50, 100, 200, 400, 800]  # In thousands
config.COMPARISON_METHODS = ['stratified', 'balanced']

# Scaling zone: 1M-4M (stratified only)
config.SCALING_SIZES = [1000, 2000, 3000, 4000]  # In thousands

# Seeds:
#  - 3 seeds for sizes ≤ 200k
#  - 1 seed for sizes > 200k
all_sizes = config.COMPARISON_SIZES + config.SCALING_SIZES
config.SEED_CONFIG = {
    size: [42, 123, 456] if size <= 5000 else [42]
    for size in all_sizes
}

# %% EXPERIMENT RUNNER

def run_all_experiments(X_train, X_test, y_train, y_test, config, logger):
    """Run all RQ1 experiments (comparison + scaling)."""
    logger.info("=" * 80)
    logger.info("RUNNING RQ1 FULL TRAINING")
    logger.info("=" * 80)
    
    results_list = []
    experiment_num = 0

    # Precompute test-set class counts (same for all experiments)
    test_counts_global = y_test.value_counts().sort_index()
    
    # Calculate total experiments
    comparison_exp = sum(
        len(config.SEED_CONFIG[size]) * len(config.COMPARISON_METHODS)
        for size in config.COMPARISON_SIZES
    )
    scaling_exp = sum(
        len(config.SEED_CONFIG[size])
        for size in config.SCALING_SIZES
    )
    total_experiments = comparison_exp + scaling_exp
    
    logger.info(f"Comparison zone experiments: {comparison_exp}")
    logger.info(f"Scaling zone experiments: {scaling_exp}")
    logger.info(f"Total experiments: {total_experiments}")
    logger.info("")
    
    overall_start = time.time()
    
    # PHASE 1: COMPARISON ZONE (50k-800k, both methods)
    logger.info("=" * 80)
    logger.info("PHASE 1: COMPARISON ZONE (50k-800k)")
    logger.info("=" * 80)
    logger.info("")
    
    for train_size_k in config.COMPARISON_SIZES:
        seeds = config.SEED_CONFIG[train_size_k]
        
        for method in config.COMPARISON_METHODS:
            for seed in seeds:
                experiment_num += 1
                
                logger.info("=" * 80)
                logger.info(f"EXPERIMENT {experiment_num}/{total_experiments}")
                logger.info(f"Training size: {train_size_k}k | Method: {method} | Seed: {seed}")
                logger.info("=" * 80)
                
                exp_start = time.time()
                
                # Sample data
                logger.info("Sampling data...")
                sample_start = time.time()
                X_sample, y_sample, sample_weights = sample_data(
                    X_train, y_train, train_size_k, method, seed, config, logger
                )
                sample_time = time.time() - sample_start
                
                logger.info(f"Sample created: {len(X_sample):,} observations")
                logger.info(f"Sampling time: {sample_time:.1f} seconds")
                
                # Log sample class distribution
                sample_counts = y_sample.value_counts().sort_index()
                logger.info("Sample class distribution:")
                for class_label, count in sample_counts.items():
                    pct = count / len(y_sample) * 100
                    logger.info(f"  {class_label}: {count:,} ({pct:.2f}%)")
                
                # Train model
                logger.info("Training model...")
                train_start = time.time()
                model = train_model(X_sample, y_sample, sample_weights, config)
                train_time = time.time() - train_start
                
                logger.info(f"Training time: {train_time:.1f} seconds")
                
                # Evaluate model
                logger.info("Evaluating model...")
                eval_start = time.time()
                metrics = evaluate_model(model, X_test, y_test, config)
                eval_time = time.time() - eval_start
                
                logger.info(f"Evaluation time: {eval_time:.1f} seconds")

                # Per-class train/test counts for this experiment
                per_class_counts = {}
                for idx, class_name in enumerate(config.CLASS_NAMES):
                    train_count = int(sample_counts.get(idx, 0))
                    test_count = int(test_counts_global.get(idx, 0))
                    per_class_counts[f"train_n_{class_name}"] = train_count
                    per_class_counts[f"test_n_{class_name}"] = test_count
                
                # Log results
                logger.info("")
                logger.info("Results:")
                logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
                logger.info(f"  Macro F1: {metrics['macro_f1']:.4f}")
                logger.info(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
                
                exp_time = time.time() - exp_start
                logger.info(f"Experiment time: {exp_time:.1f} seconds")
                
                # Store results
                result = {
                    'experiment_id': experiment_num,
                    'zone': 'comparison',
                    'training_size_k': train_size_k,
                    'method': method,
                    'seed': seed,
                    'sample_time': sample_time,
                    'train_time': train_time,
                    'eval_time': eval_time,
                    'total_time': exp_time,
                    **metrics,
                    **per_class_counts
                }
                results_list.append(result)
                
                # Progress update
                elapsed = time.time() - overall_start
                avg_time = elapsed / experiment_num
                remaining = (total_experiments - experiment_num) * avg_time
                
                logger.info("")
                logger.info(f"Progress: {experiment_num}/{total_experiments} ({experiment_num/total_experiments*100:.1f}%)")
                logger.info(f"Elapsed: {elapsed/60:.1f} min | Est. remaining: {remaining/60:.1f} min")
                logger.info("")
                
                # Free memory
                del X_sample, y_sample, sample_weights, model
                gc.collect()
    
    # PHASE 2: SCALING ZONE (1M-4M, stratified only)
    logger.info("=" * 80)
    logger.info("PHASE 2: SCALING ZONE (1M-4M)")
    logger.info("=" * 80)
    logger.info("")
    
    for train_size_k in config.SCALING_SIZES:
        seeds = config.SEED_CONFIG[train_size_k]
        method = 'stratified'  # Only stratified in scaling zone
        
        for seed in seeds:
            experiment_num += 1
            
            logger.info("=" * 80)
            logger.info(f"EXPERIMENT {experiment_num}/{total_experiments}")
            logger.info(f"Training size: {train_size_k}k | Method: {method} | Seed: {seed}")
            logger.info("=" * 80)
            
            exp_start = time.time()
            
            # Sample data
            logger.info("Sampling data...")
            sample_start = time.time()
            X_sample, y_sample, sample_weights = sample_data(
                X_train, y_train, train_size_k, method, seed, config, logger
            )
            sample_time = time.time() - sample_start
            
            logger.info(f"Sample created: {len(X_sample):,} observations")
            logger.info(f"Sampling time: {sample_time:.1f} seconds")
            
            # Log sample class distribution
            sample_counts = y_sample.value_counts().sort_index()
            logger.info("Sample class distribution:")
            for class_label, count in sample_counts.items():
                pct = count / len(y_sample) * 100
                logger.info(f"  {class_label}: {count:,} ({pct:.2f}%)")
            
            # Train model
            logger.info("Training model...")
            train_start = time.time()
            model = train_model(X_sample, y_sample, sample_weights, config)
            train_time = time.time() - train_start
            
            logger.info(f"Training time: {train_time:.1f} seconds")
            
            # Evaluate model
            logger.info("Evaluating model...")
            eval_start = time.time()
            metrics = evaluate_model(model, X_test, y_test, config)
            eval_time = time.time() - eval_start
            
            logger.info(f"Evaluation time: {eval_time:.1f} seconds")
            
            # Per-class train/test counts for this experiment
            per_class_counts = {}
            for idx, class_name in enumerate(config.CLASS_NAMES):
                train_count = int(sample_counts.get(idx, 0))
                test_count = int(test_counts_global.get(idx, 0))
                per_class_counts[f"train_n_{class_name}"] = train_count
                per_class_counts[f"test_n_{class_name}"] = test_count
            
            # Log results
            logger.info("")
            logger.info("Results:")
            logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            logger.info(f"  Macro F1: {metrics['macro_f1']:.4f}")
            logger.info(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
            
            exp_time = time.time() - exp_start
            logger.info(f"Experiment time: {exp_time:.1f} seconds")
            
            # Store results
            result = {
                'experiment_id': experiment_num,
                'zone': 'scaling',
                'training_size_k': train_size_k,
                'method': method,
                'seed': seed,
                'sample_time': sample_time,
                'train_time': train_time,
                'eval_time': eval_time,
                'total_time': exp_time,
                **metrics,
                **per_class_counts
            }
            results_list.append(result)
            
            # Progress update
            elapsed = time.time() - overall_start
            avg_time = elapsed / experiment_num
            remaining = (total_experiments - experiment_num) * avg_time
            
            logger.info("")
            logger.info(f"Progress: {experiment_num}/{total_experiments} ({experiment_num/total_experiments*100:.1f}%)")
            logger.info(f"Elapsed: {elapsed/60:.1f} min | Est. remaining: {remaining/60:.1f} min")
            logger.info("")
            
            # Free memory
            del X_sample, y_sample, sample_weights, model
            gc.collect()
    
    total_time = time.time() - overall_start
    logger.info("=" * 80)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info(f"Total runtime: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    logger.info("=" * 80)
    logger.info("")
    
    return results_list

# %% PARETO OPTIMIZATION

def identify_pareto_frontier(results_df, logger):
    """Identify Pareto optimal models (accuracy vs training time)."""
    logger.info("=" * 80)
    logger.info("PARETO FRONTIER ANALYSIS")
    logger.info("=" * 80)
    logger.info("")
    
    # Exclude baseline from Pareto analysis
    df_for_pareto = results_df[results_df['zone'] != 'baseline'].copy()
    
    # Average over seeds for each configuration
    summary = df_for_pareto.groupby(['training_size_k', 'method']).agg({
        'balanced_accuracy': 'mean',
        'train_time': 'mean',
        'macro_f1': 'mean'
    }).reset_index()
    
    # Identify Pareto frontier
    pareto_indices = []
    for idx, row in summary.iterrows():
        is_dominated = False
        for _, other in summary.iterrows():
            # Check if 'other' dominates 'row'
            if (other['balanced_accuracy'] >= row['balanced_accuracy'] and 
                other['train_time'] <= row['train_time'] and
                (other['balanced_accuracy'] > row['balanced_accuracy'] or 
                 other['train_time'] < row['train_time'])):
                is_dominated = True
                break
        if not is_dominated:
            pareto_indices.append(idx)
    
    pareto_df = summary.loc[pareto_indices].copy().sort_values('balanced_accuracy', ascending=False)
    
    logger.info(f"Total configurations: {len(summary)}")
    logger.info(f"Pareto optimal configurations: {len(pareto_df)}")
    logger.info("")
    logger.info("Pareto Frontier:")
    for _, row in pareto_df.iterrows():
        logger.info(f"  {row['training_size_k']:>4}k {row['method']:>10} | "
                   f"Acc: {row['balanced_accuracy']:.4f} | "
                   f"Time: {row['train_time']/60:>6.1f} min")
    logger.info("")
    
    return pareto_df


def select_best_model(pareto_df, results_df, logger):
    """Select best model from Pareto frontier."""
    logger.info("=" * 80)
    logger.info("MODEL SELECTION")
    logger.info("=" * 80)
    logger.info("")
    
    # Find max accuracy and second best
    max_acc = pareto_df['balanced_accuracy'].max()
    
    if len(pareto_df) > 1:
        second_best_acc = pareto_df['balanced_accuracy'].nlargest(2).iloc[-1]
    else:
        second_best_acc = max_acc
    
    logger.info(f"Max accuracy on Pareto frontier: {max_acc:.4f}")
    logger.info(f"Second best accuracy: {second_best_acc:.4f}")
    logger.info(f"Difference: {max_acc - second_best_acc:.4f}")
    logger.info("")
    
    # Selection strategy
    if max_acc - second_best_acc >= 0.001:
        # Meaningful difference: pick max accuracy
        logger.info("Selection strategy: MAX ACCURACY (difference >= 0.001)")
        best_config = pareto_df.loc[pareto_df['balanced_accuracy'].idxmax()]
    else:
        # Small difference: pick best accuracy-time tradeoff
        logger.info("Selection strategy: BEST TRADEOFF (difference < 0.001)")
        
        # Normalize metrics
        pareto_df = pareto_df.copy()
        pareto_df['acc_norm'] = (pareto_df['balanced_accuracy'] - pareto_df['balanced_accuracy'].min()) / \
                                (pareto_df['balanced_accuracy'].max() - pareto_df['balanced_accuracy'].min() + 1e-10)
        pareto_df['time_norm'] = 1 - (pareto_df['train_time'] - pareto_df['train_time'].min()) / \
                                     (pareto_df['train_time'].max() - pareto_df['train_time'].min() + 1e-10)
        
        # Composite score (70% accuracy, 30% time)
        pareto_df['composite_score'] = 0.7 * pareto_df['acc_norm'] + 0.3 * pareto_df['time_norm']
        best_config = pareto_df.loc[pareto_df['composite_score'].idxmax()]
    
    logger.info("")
    logger.info("SELECTED BEST MODEL:")
    logger.info(f"  Training size: {best_config['training_size_k']}k")
    logger.info(f"  Method: {best_config['method']}")
    logger.info(f"  Balanced Accuracy: {best_config['balanced_accuracy']:.4f}")
    logger.info(f"  Macro F1: {best_config['macro_f1']:.4f}")
    logger.info(f"  Training time: {best_config['train_time']/60:.1f} minutes")
    logger.info("")
    
    # Find all experiments with this configuration (for seed averaging)
    config_mask = (
        (results_df['training_size_k'] == best_config['training_size_k']) &
        (results_df['method'] == best_config['method'])
    )
    config_experiments = results_df[config_mask]
    
    logger.info("Experiments for selected configuration:")
    logger.info(config_experiments[['experiment_id', 'seed', 'balanced_accuracy', 'macro_f1', 'train_time']])
    logger.info("")
    
    # Use the experiment with best balanced accuracy among seeds
    best_experiment_idx = config_experiments['balanced_accuracy'].idxmax()
    best_experiment = results_df.loc[best_experiment_idx]
    
    logger.info(f"Using experiment with seed {best_experiment['seed']}")
    logger.info("")
    
    return best_config, best_experiment


def train_best_model_final(X_train, y_train, best_experiment, config, logger):
    """Re-train the best model configuration on full data for saving."""
    logger.info("=" * 80)
    logger.info("TRAINING FINAL BEST MODEL")
    logger.info("=" * 80)
    logger.info("")
    
    train_size_k = best_experiment['training_size_k']
    method = best_experiment['method']
    seed = int(best_experiment['seed'])
    
    logger.info(f"Best configuration:")
    logger.info(f"  Training size: {train_size_k}k")
    logger.info(f"  Method: {method}")
    logger.info(f"  Seed: {seed}")
    logger.info("")
    
    # Sample data again with best configuration
    logger.info("Sampling data for final model...")
    sample_start = time.time()
    X_sample, y_sample, sample_weights = sample_data(
        X_train, y_train, train_size_k, method, seed, config, logger
    )
    sample_time = time.time() - sample_start
    
    logger.info(f"Final training sample: {len(X_sample):,} observations")
    logger.info(f"Sampling time: {sample_time:.1f} seconds")
    
    # Train model
    logger.info("Training final model...")
    train_start = time.time()
    final_model = train_model(X_sample, y_sample, sample_weights, config)
    train_time = time.time() - train_start
    
    logger.info(f"Final model training time: {train_time:.1f} seconds")
    logger.info("")
    
    return final_model


def compute_majority_baseline(X_train, X_test, y_train, y_test, config, logger):
    """Compute a simple majority-class baseline and corresponding metrics."""
    logger.info("=" * 80)
    logger.info("COMPUTING MAJORITY-CLASS BASELINE")
    logger.info("=" * 80)
    logger.info("")
    
    # Train-set class distribution (full training data)
    train_counts_full = y_train.value_counts().sort_index()
    test_counts_full = y_test.value_counts().sort_index()
    majority_class = train_counts_full.idxmax()
    majority_count = train_counts_full.max()
    majority_pct = majority_count / len(y_train) * 100
    
    logger.info("Training set class distribution:")
    for class_label, count in train_counts_full.items():
        pct = count / len(y_train) * 100
        logger.info(f"  {class_label}: {count:,} ({pct:.2f}%)")
    logger.info("")
    logger.info(
        f"Majority class: {majority_class} "
        f"({majority_count:,} observations, {majority_pct:.2f}%)"
    )
    logger.info("")
    
    class MajorityClassModel:
        def __init__(self, constant_class):
            self.constant_class = constant_class
        
        def predict(self, X):
            return np.full(shape=(len(X),), fill_value=self.constant_class)
    
    model = MajorityClassModel(majority_class)
    
    eval_start = time.time()
    metrics = evaluate_model(model, X_test, y_test, config)
    eval_time = time.time() - eval_start
    
    # Per-class train/test counts (using full train/test sets)
    per_class_counts = {}
    for idx, class_name in enumerate(config.CLASS_NAMES):
        train_count = int(train_counts_full.get(idx, 0))
        test_count = int(test_counts_full.get(idx, 0))
        per_class_counts[f"train_n_{class_name}"] = train_count
        per_class_counts[f"test_n_{class_name}"] = test_count
    
    result = {
        'experiment_id': 0,
        'zone': 'baseline',
        'training_size_k': len(y_train) / 1000.0,
        'method': 'majority_class',
        'seed': None,
        'sample_time': 0.0,
        'train_time': 0.0,
        'eval_time': eval_time,
        'total_time': eval_time,
        **metrics,
        **per_class_counts
    }
    
    logger.info("Baseline metrics (majority-class predictor):")
    logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"  Macro F1: {metrics['macro_f1']:.4f}")
    logger.info(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    logger.info("")
    
    return result


def create_results_dataframe(results_list):
    """Create DataFrame from results list."""
    df = pd.DataFrame(results_list)
    
    # Reorder columns
    col_order = [
        'experiment_id', 'zone', 'training_size_k', 'method', 'seed',
        'balanced_accuracy', 'macro_f1', 'weighted_f1'
    ]
    
    # Add per-class metrics
    class_metrics = [col for col in df.columns if any(
        col.startswith(prefix) for prefix in ['f1_', 'precision_', 'recall_']
    )]
    col_order.extend(sorted(class_metrics))
    
    # Add per-class counts if present
    count_cols = [col for col in df.columns
                  if col.startswith('train_n_') or col.startswith('test_n_')]
    col_order.extend(sorted(count_cols))
    
    # Add timing columns
    time_cols = ['sample_time', 'train_time', 'eval_time', 'total_time']
    col_order.extend(time_cols)
    
    df = df[col_order]
    
    return df


def save_results(results_df, pareto_df, best_model, config, logger):
    """Save all results and best model."""
    logger.info("=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)
    logger.info("")
    
    # Save detailed results
    detailed_path = config.TABLES_DIR / 'rq1_all_experiments.csv'
    logger.info(f"Saving all experiments to: {detailed_path}")
    results_df.to_csv(detailed_path, index=False)
    logger.info(f"  Saved {len(results_df)} experiments")
    logger.info("")
    
    # Save Pareto frontier
    pareto_path = config.TABLES_DIR / 'rq1_pareto_frontier.csv'
    logger.info(f"Saving Pareto frontier to: {pareto_path}")
    pareto_df.to_csv(pareto_path, index=False)
    logger.info(f"  Saved {len(pareto_df)} Pareto optimal configurations")
    logger.info("")
    
    # Save best model
    model_path = config.MODELS_DIR / 'rq1_best_model.pkl'
    logger.info(f"Saving best model to: {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    logger.info("  Model saved successfully")
    logger.info("")


def print_final_summary(results_df, logger):
    """Print final summary of all experiments."""
    logger.info("=" * 80)
    logger.info("RQ1 FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info("")
    
    logger.info("EXPERIMENT OVERVIEW:")
    logger.info(f"  Total experiments (incl. baseline): {len(results_df)}")
    comparison_exp = results_df[results_df['zone'] == 'comparison']
    scaling_exp = results_df[results_df['zone'] == 'scaling']
    logger.info(f"  Comparison zone (50k-800k): {len(comparison_exp)} experiments")
    logger.info(f"  Scaling zone (1M-4M): {len(scaling_exp)} experiments")
    logger.info("")
    
    logger.info("TIMING SUMMARY:")
    total_time = results_df['total_time'].sum()
    logger.info(f"  Total experiment time: {total_time/3600:.1f} hours")
    logger.info(f"  Average per experiment: {results_df['total_time'].mean()/60:.1f} minutes")
    logger.info("")
    
    # Best in each zone (excluding baseline)
    logger.info("BEST PERFORMANCE BY ZONE:")
    
    if not comparison_exp.empty:
        comparison_best = comparison_exp.loc[comparison_exp['balanced_accuracy'].idxmax()]
        logger.info(f"  Comparison zone: {comparison_best['training_size_k']}k {comparison_best['method']}")
        logger.info(f"    Balanced Accuracy: {comparison_best['balanced_accuracy']:.4f}")
        logger.info("")
    
    if not scaling_exp.empty:
        scaling_best = scaling_exp.loc[scaling_exp['balanced_accuracy'].idxmax()]
        logger.info(f"  Scaling zone: {scaling_best['training_size_k']}k {scaling_best['method']}")
        logger.info(f"    Balanced Accuracy: {scaling_best['balanced_accuracy']:.4f}")
        logger.info("")
    
    logger.info("=" * 80)

# %% MAIN EXECUTION

def main():
    """Main function to run full RQ1 training."""
    script_start = time.time()
    
    logger.info("=" * 80)
    logger.info("RQ1: FULL TRAINING (UNIFIED VERSION)")
    logger.info("RUN STARTED AT: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 80)
    logger.info("")
    logger.info("Experiment Design:")
    logger.info(f"  Comparison zone: {config.COMPARISON_SIZES} (in thousands)")
    logger.info(f"    Methods: {config.COMPARISON_METHODS}")
    logger.info(f"  Scaling zone: {config.SCALING_SIZES} (in thousands)")
    logger.info(f"    Methods: ['stratified']")
    logger.info("")
    total_exp = (
        sum(len(config.SEED_CONFIG[size]) * len(config.COMPARISON_METHODS)
            for size in config.COMPARISON_SIZES)
        + sum(len(config.SEED_CONFIG[size]) for size in config.SCALING_SIZES)
    )
    logger.info(f"  Total experiments (excluding baseline): {total_exp}")
    logger.info("")
    logger.info("Expected runtime: 10-12 hours")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        # Load data
        data = load_data(config, logger)
        
        # Preprocess
        X, y = preprocess_data(data, config, logger)
        
        # Free memory
        del data
        gc.collect()
        logger.info("Raw data freed from memory")
        logger.info("")
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y, config, logger)
        
        # Free memory
        del X, y
        gc.collect()
        logger.info("Full data freed from memory")
        logger.info("")
        
        # Majority-class baseline
        baseline_result = compute_majority_baseline(
            X_train, X_test, y_train, y_test, config, logger
        )
        
        # Run all experiments
        results_list = run_all_experiments(X_train, X_test, y_train, y_test, config, logger)
        
        # Create results DataFrame (baseline + experiments)
        logger.info("Processing results...")
        all_results = [baseline_result] + results_list
        results_df = create_results_dataframe(all_results)
        
        # Pareto analysis
        pareto_df = identify_pareto_frontier(results_df, logger)
        
        # Select best model
        best_config, best_experiment = select_best_model(pareto_df, results_df, logger)
        
        # Train final best model
        best_model = train_best_model_final(X_train, y_train, best_experiment, config, logger)
        
        # Save everything
        save_results(results_df, pareto_df, best_model, config, logger)
        
        # Print summary
        print_final_summary(results_df, logger)
        
        # Final summary
        script_time = time.time() - script_start
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("RQ1 FULL TRAINING COMPLETE")
        logger.info("RUN ENDED AT: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"Total runtime: {script_time/60:.1f} minutes ({script_time/3600:.1f} hours)")
        logger.info("")
        logger.info("Outputs saved:")
        logger.info(f"  {config.TABLES_DIR / 'rq1_all_experiments.csv'}")
        logger.info(f"  {config.TABLES_DIR / 'rq1_pareto_frontier.csv'}")
        logger.info(f"  {config.MODELS_DIR / 'rq1_best_model.pkl'}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Copy results back to local machine")
        logger.info("  2. Run analyze_rq1_results.py to create visualizations")
        logger.info("  3. Optional: Run refinement if Pareto suggests improvement")
        logger.info("  4. Run RQ2 feature importance and geographic analysis")
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
    1. Ensure ALL shared modules are in the same directory
    2. Update DATA_PATH and OUTPUT_DIR
    3. Run with F5
    4. Wait ~12-14 hours
    """
    main()

# %% END OF SCRIPT
