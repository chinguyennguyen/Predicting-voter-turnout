"""
SHARED SAMPLER 

This module contains sampling strategies.
Includes both stratified and balanced sampling methods.

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_sample_weight


def sample_data(X_train, y_train, train_size_k, method, seed, config, logger=None):
    """
    Sample training data using specified method.
    
    Parameters:
        X_train: Training features DataFrame
        y_train: Training target Series
        train_size_k: Training size in thousands (e.g., 0.05, 0.1, 0.2, 1.0)
        method: 'stratified' or 'balanced'
        seed: Random seed for reproducibility
        config: Configuration object with NUM_CLASSES
        logger: Optional logger instance
    
    Returns:
        X_sample: Sampled features
        y_sample: Sampled target
        sample_weights: Sample weights (None for balanced sampling)
    """
    train_size = int(train_size_k * 1000)  # Convert to actual number and ensure int
    
    if method == 'stratified':
        return sample_stratified(X_train, y_train, train_size, seed, logger)
    elif method == 'balanced':
        return sample_balanced(X_train, y_train, train_size, seed, config, logger)
    else:
        raise ValueError(f"Unknown sampling method: {method}")


def sample_stratified(X_train, y_train, train_size, seed, logger=None):
    """
    Stratified sampling: maintain class proportions with class weights.
    
    This method:
    - Samples data maintaining the original class distribution
    - Uses class weights to compensate for imbalance during training
    
    Parameters:
        X_train: Training features
        y_train: Training target
        train_size: Number of samples to draw (int)
        seed: Random seed
        logger: Optional logger
    
    Returns:
        X_sample, y_sample, sample_weights
    """
    # Ensure train_size is int (defense in depth)
    train_size = int(train_size)
    
    # Check if we're sampling the full training set
    if train_size >= len(X_train):
        if logger:
            logger.info(f"  Requested {train_size:,} samples, but only {len(X_train):,} available")
            logger.info(f"  Using full training set (no sampling needed)")
        X_sample = X_train
        y_sample = y_train
    else:
        # Stratified sampling
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train,
            train_size=train_size,
            random_state=seed,
            stratify=y_train
        )
    
    # Compute sample weights for class balance
    sample_weights = compute_sample_weight('balanced', y_sample)
    
    return X_sample, y_sample, sample_weights


def sample_balanced(X_train, y_train, train_size, seed, config, logger=None):
    """
    Balanced sampling: equal samples per class.
    
    This method:
    - Samples equal number of observations from each class
    - No class weights needed (data is already balanced)
    
    Parameters:
        X_train: Training features
        y_train: Training target
        train_size: Total number of samples to draw (int)
        seed: Random seed
        config: Configuration object with NUM_CLASSES
        logger: Optional logger
    
    Returns:
        X_sample, y_sample, sample_weights (None)
    """
    # Ensure train_size is int (defense in depth)
    train_size = int(train_size)
    
    samples_per_class = train_size // config.NUM_CLASSES
    
    X_list, y_list = [], []
    for class_label in sorted(y_train.unique()):
        class_indices = y_train[y_train == class_label].index
        
        # Check if we have enough samples for this class
        if len(class_indices) < samples_per_class:
            if logger:
                logger.warning(f"  Class {class_label}: requested {samples_per_class:,} samples, "
                             f"but only {len(class_indices):,} available!")
                logger.warning(f"  This training size is too large for balanced sampling!")
            raise ValueError(f"Insufficient samples for balanced sampling at size {train_size}")
        
        sampled_indices = resample(
            class_indices,
            n_samples=samples_per_class,
            random_state=seed,
            replace=False
        )
        X_list.append(X_train.loc[sampled_indices])
        y_list.append(y_train.loc[sampled_indices])
    
    X_sample = pd.concat(X_list, axis=0)
    y_sample = pd.concat(y_list, axis=0)
    
    # No need for sample weights with balanced sampling
    sample_weights = None
    
    return X_sample, y_sample, sample_weights