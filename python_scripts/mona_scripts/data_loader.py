"""
SHARED DATA LOADER 

This module contains functions for loading and preprocessing data.

"""

import pandas as pd
import time
from sklearn.model_selection import train_test_split


def load_data(config, logger):
    """
    Load Stata data file.
    
    Parameters:
        config: Configuration object with DATA_FILE and ALL_FEATURES
        logger: Logger instance
    
    Returns:
        data: Loaded DataFrame
    """
    logger.info("=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)
    
    if not config.DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {config.DATA_FILE}")
    
    logger.info(f"Reading {config.DATA_FILE.name}...")
    start_time = time.time()
    
    required_cols = [config.TARGET_COLUMN] + config.ALL_FEATURES
    
    try:
        data = pd.read_stata(str(config.DATA_FILE), columns=required_cols)
    except Exception as e:
        logger.warning(f"Could not load with columns parameter: {e}")
        data = pd.read_stata(str(config.DATA_FILE))
        data = data[required_cols]
    
    elapsed = time.time() - start_time
    
    logger.info(f"Data loaded in {elapsed:.1f} seconds!")
    logger.info(f"Shape: {data.shape}")
    logger.info(f"Memory: {data.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    logger.info("")
    
    return data


def preprocess_data(data, config, logger):
    """
    Preprocess data for modeling.
    
    Parameters:
        data: Raw DataFrame
        config: Configuration object
        logger: Logger instance
    
    Returns:
        X: Feature DataFrame
        y: Target Series (encoded as integers 0..NUM_CLASSES-1)
    """
    logger.info("=" * 80)
    logger.info("PREPROCESSING DATA")
    logger.info("=" * 80)
    
    # ---------------------------------------------------------------------
    # 1. Separate features and raw target
    # ---------------------------------------------------------------------
    X = data[config.ALL_FEATURES].copy()
    y_raw = data[config.TARGET_COLUMN].copy()
    
    # ---------------------------------------------------------------------
    # 2. Encode target labels to integers (XGBoost requirement)
    #    - Keep original labels as config.CLASS_NAMES
    #    - Map original -> int in a stable, sorted order
    # ---------------------------------------------------------------------
    unique_labels = sorted(y_raw.unique())
    config.CLASS_NAMES = [str(lbl) for lbl in unique_labels]
    config.NUM_CLASSES = len(config.CLASS_NAMES)
    
    label_to_int = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    idx_to_label = {idx: lbl for lbl, idx in label_to_int.items()}
    
    # optional but handy if you ever want to invert the mapping
    config.CLASS_TO_IDX = label_to_int
    config.IDX_TO_CLASS = idx_to_label
    
    y = y_raw.map(label_to_int)
    
    logger.info(f"Target variable: {config.TARGET_COLUMN}")
    logger.info(f"Classes found: {config.NUM_CLASSES}")
    logger.info(f"Original class names: {config.CLASS_NAMES}")
    logger.info("Class distribution (encoded):")
    
    n_total = len(y)
    for idx, orig_label in enumerate(config.CLASS_NAMES):
        count = (y == idx).sum()
        pct = count / n_total * 100
        logger.info(f"  {orig_label} -> {idx}: {count:,} ({pct:.2f}%)")
    
    # ---------------------------------------------------------------------
    # 3. Handle missing values in numerical features
    # ---------------------------------------------------------------------
    logger.info("")
    logger.info("Handling missing values in numerical features...")
    for col in config.NUMERICAL_FEATURES:
        if col not in X.columns:
            logger.warning(f"Numerical column '{col}' not in data; skipping")
            continue
        n_missing = X[col].isnull().sum()
        if n_missing > 0:
            X[col].fillna(config.HANDLE_MISSING_NUMERICAL, inplace=True)
            logger.info(f"  {col}: {n_missing:,} missing values filled")
    
    # ---------------------------------------------------------------------
    # 4. Convert categorical features to 'category' dtype (for XGBoost)
    # ---------------------------------------------------------------------
    logger.info("")
    logger.info("Converting categorical features...")
    for col in config.CATEGORICAL_FEATURES:
        if col not in X.columns:
            logger.warning(f"Categorical column '{col}' not in data; skipping")
            continue
        X[col] = X[col].astype('category')
        logger.info(f"  {col}: {X[col].nunique()} categories")
    
    logger.info("")
    logger.info("Preprocessing complete!")
    logger.info("")
    
    return X, y


def split_data(X, y, config, logger):
    """
    Split data into train and test sets.
    
    CRITICAL: Uses same random_state across all experiments for consistency!
    
    Parameters:
        X: Feature DataFrame
        y: Target Series
        config: Configuration object
        logger: Logger instance
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info("=" * 80)
    logger.info("SPLITTING DATA")
    logger.info("=" * 80)
    
    logger.info(f"Test size: {config.TEST_SIZE*100:.1f}%")
    logger.info(f"Random state: {config.RANDOM_STATE}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    logger.info(f"Train set: {len(X_train):,} samples")
    logger.info(f"Test set: {len(X_test):,} samples")
    logger.info("")
    
    # Log class distribution in train/test
    logger.info("Train set class distribution:")
    train_counts = y_train.value_counts().sort_index()
    for class_label, count in train_counts.items():
        pct = count / len(y_train) * 100
        logger.info(f"  {class_label}: {count:,} ({pct:.2f}%)")
    
    logger.info("")
    logger.info("Test set class distribution:")
    test_counts = y_test.value_counts().sort_index()
    for class_label, count in test_counts.items():
        pct = count / len(y_test) * 100
        logger.info(f"  {class_label}: {count:,} ({pct:.2f}%)")
    
    logger.info("")
    return X_train, X_test, y_train, y_test