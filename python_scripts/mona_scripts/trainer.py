"""
SHARED TRAINER & EVALUATOR

This module contains functions for training models and evaluating them.

"""

import xgboost as xgb
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


def train_model(X_train_sample, y_train_sample, sample_weights, config):
    """
    Train XGBoost model.
    
    Parameters:
        X_train_sample: Training features
        y_train_sample: Training target
        sample_weights: Sample weights (None for balanced sampling)
        config: Configuration object with XGBOOST_PARAMS
    
    Returns:
        model: Trained XGBoost classifier
    """
    model = xgb.XGBClassifier(**config.XGBOOST_PARAMS)
    
    if sample_weights is not None:
        model.fit(X_train_sample, y_train_sample, sample_weight=sample_weights)
    else:
        model.fit(X_train_sample, y_train_sample)
    
    return model


def evaluate_model(model, X_test, y_test, config):
    """
    Evaluate model and return comprehensive metrics.
    
    Parameters:
        model: Trained model
        X_test: Test features
        y_test: Test target
        config: Configuration object with CLASS_NAMES
    
    Returns:
        metrics: Dictionary of evaluation metrics
            - balanced_accuracy: Balanced accuracy score
            - macro_f1: Macro-averaged F1 score
            - weighted_f1: Weighted F1 score
            - f1_{class}: F1 score for each class
            - precision_{class}: Precision for each class
            - recall_{class}: Recall for each class
    """
    y_pred = model.predict(X_test)
    
    # Overall metrics
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Per-class metrics
    per_class_f1 = f1_score(y_test, y_pred, average=None)
    per_class_precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    
    metrics = {
        'balanced_accuracy': balanced_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(config.CLASS_NAMES):
        metrics[f'f1_{class_name}'] = per_class_f1[i]
        metrics[f'precision_{class_name}'] = per_class_precision[i]
        metrics[f'recall_{class_name}'] = per_class_recall[i]
    
    return metrics