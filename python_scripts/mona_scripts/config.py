"""
SHARED CONFIGURATION 

This module contains shared configuration used across all scripts.

"""

from pathlib import Path

class BaseConfig:
    """Base configuration shared across all RQ1 experiments."""
    
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
    
    # Preprocessing
    HANDLE_MISSING_NUMERICAL = -1
    
    # Train/test split (SAME ACROSS ALL EXPERIMENTS!)
    RANDOM_STATE = 42
    TEST_SIZE = 0.3
    
    # XGBoost parameters (SAME ACROSS ALL EXPERIMENTS!)
    XGBOOST_PARAMS = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,  # 
        'n_jobs': -1,
        'tree_method': 'hist',
        'enable_categorical': True,
        'eval_metric': 'mlogloss',
        'verbosity': 0
    }
    
    # Class information (populated after loading data)
    CLASS_NAMES = None
    NUM_CLASSES = None


def setup_config(data_path, output_dir):
    """
    Setup configuration for a specific RQ1 experiment.
    
    Parameters:
        data_path: Path to Stata data file
        output_dir: Path to output directory
    
    Returns:
        config: Configuration object with paths set
    """
    config = BaseConfig()
    
    # Set paths
    config.DATA_FILE = Path(data_path)
    config.OUTPUT_DIR = Path(output_dir)
    config.TABLES_DIR = config.OUTPUT_DIR / "tables"
    config.PLOTS_DIR = config.OUTPUT_DIR / "plots"
    config.MODELS_DIR = config.OUTPUT_DIR / "models"
    
    # Create directories
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    return config