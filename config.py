"""
TSFresh Ensemble - Configuration Module
Uses pre-extracted tsfresh features from Auto_Feature_ML_No_Window_12K
Trains 9 binary detectors (one per class)
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
ML_DIR = Path(__file__).parent

# Source data (pre-extracted tsfresh features)
SOURCE_DATA_DIR = BASE_DIR / "Auto_Feature_ML_No_Window_12K" / "processed_data_no_window_12k"

# Output directories
MODELS_DIR = ML_DIR / "trained_models"
RESULTS_DIR = ML_DIR / "results"

# 9 classes (same as source)
CLASSES = [
    'collective_anomaly',
    'contextual_anomaly',
    'deterministic_trend',
    'mean_shift',
    'point_anomaly',
    'stochastic_trend',
    'trend_shift',
    'variance_shift',
    'volatility'
]

# Class index mapping (matching source data)
CLASS_INDEX_MAP = {
    'collective_anomaly': 0,
    'contextual_anomaly': 1,
    'deterministic_trend': 2,
    'mean_shift': 3,
    'point_anomaly': 4,
    'stochastic_trend': 5,  # Note: source uses 'Stochastic Trend'
    'trend_shift': 6,
    'variance_shift': 7,
    'volatility': 8  # Note: source uses 'Volatility'
}

# Model Training Settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Models to train for each detector
MODELS_TO_TRAIN = ['lightgbm', 'xgboost', 'mlp']
