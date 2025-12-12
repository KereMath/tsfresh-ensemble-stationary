"""
TSFresh Ensemble - Train 9 Binary Detectors
Uses pre-extracted tsfresh features from Auto_Feature_ML_No_Window_12K
"""
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

import config


def get_models():
    """Return dictionary of models to train"""
    return {
        'lightgbm': LGBMClassifier(
            n_estimators=300,
            random_state=config.RANDOM_STATE,
            verbose=-1,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'xgboost': XGBClassifier(
            n_estimators=300,
            random_state=config.RANDOM_STATE,
            tree_method='hist',
            n_jobs=-1,
            eval_metric='logloss'
        ),
        'mlp': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=500,
            random_state=config.RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1
        )
    }


def load_tsfresh_data():
    """Load pre-extracted tsfresh features"""
    print("=" * 70)
    print("  LOADING TSFRESH FEATURES")
    print("=" * 70)

    X_train = np.load(config.SOURCE_DATA_DIR / 'X_train.npy')
    X_test = np.load(config.SOURCE_DATA_DIR / 'X_test.npy')
    y_train = np.load(config.SOURCE_DATA_DIR / 'y_train.npy')
    y_test = np.load(config.SOURCE_DATA_DIR / 'y_test.npy')

    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape} (classes: {np.unique(y_train)})")
    print(f"  y_test: {y_test.shape}")

    # Load feature names
    with open(config.SOURCE_DATA_DIR / 'feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]

    print(f"  Features: {len(feature_names)}")
    print("=" * 70)

    return X_train, X_test, y_train, y_test, feature_names


def train_binary_detector(class_name, class_idx, X_train, X_test, y_train, y_test):
    """Train binary detector for a single class"""
    print(f"\n{'='*70}")
    print(f"  TRAINING: {class_name} detector (class {class_idx})")
    print(f"{'='*70}")

    # Convert to binary labels (target class = 1, others = 0)
    y_train_binary = (y_train == class_idx).astype(int)
    y_test_binary = (y_test == class_idx).astype(int)

    print(f"  Train: {np.sum(y_train_binary)} positive, {np.sum(y_train_binary==0)} negative")
    print(f"  Test: {np.sum(y_test_binary)} positive, {np.sum(y_test_binary==0)} negative")

    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    models = get_models()
    results = {}
    best_model = None
    best_model_name = None
    best_f1 = 0

    for name, model in models.items():
        print(f"\n  Training {name}...")
        start_time = time.time()

        model.fit(X_train_scaled, y_train_binary)
        train_time = time.time() - start_time

        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        f1 = f1_score(y_test_binary, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test_binary, y_pred)
        precision = precision_score(y_test_binary, y_pred, zero_division=0)
        recall = recall_score(y_test_binary, y_pred, zero_division=0)

        try:
            roc_auc = roc_auc_score(y_test_binary, y_prob)
        except:
            roc_auc = 0.0

        results[name] = {
            'f1': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'train_time': train_time
        }

        print(f"    F1: {f1:.4f}, Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    print(f"\n  Best model: {best_model_name} (F1: {best_f1:.4f})")

    return {
        'model': best_model,
        'scaler': scaler,
        'best_model_name': best_model_name,
        'results': results,
        'best_f1': best_f1
    }


def train_all_detectors():
    """Train all 9 binary detectors"""
    print("\n" + "=" * 70)
    print("  TSFRESH ENSEMBLE - TRAINING 9 BINARY DETECTORS")
    print("=" * 70)

    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_tsfresh_data()

    # Create output directory
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Train each detector
    all_results = {}

    for class_name in config.CLASSES:
        class_idx = config.CLASS_INDEX_MAP[class_name]

        result = train_binary_detector(
            class_name, class_idx,
            X_train, X_test, y_train, y_test
        )

        # Save model and scaler
        detector_dir = config.MODELS_DIR / class_name
        detector_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(result['model'], detector_dir / f"{result['best_model_name']}.joblib")
        joblib.dump({'scaler': result['scaler']}, detector_dir / "scalers.pkl")

        # Save best model info
        with open(detector_dir / "best_model_info.json", 'w') as f:
            json.dump({
                'best_model': result['best_model_name'],
                'best_f1': result['best_f1']
            }, f, indent=2)

        all_results[class_name] = {
            'best_model': result['best_model_name'],
            'best_f1': result['best_f1'],
            'results': result['results']
        }

    # Save overall results
    with open(config.MODELS_DIR / "training_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("  TRAINING SUMMARY")
    print("=" * 70)
    print(f"\n  {'Class':<25} {'Best Model':<12} {'F1':>8}")
    print("  " + "-" * 50)

    for class_name, result in all_results.items():
        print(f"  {class_name:<25} {result['best_model']:<12} {result['best_f1']:>8.4f}")

    avg_f1 = np.mean([r['best_f1'] for r in all_results.values()])
    print("  " + "-" * 50)
    print(f"  {'Average':<25} {'':<12} {avg_f1:>8.4f}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    results = train_all_detectors()
