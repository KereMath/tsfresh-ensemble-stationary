"""
TSFresh Ensemble Classifier
Evaluates 9 binary detectors using pre-extracted tsfresh features
"""
import numpy as np
import json
from pathlib import Path
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

import config


def load_detectors():
    """Load all 9 trained detector models"""
    print("=" * 70)
    print("  LOADING TSFRESH DETECTOR MODELS")
    print("=" * 70)

    loaded = {}

    for class_name in config.CLASSES:
        detector_dir = config.MODELS_DIR / class_name

        try:
            # Load best model info
            with open(detector_dir / "best_model_info.json") as f:
                best_info = json.load(f)
            best_model_name = best_info['best_model']

            # Load model and scaler
            model = joblib.load(detector_dir / f"{best_model_name}.joblib")
            scalers = joblib.load(detector_dir / "scalers.pkl")

            loaded[class_name] = {
                'model': model,
                'scaler': scalers['scaler']
            }
            print(f"  [OK] {class_name} ({best_model_name})")

        except Exception as e:
            print(f"  [FAIL] {class_name}: {e}")

    print(f"\n  Loaded {len(loaded)}/9 detectors")
    print("=" * 70)

    return loaded


def load_test_data():
    """Load test data from tsfresh processed data"""
    print("\n" + "-" * 70)
    print("  LOADING TEST DATA")
    print("-" * 70)

    X_test = np.load(config.SOURCE_DATA_DIR / 'X_test.npy')
    y_test = np.load(config.SOURCE_DATA_DIR / 'y_test.npy')

    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    print(f"  Classes: {np.unique(y_test)}")

    # Count per class
    for class_name in config.CLASSES:
        class_idx = config.CLASS_INDEX_MAP[class_name]
        count = np.sum(y_test == class_idx)
        print(f"    {class_name}: {count} samples")

    return X_test, y_test


def predict_single(detectors, X_sample):
    """Run all detectors on a single sample"""
    results = {}

    for class_name, det in detectors.items():
        try:
            # Scale
            X_scaled = det['scaler'].transform(X_sample.reshape(1, -1))

            # Predict
            prob = det['model'].predict_proba(X_scaled)[0, 1]
            results[class_name] = prob

        except Exception as e:
            results[class_name] = 0.0

    return results


def ensemble_decision(detector_results):
    """Return class with highest probability"""
    best_class = max(detector_results, key=detector_results.get)
    best_prob = detector_results[best_class]
    return best_class, best_prob


def evaluate_ensemble():
    """Main evaluation function"""
    print("\n" + "=" * 70)
    print("  TSFRESH ENSEMBLE EVALUATION")
    print("=" * 70)

    # Load detectors
    detectors = load_detectors()

    if len(detectors) < 9:
        print(f"\nWARNING: Only {len(detectors)} detectors loaded!")
        return None

    # Load test data
    X_test, y_test = load_test_data()

    # Run predictions
    print("\n" + "-" * 70)
    print("  RUNNING PREDICTIONS")
    print("-" * 70)

    y_pred_indices = []
    start_time = time.time()

    for i in range(len(X_test)):
        if i > 0 and i % 500 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (len(X_test) - i) / rate
            print(f"  Progress: {i}/{len(X_test)} ({100*i/len(X_test):.1f}%) | Rate: {rate:.0f}/sec | ETA: {remaining:.0f}s")

        # Get probabilities from all detectors
        detector_results = predict_single(detectors, X_test[i])

        # Make decision
        pred_class, _ = ensemble_decision(detector_results)

        # Convert class name to index
        pred_idx = config.CLASS_INDEX_MAP[pred_class]
        y_pred_indices.append(pred_idx)

    total_time = time.time() - start_time
    print(f"  Progress: {len(X_test)}/{len(X_test)} (100.0%) | Rate: {len(X_test)/total_time:.0f}/sec")

    y_pred = np.array(y_pred_indices)

    # Calculate metrics
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    correct = np.sum(y_pred == y_test)
    accuracy = correct / len(y_test)

    print(f"\n  Overall Accuracy: {accuracy:.4f} ({correct}/{len(y_test)})")
    print(f"  Processing Time: {total_time:.1f}s ({len(X_test)/total_time:.0f} samples/sec)")

    # Per-class metrics
    print("\n" + "-" * 70)
    print("  PER-CLASS METRICS")
    print("-" * 70)

    print(f"\n  {'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("  " + "-" * 65)

    class_metrics = {}
    for class_name in config.CLASSES:
        class_idx = config.CLASS_INDEX_MAP[class_name]

        tp = np.sum((y_test == class_idx) & (y_pred == class_idx))
        fp = np.sum((y_test != class_idx) & (y_pred == class_idx))
        fn = np.sum((y_test == class_idx) & (y_pred != class_idx))
        support = np.sum(y_test == class_idx)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(support)
        }
        print(f"  {class_name:<25} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")

    # Confusion matrix
    print("\n" + "-" * 70)
    print("  CONFUSION MATRIX")
    print("-" * 70)

    # Header
    header = "  True/Pred     "
    for cls in config.CLASSES:
        header += f"{cls[:8]:>10}"
    print(header)
    print("  " + "-" * (16 + 10 * len(config.CLASSES)))

    for true_cls in config.CLASSES:
        true_idx = config.CLASS_INDEX_MAP[true_cls]
        row = f"  {true_cls[:14]:<14}"
        for pred_cls in config.CLASSES:
            pred_idx = config.CLASS_INDEX_MAP[pred_cls]
            count = np.sum((y_test == true_idx) & (y_pred == pred_idx))
            row += f"{count:>10}"
        print(row)

    # Save results
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        'accuracy': float(accuracy),
        'total_samples': int(len(y_test)),
        'correct': int(correct),
        'processing_time': total_time,
        'class_metrics': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                              for kk, vv in v.items()}
                          for k, v in class_metrics.items()}
    }

    with open(config.RESULTS_DIR / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {config.RESULTS_DIR}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = evaluate_ensemble()
