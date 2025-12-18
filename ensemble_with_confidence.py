"""
TSFresh Ensemble - Confidence-Aware Evaluation
Shows all 9 detector confidences for each prediction
Supports multi-label predictions
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
    print("  LOADING TSFRESH DETECTOR MODELS WITH CONFIDENCE TRACKING")
    print("=" * 70)

    loaded = {}

    for class_name in config.CLASSES:
        detector_dir = config.MODELS_DIR / class_name

        try:
            with open(detector_dir / "best_model_info.json") as f:
                best_info = json.load(f)
            best_model_name = best_info['best_model']

            model = joblib.load(detector_dir / f"{best_model_name}.joblib")
            scalers = joblib.load(detector_dir / "scalers.pkl")

            loaded[class_name] = {
                'model': model,
                'scaler': scalers['scaler'],
                'model_name': best_model_name
            }
            print(f"  [OK] {class_name:<25} ({best_model_name})")

        except Exception as e:
            print(f"  [FAIL] {class_name}: {e}")

    print(f"\n  Loaded {len(loaded)}/9 detectors")
    print("=" * 70)

    return loaded


def predict_with_all_confidences(detectors, X_sample):
    """Get confidence scores from all 9 detectors"""
    confidences = {}

    for class_name, det in detectors.items():
        try:
            X_scaled = det['scaler'].transform(X_sample.reshape(1, -1))
            prob = det['model'].predict_proba(X_scaled)[0, 1]
            confidences[class_name] = float(prob)
        except Exception as e:
            confidences[class_name] = 0.0

    return confidences


def get_top_predictions(confidences, threshold=0.5):
    """Get predictions based on confidence threshold"""
    sorted_classes = sorted(confidences.items(), key=lambda x: x[1], reverse=True)

    primary_class = sorted_classes[0][0]
    primary_conf = sorted_classes[0][1]

    multi_label = [cls for cls, conf in sorted_classes if conf >= threshold]
    top3 = [(cls, conf) for cls, conf in sorted_classes[:3]]

    return {
        'primary': primary_class,
        'primary_confidence': primary_conf,
        'multi_label': multi_label,
        'top3': top3,
        'all_confidences': confidences
    }


def evaluate_with_confidence(multi_label_threshold=0.5, save_detailed=True):
    """Evaluate ensemble with full confidence tracking"""
    print("\n" + "=" * 70)
    print("  TSFRESH ENSEMBLE - CONFIDENCE-AWARE EVALUATION")
    print("=" * 70)
    print(f"  Multi-label threshold: {multi_label_threshold}")
    print("=" * 70)

    # Load detectors
    detectors = load_detectors()

    if len(detectors) < 9:
        print(f"\nWARNING: Only {len(detectors)} detectors loaded!")
        return None

    # Load test data (pre-processed by Auto_Feature_ML_No_Window_12K)
    print("\n" + "-" * 70)
    print("  LOADING TEST DATA FROM TSFRESH")
    print("-" * 70)

    X = np.load(config.SOURCE_DATA_DIR / 'X_test.npy')
    y = np.load(config.SOURCE_DATA_DIR / 'y_test.npy')

    print(f"  X_test: {X.shape}")
    print(f"  y_test: {y.shape}")
    print(f"  Classes: {np.unique(y)}")

    # Count per class
    for class_name in config.CLASSES:
        class_idx = config.CLASS_INDEX_MAP[class_name]
        count = np.sum(y == class_idx)
        print(f"    {class_name}: {count} samples")

    # Run predictions
    print("\n" + "-" * 70)
    print("  RUNNING PREDICTIONS WITH CONFIDENCE TRACKING")
    print("-" * 70)

    all_predictions = []
    y_pred_primary = []

    start_time = time.time()

    for i in range(len(X)):
        if i > 0 and i % 500 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (len(X) - i) / rate
            print(f"  Progress: {i}/{len(X)} ({100*i/len(X):.1f}%) | Rate: {rate:.0f}/sec | ETA: {remaining:.0f}s")

        confidences = predict_with_all_confidences(detectors, X[i])
        pred_info = get_top_predictions(confidences, multi_label_threshold)

        true_class_name = config.CLASSES[y[i]]
        pred_info['true_class'] = true_class_name
        pred_info['sample_index'] = i

        all_predictions.append(pred_info)

        pred_idx = config.CLASS_INDEX_MAP[pred_info['primary']]
        y_pred_primary.append(pred_idx)

    total_time = time.time() - start_time
    print(f"  Progress: {len(X)}/{len(X)} (100.0%) | Rate: {len(X)/total_time:.0f}/sec")

    y_pred_primary = np.array(y_pred_primary)

    # ========== PRIMARY PREDICTION METRICS ==========
    print("\n" + "=" * 70)
    print("  PRIMARY PREDICTION RESULTS (Highest Confidence)")
    print("=" * 70)

    correct = np.sum(y_pred_primary == y)
    accuracy = correct / len(y)

    print(f"\n  Overall Accuracy: {accuracy:.4f} ({correct}/{len(y)})")
    print(f"  Processing Time: {total_time:.1f}s ({len(X)/total_time:.0f} samples/sec)")

    # Per-class metrics
    print("\n" + "-" * 70)
    print("  PER-CLASS METRICS (Primary Prediction)")
    print("-" * 70)
    print(f"\n  {'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("  " + "-" * 65)

    class_metrics = {}
    for class_name in config.CLASSES:
        class_idx = config.CLASS_INDEX_MAP[class_name]

        tp = np.sum((y == class_idx) & (y_pred_primary == class_idx))
        fp = np.sum((y != class_idx) & (y_pred_primary == class_idx))
        fn = np.sum((y == class_idx) & (y_pred_primary != class_idx))
        support = np.sum(y == class_idx)

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

    # ========== MULTI-LABEL ANALYSIS ==========
    print("\n" + "=" * 70)
    print(f"  MULTI-LABEL ANALYSIS (Threshold: {multi_label_threshold})")
    print("=" * 70)

    multi_label_counts = [len(p['multi_label']) for p in all_predictions]

    print(f"\n  Samples with 1 label: {sum(1 for c in multi_label_counts if c == 1)} ({100*sum(1 for c in multi_label_counts if c == 1)/len(all_predictions):.1f}%)")
    print(f"  Samples with 2 labels: {sum(1 for c in multi_label_counts if c == 2)} ({100*sum(1 for c in multi_label_counts if c == 2)/len(all_predictions):.1f}%)")
    print(f"  Samples with 3+ labels: {sum(1 for c in multi_label_counts if c >= 3)} ({100*sum(1 for c in multi_label_counts if c >= 3)/len(all_predictions):.1f}%)")

    multi_label_hits = sum(1 for p in all_predictions if p['true_class'] in p['multi_label'])
    multi_label_hit_rate = multi_label_hits / len(all_predictions)

    print(f"\n  Multi-label hit rate: {multi_label_hit_rate:.4f} ({multi_label_hits}/{len(all_predictions)})")

    # ========== CONFIDENCE STATISTICS ==========
    print("\n" + "=" * 70)
    print("  CONFIDENCE STATISTICS")
    print("=" * 70)

    correct_confidences = [p['primary_confidence'] for i, p in enumerate(all_predictions)
                          if y_pred_primary[i] == y[i]]
    incorrect_confidences = [p['primary_confidence'] for i, p in enumerate(all_predictions)
                            if y_pred_primary[i] != y[i]]

    print(f"\n  Correct predictions:")
    print(f"    Mean confidence: {np.mean(correct_confidences):.4f}")
    print(f"    Median confidence: {np.median(correct_confidences):.4f}")

    print(f"\n  Incorrect predictions:")
    print(f"    Mean confidence: {np.mean(incorrect_confidences):.4f}")
    print(f"    Median confidence: {np.median(incorrect_confidences):.4f}")

    # ========== SAMPLE EXAMPLES ==========
    print("\n" + "=" * 70)
    print("  EXAMPLE PREDICTIONS (First 10 samples)")
    print("=" * 70)

    for i in range(min(10, len(all_predictions))):
        p = all_predictions[i]
        correct_marker = "[OK]" if p['primary'] == p['true_class'] else "[WRONG]"

        print(f"\n  Sample {i} {correct_marker}")
        print(f"    True: {p['true_class']}")
        print(f"    Predicted: {p['primary']} (conf: {p['primary_confidence']:.4f})")
        print(f"    Top 3:")
        for cls, conf in p['top3']:
            marker = "<--" if cls == p['true_class'] else "   "
            print(f"      {cls:<25} {conf:.4f} {marker}")
        if len(p['multi_label']) > 1:
            print(f"    Multi-label: {', '.join(p['multi_label'])}")

    # ========== SAVE RESULTS ==========
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        'primary_prediction': {
            'accuracy': float(accuracy),
            'total_samples': int(len(y)),
            'correct': int(correct),
            'processing_time': total_time,
            'class_metrics': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                                  for kk, vv in v.items()}
                              for k, v in class_metrics.items()}
        },
        'multi_label_analysis': {
            'threshold': multi_label_threshold,
            'samples_with_1_label': int(sum(1 for c in multi_label_counts if c == 1)),
            'samples_with_2_labels': int(sum(1 for c in multi_label_counts if c == 2)),
            'samples_with_3plus_labels': int(sum(1 for c in multi_label_counts if c >= 3)),
            'multi_label_hit_rate': float(multi_label_hit_rate)
        },
        'confidence_statistics': {
            'correct_predictions': {
                'mean': float(np.mean(correct_confidences)),
                'median': float(np.median(correct_confidences))
            },
            'incorrect_predictions': {
                'mean': float(np.mean(incorrect_confidences)),
                'median': float(np.median(incorrect_confidences))
            }
        }
    }

    with open(config.RESULTS_DIR / "confidence_evaluation.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Summary saved to: {config.RESULTS_DIR / 'confidence_evaluation.json'}")

    if save_detailed:
        detailed_predictions = []
        for p in all_predictions:
            detailed_predictions.append({
                'sample_index': p['sample_index'],
                'true_class': p['true_class'],
                'primary_prediction': p['primary'],
                'primary_confidence': p['primary_confidence'],
                'multi_label': p['multi_label'],
                'top3': [{'class': cls, 'confidence': conf} for cls, conf in p['top3']],
                'all_confidences': p['all_confidences']
            })

        with open(config.RESULTS_DIR / "detailed_predictions.json", 'w') as f:
            json.dump(detailed_predictions, f, indent=2)

        print(f"  Detailed predictions saved to: {config.RESULTS_DIR / 'detailed_predictions.json'}")

    print("=" * 70)

    return results, all_predictions


if __name__ == "__main__":
    results, predictions = evaluate_with_confidence(
        multi_label_threshold=0.5,
        save_detailed=True
    )
