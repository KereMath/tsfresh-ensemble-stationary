"""
Comprehensive Multi-Label Testing for Combination Data

Tests the ensemble model's multi-label prediction capability on ground-truth
combination data where we know exactly what labels each sample should have.

Metrics:
- Full Match: Both labels predicted correctly
- Partial Match: One label predicted correctly
- No Match: No labels predicted correctly
- Label-wise accuracy: Individual label detection rates
- Confusion analysis: Which label pairs are confused
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import config
from combination_mapping import COMBINATION_MAPPING, get_labels_from_folder

# Model loading utilities
import joblib


def load_all_models(models_dir):
    """
    Load all trained binary detector models

    Args:
        models_dir: Path to trained_models directory

    Returns:
        Dict: {class_name: {model_name: model, 'scaler': scaler}}
    """
    models_dir = Path(models_dir)
    all_models = {}

    for class_name in config.CLASSES:
        class_dir = models_dir / class_name

        if not class_dir.exists():
            print(f"  WARNING: {class_dir} not found, skipping {class_name}")
            continue

        # Load best model info
        info_file = class_dir / "best_model_info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                info = json.load(f)
                best_model_name = info['best_model']
        else:
            # Default to lightgbm if info not found
            best_model_name = 'lightgbm'

        # Load model
        model_file = class_dir / f"{best_model_name}.joblib"
        if not model_file.exists():
            print(f"  WARNING: {model_file} not found, skipping {class_name}")
            continue

        model = joblib.load(model_file)

        # Load scaler
        scaler_file = class_dir / "scalers.pkl"
        if scaler_file.exists():
            scaler_data = joblib.load(scaler_file)
            scaler = scaler_data['scaler']
        else:
            scaler = None
            print(f"  WARNING: No scaler found for {class_name}")

        all_models[class_name] = {
            best_model_name: model,
            'scaler': scaler
        }

        print(f"  Loaded {class_name}: {best_model_name}")

    return all_models


def load_combination_samples(combinations_dir, samples_per_combo=50):
    """
    Load samples from each combination folder (recursively searches for data/ folders)

    Args:
        combinations_dir: Path to Combinations folder
        samples_per_combo: Number of samples to load per combination

    Returns:
        List of dicts with 'time_series', 'true_labels', 'combination_name'
    """
    print("=" * 70)
    print("  LOADING COMBINATION SAMPLES")
    print("=" * 70)

    samples = []
    combinations_dir = Path(combinations_dir)

    # Map of base folders to combination folders
    base_folders = {
        'Cubic Base': ['cubic_collective_anomaly', 'Cubic + Mean Shift',
                       'Cubic + Point Anomaly', 'Cubic + Variance Shift'],
        'Damped Base': ['Damped + Collective Anomaly', 'Damped + Mean Shift',
                        'Damped + Point Anomaly', 'Damped + Variance Shift'],
        'Exponential Base': ['exponential_collective_anomaly', 'Exponential + Mean Shift',
                             'exponential_point_anomaly', 'exponential_variance_shift'],
        'Linear Base': ['Linear + Collective Anomaly', 'Linear + Mean Shift',
                        'Linear + Point Anomaly', 'Linear + Trend Shift',
                        'Linear + Variance Shift'],
        'Quadratic Base': ['Quadratic + Collective anomaly', 'Quadratic + Mean Shift',
                           'Quadratic + Point Anomaly', 'Quadratic + Variance Shift']
    }

    for base_folder, combo_folders in base_folders.items():
        for combo_name in combo_folders:
            # Get expected labels
            expected_labels = get_labels_from_folder(combo_name)

            # Base combination path
            combo_base_path = combinations_dir / base_folder / combo_name

            if not combo_base_path.exists():
                print(f"\n  WARNING: {combo_base_path} not found, skipping...")
                continue

            # Recursively find all 'data' folders under this combination
            data_folders = list(combo_base_path.rglob("data"))

            if len(data_folders) == 0:
                print(f"\n  WARNING: No 'data' folders found in {combo_base_path}, skipping...")
                continue

            # Collect all CSV files from all data folders
            all_csv_files = []
            for data_folder in data_folders:
                csv_files = list(data_folder.glob("*.csv"))
                all_csv_files.extend(csv_files)

            if len(all_csv_files) == 0:
                print(f"\n  WARNING: No CSV files found in {combo_name}, skipping...")
                continue

            # Sample randomly from all CSV files
            selected_files = np.random.choice(all_csv_files,
                                             size=min(samples_per_combo, len(all_csv_files)),
                                             replace=False)

            print(f"\n  {combo_name}: Loading {len(selected_files)} samples from {len(data_folders)} data folder(s)...")

            for csv_file in selected_files:
                try:
                    # Load time series
                    df = pd.read_csv(csv_file)

                    # Assuming CSV has a 'value' column
                    if 'value' in df.columns:
                        time_series = df['value'].values
                    elif len(df.columns) == 1:
                        time_series = df.iloc[:, 0].values
                    else:
                        # Take first numeric column
                        time_series = df.select_dtypes(include=[np.number]).iloc[:, 0].values

                    samples.append({
                        'time_series': time_series,
                        'true_labels': expected_labels,
                        'combination_name': combo_name,
                        'file_name': csv_file.name
                    })

                except Exception as e:
                    print(f"    Error loading {csv_file.name}: {e}")

    print(f"\n  Total samples loaded: {len(samples)}")
    print("=" * 70)

    return samples


def extract_tsfresh_features(time_series, expected_features=None):
    """
    Extract TSFresh features from a time series
    (This should match the feature extraction used during training)

    Args:
        time_series: 1D numpy array
        expected_features: List of expected feature names (for alignment)

    Returns:
        Feature vector (777 features, aligned with training)
    """
    from tsfresh import extract_features
    from tsfresh.feature_extraction import EfficientFCParameters

    # Create DataFrame in tsfresh format
    df = pd.DataFrame({
        'id': [0] * len(time_series),
        'time': range(len(time_series)),
        'value': time_series
    })

    # Extract features (MUST match training: EfficientFCParameters!)
    features_df = extract_features(df,
                                   column_id='id',
                                   column_sort='time',
                                   default_fc_parameters=EfficientFCParameters(),
                                   disable_progressbar=True)

    # If expected features provided, align to match training
    if expected_features is not None:
        # Reindex to match expected features (fills missing with 0, drops extra)
        features_df = features_df.reindex(columns=expected_features, fill_value=0)

    # Fill any remaining NaN values with 0 (some tsfresh features may return NaN)
    features_df = features_df.fillna(0)

    return features_df.values[0]


def predict_multilabel_ensemble(models, features, threshold=0.5):
    """
    Predict multi-label using ensemble

    Args:
        models: Dict of trained models for each class
        features: Feature vector
        threshold: Confidence threshold for multi-label

    Returns:
        Dict with predictions and confidences
    """
    confidences = {}

    # Classes not present in combination data
    UNUSED_CLASSES = {'contextual_anomaly', 'stochastic_trend', 'volatility'}

    # Get prediction from each binary classifier
    for class_name in config.CLASSES:
        # Skip classes not present in combination data
        if class_name in UNUSED_CLASSES:
            continue

        if class_name not in models:
            confidences[class_name] = 0.0
            continue

        class_data = models[class_name]
        scaler = class_data.get('scaler')

        # Scale features if scaler available
        if scaler is not None:
            features_scaled = scaler.transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)

        # Get model (skip 'scaler' key)
        model = None
        for key, value in class_data.items():
            if key != 'scaler':
                model = value
                break

        if model is None:
            confidences[class_name] = 0.0
            continue

        # Predict
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(features_scaled)[0, 1]
        else:
            # For models without predict_proba, use decision function
            prob = model.predict(features_scaled)[0]

        confidences[class_name] = float(prob)

    # Get multi-label predictions
    multi_label = [cls for cls, conf in confidences.items() if conf >= threshold]

    # Sort by confidence
    sorted_preds = sorted(confidences.items(), key=lambda x: x[1], reverse=True)

    return {
        'multi_label': multi_label,
        'all_confidences': confidences,
        'top3': sorted_preds[:3],
        'primary': sorted_preds[0][0],
        'primary_confidence': sorted_preds[0][1]
    }


def evaluate_multilabel_predictions(samples, models, expected_features, threshold=0.5):
    """
    Comprehensive evaluation of multi-label predictions

    Args:
        samples: List of sample dicts
        models: Trained ensemble models
        expected_features: List of feature names from training
        threshold: Multi-label threshold

    Returns:
        Detailed results dict
    """
    print("\n" + "=" * 70)
    print("  PREDICTING AND EVALUATING")
    print("=" * 70)

    results = []

    for sample in tqdm(samples, desc="  Predicting"):
        # Extract features (aligned with training)
        features = extract_tsfresh_features(sample['time_series'], expected_features)

        # Predict
        prediction = predict_multilabel_ensemble(models, features, threshold)

        # Compare with ground truth
        true_labels = set(sample['true_labels'])
        pred_labels = set(prediction['multi_label'])

        # Match analysis
        full_match = true_labels == pred_labels
        partial_match = len(true_labels & pred_labels) > 0
        no_match = len(true_labels & pred_labels) == 0

        # Individual label detection
        label_detection = {}
        for label in sample['true_labels']:
            label_detection[label] = label in pred_labels

        results.append({
            'combination_name': sample['combination_name'],
            'file_name': sample['file_name'],
            'true_labels': sample['true_labels'],
            'predicted_labels': prediction['multi_label'],
            'all_confidences': prediction['all_confidences'],
            'full_match': full_match,
            'partial_match': partial_match,
            'no_match': no_match,
            'label_detection': label_detection,
            'num_true_labels': len(true_labels),
            'num_pred_labels': len(pred_labels),
            'intersection_size': len(true_labels & pred_labels)
        })

    return results


def analyze_results(results):
    """
    Comprehensive analysis of multi-label prediction results

    Args:
        results: List of result dicts

    Returns:
        Analysis dict
    """
    print("\n" + "=" * 70)
    print("  MULTI-LABEL PERFORMANCE ANALYSIS")
    print("=" * 70)

    total = len(results)

    # Overall match statistics
    full_matches = sum(1 for r in results if r['full_match'])
    partial_matches = sum(1 for r in results if r['partial_match'] and not r['full_match'])
    no_matches = sum(1 for r in results if r['no_match'])

    print(f"\n  Overall Match Statistics:")
    print(f"    Full Match (both labels correct):     {full_matches:5d} ({100*full_matches/total:5.1f}%)")
    print(f"    Partial Match (one label correct):    {partial_matches:5d} ({100*partial_matches/total:5.1f}%)")
    print(f"    No Match (no labels correct):         {no_matches:5d} ({100*no_matches/total:5.1f}%)")

    # Label-wise detection rates
    print(f"\n  Label-Wise Detection Rates:")
    print(f"    (How often each label is correctly detected when it's true)")

    label_stats = {}
    for label in config.CLASSES:
        # Samples where this label is true
        true_samples = [r for r in results if label in r['true_labels']]
        if len(true_samples) == 0:
            continue

        # How many times it was detected
        detected = sum(1 for r in true_samples if r['label_detection'].get(label, False))

        label_stats[label] = {
            'total': len(true_samples),
            'detected': detected,
            'detection_rate': detected / len(true_samples)
        }

    for label in sorted(label_stats.keys(), key=lambda x: label_stats[x]['detection_rate'], reverse=True):
        stats = label_stats[label]
        print(f"    {label:<25} {stats['detected']:4d}/{stats['total']:4d} ({stats['detection_rate']:5.1%})")

    # Combination-wise accuracy
    print(f"\n  Combination-Wise Full Match Rates:")

    combo_stats = {}
    for combo_name in set(r['combination_name'] for r in results):
        combo_results = [r for r in results if r['combination_name'] == combo_name]
        combo_full = sum(1 for r in combo_results if r['full_match'])

        combo_stats[combo_name] = {
            'total': len(combo_results),
            'full_match': combo_full,
            'full_match_rate': combo_full / len(combo_results) if combo_results else 0
        }

    for combo in sorted(combo_stats.keys(), key=lambda x: combo_stats[x]['full_match_rate'], reverse=True):
        stats = combo_stats[combo]
        print(f"    {combo:<40} {stats['full_match']:3d}/{stats['total']:3d} ({stats['full_match_rate']:5.1%})")

    # Intersection size distribution
    print(f"\n  Intersection Size Distribution:")
    print(f"    (How many true labels were predicted)")

    intersection_counts = {}
    for r in results:
        size = r['intersection_size']
        intersection_counts[size] = intersection_counts.get(size, 0) + 1

    for size in sorted(intersection_counts.keys()):
        count = intersection_counts[size]
        print(f"    {size} labels correct: {count:5d} ({100*count/total:5.1f}%)")

    # Prediction size distribution
    print(f"\n  Predicted Label Count Distribution:")

    pred_size_counts = {}
    for r in results:
        size = r['num_pred_labels']
        pred_size_counts[size] = pred_size_counts.get(size, 0) + 1

    for size in sorted(pred_size_counts.keys()):
        count = pred_size_counts[size]
        print(f"    {size} labels predicted: {count:5d} ({100*count/total:5.1f}%)")

    print("=" * 70)

    return {
        'overall': {
            'total': total,
            'full_match': full_matches,
            'partial_match': partial_matches,
            'no_match': no_matches,
            'full_match_rate': full_matches / total,
            'partial_match_rate': partial_matches / total,
            'no_match_rate': no_matches / total
        },
        'label_wise': label_stats,
        'combination_wise': combo_stats,
        'intersection_distribution': intersection_counts,
        'prediction_size_distribution': pred_size_counts
    }


def main():
    """Main testing pipeline"""
    # Paths
    combinations_dir = Path("c:/Users/user/Desktop/STATIONARY/Combinations")
    models_dir = config.MODELS_DIR
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Load trained models
    print("=" * 70)
    print("  LOADING TRAINED ENSEMBLE MODELS")
    print("=" * 70)

    models = load_all_models(models_dir)

    print(f"\n  Loaded models for {len(models)} classes")

    # Load expected feature names from training
    print("\n" + "=" * 70)
    print("  LOADING TRAINING FEATURE NAMES")
    print("=" * 70)

    feature_names_file = config.SOURCE_DATA_DIR / 'feature_names.txt'
    if feature_names_file.exists():
        with open(feature_names_file, 'r') as f:
            expected_features = [line.strip() for line in f.readlines()]
        print(f"\n  Loaded {len(expected_features)} feature names from training")
    else:
        print(f"\n  WARNING: {feature_names_file} not found!")
        print("  Feature alignment may fail. Continuing without alignment...")
        expected_features = None

    # Load combination samples (reduced for faster testing)
    samples = load_combination_samples(combinations_dir, samples_per_combo=10)

    # Predict and evaluate (lowered threshold for combination data)
    results = evaluate_multilabel_predictions(samples, models, expected_features, threshold=0.0005)

    # Analyze
    analysis = analyze_results(results)

    # Save results
    output_file = results_dir / "multilabel_combination_test.json"

    # Convert results for JSON serialization
    json_results = []
    for r in results:
        json_results.append({
            'combination_name': r['combination_name'],
            'file_name': r['file_name'],
            'true_labels': r['true_labels'],
            'predicted_labels': r['predicted_labels'],
            'all_confidences': {k: float(v) for k, v in r['all_confidences'].items()},
            'full_match': r['full_match'],
            'partial_match': r['partial_match'],
            'no_match': r['no_match'],
            'intersection_size': r['intersection_size']
        })

    # Convert analysis for JSON
    json_analysis = {
        'overall': analysis['overall'],
        'label_wise': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                          for kk, vv in v.items()}
                      for k, v in analysis['label_wise'].items()},
        'combination_wise': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                                for kk, vv in v.items()}
                            for k, v in analysis['combination_wise'].items()},
        'intersection_distribution': {int(k): v for k, v in analysis['intersection_distribution'].items()},
        'prediction_size_distribution': {int(k): v for k, v in analysis['prediction_size_distribution'].items()}
    }

    with open(output_file, 'w') as f:
        json.dump({
            'results': json_results,
            'analysis': json_analysis
        }, f, indent=2)

    print(f"\n  Results saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
