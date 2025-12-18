"""
Uncertain Samples Analysis
Identifies and analyzes samples where the model is uncertain (multiple high-confidence labels)
"""
import numpy as np
import json
from pathlib import Path
import config


def analyze_uncertain_samples():
    """
    Analyze uncertain predictions from detailed_predictions.json
    """
    print("=" * 70)
    print("  UNCERTAIN SAMPLES ANALYSIS")
    print("=" * 70)

    # Load detailed predictions
    predictions_file = config.RESULTS_DIR / "detailed_predictions.json"

    if not predictions_file.exists():
        print(f"\nERROR: {predictions_file} not found!")
        print("Please run ensemble_with_confidence.py first.")
        return

    print(f"\n  Loading predictions from: {predictions_file}")

    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    print(f"  Total predictions: {len(predictions)}")

    # ========== UNCERTAINTY METRICS ==========
    print("\n" + "=" * 70)
    print("  UNCERTAINTY METRICS")
    print("=" * 70)

    # 1. Multi-label counts
    multi_label_counts = [len(p['multi_label']) for p in predictions]

    uncertain_2 = [p for p in predictions if len(p['multi_label']) == 2]
    uncertain_3plus = [p for p in predictions if len(p['multi_label']) >= 3]

    print(f"\n  Single label (certain): {sum(1 for c in multi_label_counts if c == 1)} ({100*sum(1 for c in multi_label_counts if c == 1)/len(predictions):.1f}%)")
    print(f"  Two labels (uncertain): {len(uncertain_2)} ({100*len(uncertain_2)/len(predictions):.1f}%)")
    print(f"  Three+ labels (highly uncertain): {len(uncertain_3plus)} ({100*len(uncertain_3plus)/len(predictions):.1f}%)")

    # 2. Confidence gap analysis
    print("\n" + "-" * 70)
    print("  CONFIDENCE GAP ANALYSIS")
    print("-" * 70)
    print("  (Gap between 1st and 2nd highest confidence)")

    confidence_gaps = []
    for p in predictions:
        top3 = p['top3']
        if len(top3) >= 2:
            gap = top3[0]['confidence'] - top3[1]['confidence']
            confidence_gaps.append(gap)

    print(f"\n  Mean confidence gap: {np.mean(confidence_gaps):.4f}")
    print(f"  Median confidence gap: {np.median(confidence_gaps):.4f}")
    print(f"  Std dev: {np.std(confidence_gaps):.4f}")

    # Small gap = uncertain
    small_gap = [p for p, gap in zip(predictions, confidence_gaps) if gap < 0.2]
    medium_gap = [p for p, gap in zip(predictions, confidence_gaps) if 0.2 <= gap < 0.5]
    large_gap = [p for p, gap in zip(predictions, confidence_gaps) if gap >= 0.5]

    print(f"\n  Small gap (<0.2): {len(small_gap)} ({100*len(small_gap)/len(predictions):.1f}%)")
    print(f"  Medium gap (0.2-0.5): {len(medium_gap)} ({100*len(medium_gap)/len(predictions):.1f}%)")
    print(f"  Large gap (>0.5): {len(large_gap)} ({100*len(large_gap)/len(predictions):.1f}%)")

    # ========== CLASS-WISE UNCERTAINTY ==========
    print("\n" + "=" * 70)
    print("  CLASS-WISE UNCERTAINTY")
    print("=" * 70)

    class_uncertainty = {}
    for class_name in config.CLASSES:
        class_preds = [p for p in predictions if p['true_class'] == class_name]
        uncertain_preds = [p for p in class_preds if len(p['multi_label']) >= 2]

        class_uncertainty[class_name] = {
            'total': len(class_preds),
            'uncertain': len(uncertain_preds),
            'uncertain_ratio': len(uncertain_preds) / len(class_preds) if class_preds else 0
        }

    print(f"\n  {'Class':<25} {'Total':>8} {'Uncertain':>10} {'Ratio':>8}")
    print("  " + "-" * 55)

    for class_name in sorted(class_uncertainty.keys(), key=lambda x: class_uncertainty[x]['uncertain_ratio'], reverse=True):
        stats = class_uncertainty[class_name]
        print(f"  {class_name:<25} {stats['total']:>8} {stats['uncertain']:>10} {stats['uncertain_ratio']:>7.1%}")

    # ========== MOST CONFUSED CLASS PAIRS ==========
    print("\n" + "=" * 70)
    print("  MOST CONFUSED CLASS PAIRS (in uncertain predictions)")
    print("=" * 70)

    confusion_pairs = {}
    for p in uncertain_2:
        classes = sorted(p['multi_label'])
        pair = f"{classes[0]} <-> {classes[1]}"
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    print(f"\n  {'Class Pair':<50} {'Count':>8}")
    print("  " + "-" * 60)

    for pair, count in sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {pair:<50} {count:>8}")

    # ========== ACCURACY ON UNCERTAIN SAMPLES ==========
    print("\n" + "=" * 70)
    print("  ACCURACY ON UNCERTAIN SAMPLES")
    print("=" * 70)

    # Certain samples
    certain = [p for p in predictions if len(p['multi_label']) == 1]
    certain_correct = sum(1 for p in certain if p['primary_prediction'] == p['true_class'])
    certain_acc = certain_correct / len(certain) if certain else 0

    # Uncertain samples
    uncertain_all = [p for p in predictions if len(p['multi_label']) >= 2]
    uncertain_correct = sum(1 for p in uncertain_all if p['primary_prediction'] == p['true_class'])
    uncertain_acc = uncertain_correct / len(uncertain_all) if uncertain_all else 0

    # But check if true class is in multi-label
    uncertain_in_multilabel = sum(1 for p in uncertain_all if p['true_class'] in p['multi_label'])
    multilabel_hit_rate = uncertain_in_multilabel / len(uncertain_all) if uncertain_all else 0

    print(f"\n  Certain samples (1 label):")
    print(f"    Total: {len(certain)}")
    print(f"    Accuracy: {certain_acc:.2%} ({certain_correct}/{len(certain)})")

    print(f"\n  Uncertain samples (2+ labels):")
    print(f"    Total: {len(uncertain_all)}")
    print(f"    Primary prediction accuracy: {uncertain_acc:.2%} ({uncertain_correct}/{len(uncertain_all)})")
    print(f"    True class in multi-label: {multilabel_hit_rate:.2%} ({uncertain_in_multilabel}/{len(uncertain_all)})")

    # ========== EXAMPLES ==========
    print("\n" + "=" * 70)
    print("  EXAMPLE UNCERTAIN PREDICTIONS (Top 10 most uncertain)")
    print("=" * 70)

    # Sort by confidence gap (smallest gap = most uncertain)
    uncertain_with_gap = [(p, gap) for p, gap in zip(predictions, confidence_gaps)]
    uncertain_sorted = sorted(uncertain_with_gap, key=lambda x: x[1])

    for i, (p, gap) in enumerate(uncertain_sorted[:10]):
        correct = "[OK]" if p['primary_prediction'] == p['true_class'] else "[WRONG]"

        print(f"\n  Sample {p['sample_index']} {correct} (Gap: {gap:.4f})")
        print(f"    True: {p['true_class']}")
        print(f"    Predicted: {p['primary_prediction']} (conf: {p['primary_confidence']:.4f})")
        print(f"    Top 3:")
        for item in p['top3']:
            marker = "<--" if item['class'] == p['true_class'] else "   "
            print(f"      {item['class']:<25} {item['confidence']:.4f} {marker}")
        if len(p['multi_label']) > 1:
            print(f"    Multi-label: {p['multi_label']}")

    # ========== SAVE RESULTS ==========
    results = {
        'uncertainty_metrics': {
            'single_label': sum(1 for c in multi_label_counts if c == 1),
            'two_labels': len(uncertain_2),
            'three_plus_labels': len(uncertain_3plus)
        },
        'confidence_gap': {
            'mean': float(np.mean(confidence_gaps)),
            'median': float(np.median(confidence_gaps)),
            'std': float(np.std(confidence_gaps))
        },
        'class_wise_uncertainty': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                                       for kk, vv in v.items()}
                                  for k, v in class_uncertainty.items()},
        'confused_pairs': confusion_pairs,
        'accuracy_analysis': {
            'certain_samples': {
                'total': len(certain),
                'accuracy': float(certain_acc)
            },
            'uncertain_samples': {
                'total': len(uncertain_all),
                'primary_accuracy': float(uncertain_acc),
                'multilabel_hit_rate': float(multilabel_hit_rate)
            }
        }
    }

    output_file = config.RESULTS_DIR / "uncertain_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {output_file}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = analyze_uncertain_samples()
