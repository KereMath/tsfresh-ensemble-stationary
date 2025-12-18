"""
Top-K Accuracy Analysis
Calculates accuracy when considering top-K predictions (Top-1, Top-3, Top-5)
"""
import numpy as np
import json
from pathlib import Path
import config


def analyze_topk_accuracy():
    """
    Analyze Top-K accuracy from detailed_predictions.json
    """
    print("=" * 70)
    print("  TOP-K ACCURACY ANALYSIS")
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

    # ========== TOP-K ACCURACY ==========
    print("\n" + "=" * 70)
    print("  TOP-K ACCURACY METRICS")
    print("=" * 70)

    # Top-1 (Primary prediction)
    top1_correct = sum(1 for p in predictions if p['primary_prediction'] == p['true_class'])
    top1_accuracy = top1_correct / len(predictions)

    # Top-2 (True class in top 2 predictions)
    top2_classes = [[item['class'] for item in p['top3'][:2]] for p in predictions]
    top2_correct = sum(1 for i, p in enumerate(predictions)
                      if p['true_class'] in top2_classes[i])
    top2_accuracy = top2_correct / len(predictions)

    # Top-3 (True class in top 3 predictions)
    top3_classes = [[item['class'] for item in p['top3']] for p in predictions]
    top3_correct = sum(1 for i, p in enumerate(predictions)
                      if p['true_class'] in top3_classes[i])
    top3_accuracy = top3_correct / len(predictions)

    # Top-5 (True class in top 5 predictions)
    top5_classes = [sorted(p['all_confidences'].items(),
                          key=lambda x: x[1], reverse=True)[:5]
                   for p in predictions]
    top5_correct = sum(1 for i, p in enumerate(predictions)
                      if p['true_class'] in [cls for cls, _ in top5_classes[i]])
    top5_accuracy = top5_correct / len(predictions)

    print(f"\n  {'Metric':<20} {'Correct':>10} {'Total':>10} {'Accuracy':>12}")
    print("  " + "-" * 55)
    print(f"  {'Top-1 (Primary)':<20} {top1_correct:>10} {len(predictions):>10} {top1_accuracy:>11.2%}")
    print(f"  {'Top-2':<20} {top2_correct:>10} {len(predictions):>10} {top2_accuracy:>11.2%}")
    print(f"  {'Top-3':<20} {top3_correct:>10} {len(predictions):>10} {top3_accuracy:>11.2%}")
    print(f"  {'Top-5':<20} {top5_correct:>10} {len(predictions):>10} {top5_accuracy:>11.2%}")

    improvement_2 = ((top2_accuracy - top1_accuracy) / top1_accuracy) * 100
    improvement_3 = ((top3_accuracy - top1_accuracy) / top1_accuracy) * 100
    improvement_5 = ((top5_accuracy - top1_accuracy) / top1_accuracy) * 100

    print(f"\n  Improvement over Top-1:")
    print(f"    Top-2: +{improvement_2:.2f}% (absolute: +{top2_accuracy - top1_accuracy:.2%})")
    print(f"    Top-3: +{improvement_3:.2f}% (absolute: +{top3_accuracy - top1_accuracy:.2%})")
    print(f"    Top-5: +{improvement_5:.2f}% (absolute: +{top5_accuracy - top1_accuracy:.2%})")

    # ========== CLASS-WISE TOP-K ACCURACY ==========
    print("\n" + "=" * 70)
    print("  CLASS-WISE TOP-K ACCURACY")
    print("=" * 70)

    class_topk = {}
    for class_name in config.CLASSES:
        class_preds = [p for p in predictions if p['true_class'] == class_name]

        top1 = sum(1 for p in class_preds if p['primary_prediction'] == class_name)

        top2_cls = [[item['class'] for item in p['top3'][:2]] for p in class_preds]
        top2 = sum(1 for i, p in enumerate(class_preds) if class_name in top2_cls[i])

        top3_cls = [[item['class'] for item in p['top3']] for p in class_preds]
        top3 = sum(1 for i, p in enumerate(class_preds) if class_name in top3_cls[i])

        top5_cls = [sorted(p['all_confidences'].items(),
                          key=lambda x: x[1], reverse=True)[:5]
                   for p in class_preds]
        top5 = sum(1 for i in range(len(class_preds))
                  if class_name in [cls for cls, _ in top5_cls[i]])

        total = len(class_preds)
        class_topk[class_name] = {
            'total': total,
            'top1': top1,
            'top2': top2,
            'top3': top3,
            'top5': top5,
            'top1_acc': top1 / total if total > 0 else 0,
            'top2_acc': top2 / total if total > 0 else 0,
            'top3_acc': top3 / total if total > 0 else 0,
            'top5_acc': top5 / total if total > 0 else 0
        }

    print(f"\n  {'Class':<25} {'Top-1':>10} {'Top-2':>10} {'Top-3':>10} {'Top-5':>10}")
    print("  " + "-" * 70)

    for class_name in sorted(class_topk.keys(),
                            key=lambda x: class_topk[x]['top1_acc'],
                            reverse=True):
        stats = class_topk[class_name]
        print(f"  {class_name:<25} {stats['top1_acc']:>9.1%} {stats['top2_acc']:>9.1%} {stats['top3_acc']:>9.1%} {stats['top5_acc']:>9.1%}")

    # ========== MISCLASSIFIED SAMPLES ANALYSIS ==========
    print("\n" + "=" * 70)
    print("  MISCLASSIFIED SAMPLES THAT APPEAR IN TOP-3")
    print("=" * 70)

    misclassified = [p for p in predictions if p['primary_prediction'] != p['true_class']]
    in_top3 = [p for p in misclassified
              if p['true_class'] in [item['class'] for item in p['top3']]]

    print(f"\n  Total misclassified: {len(misclassified)}")
    print(f"  True class in Top-3: {len(in_top3)} ({100*len(in_top3)/len(misclassified):.1f}%)")

    not_in_top3 = [p for p in misclassified
                   if p['true_class'] not in [item['class'] for item in p['top3']]]
    print(f"  True class NOT in Top-3: {len(not_in_top3)} ({100*len(not_in_top3)/len(misclassified):.1f}%)")

    # ========== CONFIDENCE GAP FOR MISCLASSIFIED IN TOP-3 ==========
    print("\n" + "-" * 70)
    print("  CONFIDENCE ANALYSIS FOR MISCLASSIFIED IN TOP-3")
    print("-" * 70)

    for p in in_top3:
        # Find rank of true class
        top3_classes_list = [item['class'] for item in p['top3']]
        true_rank = top3_classes_list.index(p['true_class']) + 1

        # Find confidence of true class
        true_conf = next(item['confidence'] for item in p['top3']
                        if item['class'] == p['true_class'])

        p['true_rank'] = true_rank
        p['true_conf'] = true_conf
        p['conf_gap'] = p['primary_confidence'] - true_conf

    # Average confidence gap
    avg_gap = np.mean([p['conf_gap'] for p in in_top3])
    median_gap = np.median([p['conf_gap'] for p in in_top3])

    print(f"\n  Average confidence gap (predicted - true): {avg_gap:.4f}")
    print(f"  Median confidence gap: {median_gap:.4f}")

    # Rank distribution
    rank_2 = sum(1 for p in in_top3 if p['true_rank'] == 2)
    rank_3 = sum(1 for p in in_top3 if p['true_rank'] == 3)

    print(f"\n  True class rank distribution:")
    print(f"    Rank 2: {rank_2} ({100*rank_2/len(in_top3):.1f}%)")
    print(f"    Rank 3: {rank_3} ({100*rank_3/len(in_top3):.1f}%)")

    # ========== EXAMPLES ==========
    print("\n" + "=" * 70)
    print("  EXAMPLES: MISCLASSIFIED BUT TRUE CLASS IN TOP-3")
    print("=" * 70)

    # Sort by smallest confidence gap (closest calls)
    in_top3_sorted = sorted(in_top3, key=lambda x: x['conf_gap'])

    for i, p in enumerate(in_top3_sorted[:10]):
        print(f"\n  Sample {p['sample_index']} (Gap: {p['conf_gap']:.4f})")
        print(f"    True: {p['true_class']} (Rank {p['true_rank']}, Conf: {p['true_conf']:.4f})")
        print(f"    Predicted: {p['primary_prediction']} (Conf: {p['primary_confidence']:.4f})")
        print(f"    Top 3:")
        for item in p['top3']:
            marker = "<-- TRUE" if item['class'] == p['true_class'] else ""
            print(f"      {item['class']:<25} {item['confidence']:.4f} {marker}")

    # ========== SAVE RESULTS ==========
    results = {
        'topk_accuracy': {
            'top1': {
                'correct': top1_correct,
                'total': len(predictions),
                'accuracy': float(top1_accuracy)
            },
            'top2': {
                'correct': top2_correct,
                'total': len(predictions),
                'accuracy': float(top2_accuracy),
                'improvement_over_top1': float(improvement_2)
            },
            'top3': {
                'correct': top3_correct,
                'total': len(predictions),
                'accuracy': float(top3_accuracy),
                'improvement_over_top1': float(improvement_3)
            },
            'top5': {
                'correct': top5_correct,
                'total': len(predictions),
                'accuracy': float(top5_accuracy),
                'improvement_over_top1': float(improvement_5)
            }
        },
        'class_wise_topk': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                                for kk, vv in v.items()}
                           for k, v in class_topk.items()},
        'misclassified_analysis': {
            'total_misclassified': len(misclassified),
            'in_top3': len(in_top3),
            'in_top3_ratio': float(len(in_top3) / len(misclassified)) if misclassified else 0,
            'not_in_top3': len(not_in_top3),
            'avg_confidence_gap': float(avg_gap),
            'median_confidence_gap': float(median_gap),
            'rank_distribution': {
                'rank_2': rank_2,
                'rank_3': rank_3
            }
        }
    }

    output_file = config.RESULTS_DIR / "topk_accuracy.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {output_file}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = analyze_topk_accuracy()
