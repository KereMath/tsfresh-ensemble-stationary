"""
Combination to Multi-Label Mapping
Maps all 21 combinations from Combinations folder to their expected multi-label outputs
"""

# All base trends are deterministic, so they get 'deterministic_trend' label
# Each combination also has one anomaly type

COMBINATION_MAPPING = {
    # ========== CUBIC BASE (4) ==========
    'cubic_collective_anomaly': ['deterministic_trend', 'collective_anomaly'],
    'Cubic + Mean Shift': ['deterministic_trend', 'mean_shift'],
    'Cubic + Point Anomaly': ['deterministic_trend', 'point_anomaly'],
    'Cubic + Variance Shift': ['deterministic_trend', 'variance_shift'],

    # ========== DAMPED BASE (4) ==========
    'Damped + Collective Anomaly': ['deterministic_trend', 'collective_anomaly'],
    'Damped + Mean Shift': ['deterministic_trend', 'mean_shift'],
    'Damped + Point Anomaly': ['deterministic_trend', 'point_anomaly'],
    'Damped + Variance Shift': ['deterministic_trend', 'variance_shift'],

    # ========== EXPONENTIAL BASE (4) ==========
    'exponential_collective_anomaly': ['deterministic_trend', 'collective_anomaly'],
    'Exponential + Mean Shift': ['deterministic_trend', 'mean_shift'],
    'exponential_point_anomaly': ['deterministic_trend', 'point_anomaly'],
    'exponential_variance_shift': ['deterministic_trend', 'variance_shift'],

    # ========== LINEAR BASE (5) ==========
    'Linear + Collective Anomaly': ['deterministic_trend', 'collective_anomaly'],
    'Linear + Mean Shift': ['deterministic_trend', 'mean_shift'],
    'Linear + Point Anomaly': ['deterministic_trend', 'point_anomaly'],
    'Linear + Trend Shift': ['deterministic_trend', 'trend_shift'],
    'Linear + Variance Shift': ['deterministic_trend', 'variance_shift'],

    # ========== QUADRATIC BASE (4) ==========
    'Quadratic + Collective anomaly': ['deterministic_trend', 'collective_anomaly'],
    'Quadratic + Mean Shift': ['deterministic_trend', 'mean_shift'],
    'Quadratic + Point Anomaly': ['deterministic_trend', 'point_anomaly'],
    'Quadratic + Variance Shift': ['deterministic_trend', 'variance_shift'],
}

# Reverse mapping: folder path to labels
def get_labels_from_folder(folder_name):
    """
    Get expected labels from combination folder name

    Args:
        folder_name: Name of the combination folder (e.g., 'Cubic + Mean Shift')

    Returns:
        List of expected labels (always 2 labels)
    """
    if folder_name in COMBINATION_MAPPING:
        return COMBINATION_MAPPING[folder_name]
    else:
        raise ValueError(f"Unknown combination folder: {folder_name}")


# Statistics
def get_combination_stats():
    """Get statistics about combinations"""
    anomaly_counts = {}
    base_counts = {}

    for folder, labels in COMBINATION_MAPPING.items():
        # Count anomaly types (second label)
        anomaly = labels[1]
        anomaly_counts[anomaly] = anomaly_counts.get(anomaly, 0) + 1

        # Count base types (from folder name)
        if 'cubic' in folder.lower():
            base = 'Cubic'
        elif 'damped' in folder.lower():
            base = 'Damped'
        elif 'exponential' in folder.lower():
            base = 'Exponential'
        elif 'linear' in folder.lower():
            base = 'Linear'
        elif 'quadratic' in folder.lower():
            base = 'Quadratic'
        else:
            base = 'Unknown'

        base_counts[base] = base_counts.get(base, 0) + 1

    return {
        'total_combinations': len(COMBINATION_MAPPING),
        'anomaly_distribution': anomaly_counts,
        'base_distribution': base_counts
    }


if __name__ == "__main__":
    # Print mapping
    print("=" * 70)
    print("  COMBINATION TO MULTI-LABEL MAPPING")
    print("=" * 70)

    for i, (folder, labels) in enumerate(COMBINATION_MAPPING.items(), 1):
        print(f"\n{i:2d}. {folder}")
        print(f"    Labels: {labels}")

    # Print statistics
    stats = get_combination_stats()
    print("\n" + "=" * 70)
    print("  STATISTICS")
    print("=" * 70)
    print(f"\nTotal combinations: {stats['total_combinations']}")

    print("\nAnomaly distribution:")
    for anomaly, count in sorted(stats['anomaly_distribution'].items()):
        print(f"  {anomaly}: {count}")

    print("\nBase trend distribution:")
    for base, count in sorted(stats['base_distribution'].items()):
        print(f"  {base}: {count}")
