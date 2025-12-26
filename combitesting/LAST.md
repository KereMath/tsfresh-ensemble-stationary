# Multi-Label Combination Testing - Final Results

## Overview
Testing the tsfresh ensemble model's ability to detect **multiple simultaneous labels** on 21 ground-truth combinations. Each combination contains exactly 2 labels: `deterministic_trend` (base) + one anomaly type.

---

## Test Objective

### What We're Testing
The ensemble model's capability to:
1. **Detect both labels** in combination data (Full Match)
2. **Detect at least one label** correctly (Partial Match)
3. **Measure per-label detection rates** across different anomaly types
4. **Evaluate which combinations are hardest** to detect

### Ground Truth Structure
All 21 combinations follow this pattern:
- **Label 1**: `deterministic_trend` (always present)
- **Label 2**: One of `{collective_anomaly, mean_shift, point_anomaly, trend_shift, variance_shift}`

---

## Test Configuration

### Dataset
- **Total Samples**: 210 CSV files
- **Combinations**: 21 unique (5 base trends × 4-5 anomaly types)
- **Samples per Combination**: 10
- **Data Source**: `C:\Users\user\Desktop\STATIONARY\Combinations\`

### Model Settings
- **Threshold**: 0.5 (confidence cutoff for multi-label prediction)
- **Model**: tsfresh ensemble (trained detectors)
- **Prediction Method**: Multi-label (can predict 0-N labels per sample)

### Combination Breakdown
| Base Trend | Anomaly Types | Count |
|------------|---------------|-------|
| Cubic | collective, mean_shift, point, variance | 4 |
| Damped | collective, mean_shift, point, variance | 4 |
| Exponential | collective, mean_shift, point, variance | 4 |
| Linear | collective, mean_shift, point, trend_shift, variance | 5 |
| Quadratic | collective, mean_shift, point, variance | 4 |
| **TOTAL** | | **21** |

---

## Final Results

### Overall Performance (210 samples)

| Metric | Count | Percentage |
|--------|-------|------------|
| **Full Match** (both labels correct) | **0** | **0.00%** |
| **Partial Match** (one label correct) | **88** | **41.90%** |
| **No Match** (no labels correct) | **122** | **58.10%** |

**🚨 CRITICAL ISSUE**: Zero full matches - model **never predicts both labels correctly**!

---

## Label-Wise Detection Rates

### Per-Label Performance
| Label | Total Expected | Detected | Detection Rate |
|-------|----------------|----------|----------------|
| **deterministic_trend** | 210 | 81 | **38.57%** |
| **collective_anomaly** | 50 | 0 | **0.00%** 🚫 |
| **mean_shift** | 50 | 4 | **8.00%** |
| **point_anomaly** | 50 | 0 | **0.00%** 🚫 |
| **trend_shift** | 10 | 0 | **0.00%** 🚫 |
| **variance_shift** | 50 | 3 | **6.00%** |

### Key Observations
- ✅ **deterministic_trend**: Only moderately detected (38.57%)
- 🚫 **collective_anomaly**: Completely missed (0%)
- 🚫 **point_anomaly**: Completely missed (0%)
- 🚫 **trend_shift**: Completely missed (0%)
- ⚠️ **mean_shift**: Rarely detected (8%)
- ⚠️ **variance_shift**: Rarely detected (6%)

**Average Anomaly Detection**: (0+8+0+0+6)/5 = **2.8%** (catastrophic failure)

---

## Combination-Wise Performance

### All Combinations: 0% Full Match Rate

| Combination | Samples | Full Matches |
|-------------|---------|--------------|
| All 21 combinations | 10 each | 0/10 (0%) |

**Every single combination**: **0% full match rate**

This indicates:
- Model systematically fails to detect multiple labels
- Either threshold too high or model confidence too low
- Multi-label prediction mechanism broken

---

## Prediction Behavior Analysis

### Intersection Size Distribution
How many true labels were correctly predicted?

| Correct Labels | Count | Percentage |
|----------------|-------|------------|
| **0 labels** (No Match) | 122 | **58.10%** |
| **1 label** (Partial Match) | 88 | **41.90%** |
| **2 labels** (Full Match) | 0 | **0.00%** |

**Interpretation**: Model **never** predicts both labels, even when both are present.

---

### Prediction Count Distribution
How many labels did the model predict per sample?

| Predicted Labels | Count | Percentage |
|------------------|-------|------------|
| **0 labels** | 101 | **48.10%** |
| **1 label** | 84 | **40.00%** |
| **2 labels** | 24 | **11.43%** |
| **3 labels** | 1 | **0.48%** |

**Interpretation**:
- 48% of the time: Model predicts **nothing** (all confidences < 0.5)
- 40% of the time: Model predicts **single label**
- 11% of the time: Model predicts **two labels** (but neither are correct full matches!)
- Model rarely over-predicts (only 1 case with 3 labels)

---

## Detailed Analysis

### Why Zero Full Matches?

#### Hypothesis 1: Threshold Too High (0.5)
- 48% of samples get **zero predictions** → Many confidences below 0.5
- Even when model predicts 2 labels (11%), they're not the correct pair
- **Action**: Try threshold = 0.3 or 0.2

#### Hypothesis 2: Model Lacks Multi-Label Capability
- Model trained on **single-label data** (Generated Data)
- Never saw combinations during training
- **Action**: Retrain on combination data with multi-label loss

#### Hypothesis 3: Anomaly Detectors Too Weak
- Collective anomaly: **0% detection**
- Point anomaly: **0% detection**
- Trend shift: **0% detection**
- **Action**: Improve individual detectors before ensemble

#### Hypothesis 4: deterministic_trend Confusion
- Only 38.57% detection for deterministic_trend (should be 95%+)
- Model may confuse with `stochastic_trend` or `normal`
- **Action**: Review trend detector training

---

### Comparison to Expected Performance

| Metric | Expected (Realistic) | Actual | Gap |
|--------|---------------------|--------|-----|
| Full Match Rate | 60-80% | **0%** | **-60 to -80%** |
| Partial Match Rate | 15-30% | 41.90% | +12 to +27% |
| No Match Rate | 5-10% | **58.10%** | **+48 to +53%** |
| deterministic_trend Detection | 95%+ | **38.57%** | **-56%** |
| Avg Anomaly Detection | 85%+ | **2.8%** | **-82%** |

**Verdict**: Model performance is **catastrophically below expectations**.

---

## Root Cause Analysis

### Primary Issues
1. **Individual detectors fail on combination data**
   - Training data: Single anomalies in isolation
   - Test data: Multiple anomalies + deterministic trends
   - **Distribution mismatch**

2. **Confidence calibration broken**
   - 48% get zero predictions (too conservative)
   - Average confidence likely < 0.5 for most labels

3. **Missing multi-label training**
   - Model never learned to output multiple labels
   - Binary classifiers assume one-hot encoding

### Secondary Issues
4. **Feature interference**: Combination features may differ from isolated anomaly features
5. **Threshold miscalibration**: 0.5 may be too high for this use case
6. **Ensemble voting**: May cancel out correct predictions

---

## What Worked (Partial Successes)

### deterministic_trend (38.57%)
- While low, it's the **best performing label**
- 81/210 correct detections
- Shows model has **some** capability

### Partial Matches (41.90%)
- Model gets **at least one label** correct 42% of the time
- Better than random (which would be ~25%)
- Indicates features have **some** signal

### Low False Positives
- Only 1 sample with 3 predictions (0.48%)
- Model not wildly over-predicting
- Conservative behavior (good for precision, bad for recall)

---

## Recommendations

### Immediate Actions (High Priority)
1. **Lower threshold to 0.3**
   - Re-run test with threshold=0.3
   - See if full match rate improves
   - Check if it's just a calibration issue

2. **Analyze confidence distributions**
   - Plot histogram of max confidences
   - Check if any samples have >0.5 for two labels
   - Determine if problem is threshold or model

3. **Test on single-anomaly data**
   - Validate model on Generated Data (single labels)
   - If that also fails, detectors are broken
   - If that works, it's a combination-specific issue

### Medium-Term Fixes
4. **Retrain with combination data**
   - Add Combinations to training set
   - Use multi-label loss (BCEWithLogitsLoss)
   - Balance single vs multi-label samples

5. **Improve weak detectors**
   - Retrain collective_anomaly detector (0% detection)
   - Retrain point_anomaly detector (0% detection)
   - Retrain trend_shift detector (0% detection)

6. **Add ensemble calibration**
   - Platt scaling for probability calibration
   - Temperature scaling
   - Isotonic regression

### Long-Term Strategy
7. **Multi-label architecture**
   - Replace binary classifiers with multi-label head
   - Use sigmoid outputs (not softmax)
   - Train end-to-end on combination data

8. **Hierarchical detection**
   - First detect base trend (deterministic vs stochastic)
   - Then detect anomaly type
   - Two-stage pipeline

9. **Expand training data**
   - Generate 3-label, 4-label combinations
   - Include normal + trend combinations
   - Cover full label space

---

## Files and Resources

### Test Files
- **Script**: [test_multilabel_combinations.py](test_multilabel_combinations.py)
- **Mapping**: [combination_mapping.py](combination_mapping.py)
- **Results**: [results/multilabel_combination_test.json](results/multilabel_combination_test.json)

### Related Documentation
- **README**: [README.md](README.md) - Test methodology
- **Ensemble Config**: [../config.py](../config.py)
- **Uncertainty Analysis**: [../README_UNCERTAIN.md](../README_UNCERTAIN.md)

---

## Conclusion

This multi-label combination test reveals **critical failures** in the ensemble model:

### Summary of Failures
- ❌ **0% full match rate** - Never predicts both labels correctly
- ❌ **58% no match rate** - Fails to predict any label correctly most of the time
- ❌ **2.8% avg anomaly detection** - Misses 97% of anomalies
- ❌ **38.57% trend detection** - Even base trend poorly detected

### Core Problem
The model was trained on **single-label data** (isolated anomalies) but tested on **multi-label data** (combinations). This distribution mismatch causes:
1. Feature interference (combined patterns differ from isolated)
2. Confidence collapse (model uncertain about multiple labels)
3. Missing multi-label capability (no joint training)

### Next Steps (Priority Order)
1. **Test threshold=0.3** to rule out calibration issue
2. **Validate on single-label data** to check if detectors work at all
3. **Retrain with combination data** using multi-label loss
4. **Rebuild weak detectors** (collective, point, trend_shift at 0%)

**Status**: 🚨 **NOT PRODUCTION READY** - Requires complete rework for multi-label scenarios

---

**Test Date**: December 2024
**Test Samples**: 210 combination CSVs
**Model**: tsfresh ensemble (single-label trained)
**Threshold**: 0.5
**Result**: ❌ **Failed** (0% full match, 58% no match)
