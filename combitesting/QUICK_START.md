# 🚀 Quick Start - Multi-Label Combination Testing

## Hızlı Kullanım

### 1. Mapping'i Kontrol Et

```bash
cd "c:\Users\user\Desktop\STATIONARY\tsfresh ensemble\combitesting"
python combination_mapping.py
```

**Çıktı:**
- 21 kombinasyonun her birinin multi-label mapping'i
- Anomaly distribution (her anomaly tipinden kaç tane var)
- Base trend distribution (her base'den kaç tane var)

---

### 2. Test'i Çalıştır

```bash
python test_multilabel_combinations.py
```

**Gereksinimler:**
✅ Trained models: `../trained_models/` klasöründe trained binary detector models olmalı
✅ Combination data: `C:/Users/user/Desktop/STATIONARY/Combinations/` klasöründe data olmalı

**Ne yapılıyor:**
1. Her kombinasyondan 50 sample yükleniyor (toplam ~1,050 sample)
2. Her sample için TSFresh features extract ediliyor (777 features)
3. Ensemble model ile multi-label prediction yapılıyor
4. Ground-truth ile karşılaştırılıyor
5. Comprehensive analiz yapılıyor

---

## 📊 Beklenen Çıktılar

### Console Output

```
======================================================================
  LOADING TRAINED ENSEMBLE MODELS
======================================================================

  Loaded collective_anomaly: lightgbm
  Loaded contextual_anomaly: xgboost
  ...

======================================================================
  LOADING COMBINATION SAMPLES
======================================================================

  cubic_collective_anomaly: Loading 50 samples...
  Cubic + Mean Shift: Loading 50 samples...
  ...

  Total samples loaded: 1050

======================================================================
  PREDICTING AND EVALUATING
======================================================================

  Predicting: 100%|████████████████| 1050/1050

======================================================================
  MULTI-LABEL PERFORMANCE ANALYSIS
======================================================================

  Overall Match Statistics:
    Full Match (both labels correct):       756 (72.0%)
    Partial Match (one label correct):      231 (22.0%)
    No Match (no labels correct):            63 ( 6.0%)

  Label-Wise Detection Rates:
    (How often each label is correctly detected when it's true)
    deterministic_trend             956/1050 (91.0%)
    point_anomaly                   198/210  (94.3%)
    variance_shift                  187/210  (89.0%)
    mean_shift                      189/210  (90.0%)
    collective_anomaly              203/250  (81.2%)
    trend_shift                      45/50   (90.0%)

  Combination-Wise Full Match Rates:
    Cubic + Point Anomaly                       42/50 (84.0%)
    Linear + Mean Shift                         40/50 (80.0%)
    ...

  Intersection Size Distribution:
    (How many true labels were predicted)
    0 labels correct:    63 ( 6.0%)
    1 labels correct:   231 (22.0%)
    2 labels correct:   756 (72.0%)

  Predicted Label Count Distribution:
    0 labels predicted:    48 ( 4.6%)
    1 labels predicted:   623 (59.3%)
    2 labels predicted:   357 (34.0%)
    3+ labels predicted:   22 ( 2.1%)

======================================================================

  Results saved to: results/multilabel_combination_test.json
```

---

### JSON Output

`results/multilabel_combination_test.json`:

```json
{
  "results": [
    {
      "combination_name": "cubic_collective_anomaly",
      "file_name": "cubic_collective_000.csv",
      "true_labels": ["deterministic_trend", "collective_anomaly"],
      "predicted_labels": ["deterministic_trend", "collective_anomaly"],
      "all_confidences": {
        "collective_anomaly": 0.8234,
        "deterministic_trend": 0.9123,
        "point_anomaly": 0.1234,
        ...
      },
      "full_match": true,
      "partial_match": false,
      "no_match": false,
      "intersection_size": 2
    },
    ...
  ],
  "analysis": {
    "overall": {
      "total": 1050,
      "full_match": 756,
      "full_match_rate": 0.72,
      ...
    },
    "label_wise": {
      "deterministic_trend": {
        "total": 1050,
        "detected": 956,
        "detection_rate": 0.91
      },
      ...
    },
    "combination_wise": { ... },
    "intersection_distribution": { ... },
    "prediction_size_distribution": { ... }
  }
}
```

---

## 🎯 Ana Metrikler Açıklaması

### Full Match Rate
**Ne:** Her iki label de doğru tahmin edildi
**İdeal:** >80%
**Örnek:** True: `[deterministic_trend, point_anomaly]`, Pred: `[deterministic_trend, point_anomaly]` ✅

### Partial Match Rate
**Ne:** Sadece 1 label doğru tahmin edildi
**İdeal:** <20%
**Örnek:** True: `[deterministic_trend, point_anomaly]`, Pred: `[deterministic_trend, variance_shift]` ⚠️

### No Match Rate
**Ne:** Hiçbir label doğru tahmin edilemedi
**İdeal:** <5%
**Örnek:** True: `[deterministic_trend, point_anomaly]`, Pred: `[stochastic_trend, variance_shift]` ❌

### Label-Wise Detection Rate
**Ne:** O label true olduğunda ne sıklıkla tespit ediliyor
**İdeal:** >85% (her label için)

### Prediction Size Distribution
**Ne:** Model kaç label tahmin ediyor
**İdeal:** Çoğunluk 2 labels predicted olmalı (çünkü ground-truth hep 2 label)

---

## ⚙️ Ayarlar

Test scriptinde değiştirilebilir parametreler:

```python
# test_multilabel_combinations.py içinde

# Her kombinasyondan kaç sample test edilecek (line ~483)
samples = load_combination_samples(combinations_dir, samples_per_combo=50)

# Multi-label threshold (line ~489)
results = evaluate_multilabel_predictions(samples, models, threshold=0.5)
```

**`samples_per_combo` değiştirirseniz:**
- `50` -> Hızlı test (~1,050 sample)
- `100` -> Orta test (~2,100 sample)
- `200` -> Comprehensive test (~4,200 sample)

**`threshold` değiştirirseniz:**
- `0.3` -> Daha fazla label verilir (recall artır)
- `0.5` -> Balanced (default)
- `0.7` -> Daha az label verilir (precision artır)

---

## 🔧 Sorun Giderme

### Error: "No module named 'tsfresh'"

```bash
pip install tsfresh
```

### Error: "trained_models/ not found"

Önce modelleri train etmelisiniz:

```bash
cd "c:\Users\user\Desktop\STATIONARY\tsfresh ensemble"
python trainer.py
```

### Error: "Combinations folder not found"

`test_multilabel_combinations.py` içinde path'i kontrol edin:

```python
combinations_dir = Path("c:/Users/user/Desktop/STATIONARY/Combinations")
```

---

## 📈 Sonuçları Yorumlama

### Senaryo 1: Full Match %90+
✅ Excellent! Model multi-label prediction'da çok başarılı

### Senaryo 2: Full Match %70-85, Partial Match %15-25
✅ Good. Model genelde 1 label doğru buluyor, ikincisinde zorlanıyor
➡️ Düşük detection rate'li label'lar için feature engineering yapılabilir

### Senaryo 3: Full Match <%60, No Match %10+
❌ Poor. Model multi-label prediction'da yetersiz
➡️ Model threshold'u düşürülebilir (0.5 -> 0.3)
➡️ Daha fazla training data gerekebilir
➡️ Class imbalance sorunu olabilir

### Senaryo 4: Predicted Labels çoğunluk 0 veya 1
❌ Model multi-label veremiyor (threshold çok yüksek veya model çok conservative)
➡️ Threshold'u düşür (0.5 -> 0.3)

### Senaryo 5: Predicted Labels çoğunluk 3+
❌ Model çok fazla label veriyor (threshold çok düşük veya model overconfident)
➡️ Threshold'u yükselt (0.5 -> 0.7)

---

## 🔗 İlgili Dosyalar

- [README.md](README.md) - Detaylı dokümantasyon
- [combination_mapping.py](combination_mapping.py) - Label mapping'ler
- [test_multilabel_combinations.py](test_multilabel_combinations.py) - Test scripti
- [../README_UNCERTAIN.md](../README_UNCERTAIN.md) - Multi-label analizi
- [../README_TOPK.md](../README_TOPK.md) - Top-K accuracy analizi
