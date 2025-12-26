# 🧪 Multi-Label Combination Testing

Bu modül, ensemble modelinin **multi-label prediction** yeteneğini ground-truth combination data üzerinde test eder.

---

## 🎯 Amaç

Combinations klasöründeki 21 kombinasyonun her biri **kesin olarak 2 label** içerir:
1. `deterministic_trend` (base trend için)
2. Bir anomaly tipi (`collective_anomaly`, `mean_shift`, `point_anomaly`, `trend_shift`, `variance_shift`)

Bu test, modelin:
- **Her iki label'ı da doğru tahmin edip edemediğini** (Full Match)
- **En az bir label'ı doğru tahmin edip edemediğini** (Partial Match)
- **Hiçbir label'ı tahmin edemediğini** (No Match)
- **Her bir label için detection rate**'ini ölçer

---

## 📁 Dosya Yapısı

```
combitesting/
├── combination_mapping.py          # 21 kombinasyonun label mapping'i
├── test_multilabel_combinations.py # Ana test scripti
├── README.md                        # Bu dosya
└── results/
    └── multilabel_combination_test.json  # Test sonuçları
```

---

## 🗺️ Combination Mapping

Tüm 21 kombinasyon için expected labels:

### Cubic Base (4)
| Folder | Labels |
|--------|--------|
| `cubic_collective_anomaly` | `['deterministic_trend', 'collective_anomaly']` |
| `Cubic + Mean Shift` | `['deterministic_trend', 'mean_shift']` |
| `Cubic + Point Anomaly` | `['deterministic_trend', 'point_anomaly']` |
| `Cubic + Variance Shift` | `['deterministic_trend', 'variance_shift']` |

### Damped Base (4)
| Folder | Labels |
|--------|--------|
| `Damped + Collective Anomaly` | `['deterministic_trend', 'collective_anomaly']` |
| `Damped + Mean Shift` | `['deterministic_trend', 'mean_shift']` |
| `Damped + Point Anomaly` | `['deterministic_trend', 'point_anomaly']` |
| `Damped + Variance Shift` | `['deterministic_trend', 'variance_shift']` |

### Exponential Base (4)
| Folder | Labels |
|--------|--------|
| `exponential_collective_anomaly` | `['deterministic_trend', 'collective_anomaly']` |
| `Exponential + Mean Shift` | `['deterministic_trend', 'mean_shift']` |
| `exponential_point_anomaly` | `['deterministic_trend', 'point_anomaly']` |
| `exponential_variance_shift` | `['deterministic_trend', 'variance_shift']` |

### Linear Base (5)
| Folder | Labels |
|--------|--------|
| `Linear + Collective Anomaly` | `['deterministic_trend', 'collective_anomaly']` |
| `Linear + Mean Shift` | `['deterministic_trend', 'mean_shift']` |
| `Linear + Point Anomaly` | `['deterministic_trend', 'point_anomaly']` |
| `Linear + Trend Shift` | `['deterministic_trend', 'trend_shift']` |
| `Linear + Variance Shift` | `['deterministic_trend', 'variance_shift']` |

### Quadratic Base (4)
| Folder | Labels |
|--------|--------|
| `Quadratic + Collective anomaly` | `['deterministic_trend', 'collective_anomaly']` |
| `Quadratic + Mean Shift` | `['deterministic_trend', 'mean_shift']` |
| `Quadratic + Point Anomaly` | `['deterministic_trend', 'point_anomaly']` |
| `Quadratic + Variance Shift` | `['deterministic_trend', 'variance_shift']` |

---

## 📊 Test Metrikleri

### 1. Overall Match Statistics

```
Full Match (both labels correct):     XXX (XX.X%)
Partial Match (one label correct):    XXX (XX.X%)
No Match (no labels correct):         XXX (XX.X%)
```

**Anlamı:**
- **Full Match**: Model her iki label'ı da doğru tahmin etti
- **Partial Match**: Model sadece 1 label'ı doğru tahmin etti (diğerini kaçırdı veya yanlış ekledi)
- **No Match**: Model hiçbir label'ı doğru tahmin edemedi

---

### 2. Label-Wise Detection Rates

Her label için ayrı ayrı detection rate:

```
deterministic_trend          XXX/XXX (XX.X%)
collective_anomaly           XXX/XXX (XX.X%)
mean_shift                   XXX/XXX (XX.X%)
point_anomaly                XXX/XXX (XX.X%)
trend_shift                  XXX/XXX (XX.X%)
variance_shift               XXX/XXX (XX.X%)
```

**Anlamı:** O label true olduğunda, model onu ne sıklıkla tespit edebiliyor?

---

### 3. Combination-Wise Full Match Rates

Her kombinasyon için full match oranı:

```
cubic_collective_anomaly              XX/XX (XX.X%)
Cubic + Mean Shift                    XX/XX (XX.X%)
...
```

**Anlamı:** Hangi kombinasyonlar daha kolay/zor tahmin ediliyor?

---

### 4. Intersection Size Distribution

Kaç tane true label doğru tahmin edildi:

```
0 labels correct: XXXX (XX.X%)  <- No Match
1 labels correct: XXXX (XX.X%)  <- Partial Match
2 labels correct: XXXX (XX.X%)  <- Full Match
```

---

### 5. Predicted Label Count Distribution

Model kaç label tahmin etti:

```
0 labels predicted: XXXX (XX.X%)  <- Model hiçbir label vermedi (conf < 0.5)
1 labels predicted: XXXX (XX.X%)  <- Model tek label verdi
2 labels predicted: XXXX (XX.X%)  <- Model iki label verdi (EXPECTED)
3+ labels predicted: XXXX (XX.X%) <- Model fazla label verdi
```

---

## 🚀 Kullanım

### 1. Mapping'i Kontrol Et

```bash
cd "c:\Users\user\Desktop\STATIONARY\tsfresh ensemble\combitesting"
python combination_mapping.py
```

### 2. Test'i Çalıştır

```bash
python test_multilabel_combinations.py
```

**Gereksinimler:**
- Trained ensemble models: `tsfresh ensemble/trained_models/`
- Combination data: `c:/Users/user/Desktop/STATIONARY/Combinations/`

**Parametreler (kod içinde değiştirilebilir):**
- `samples_per_combo=50`: Her kombinasyondan kaç sample test edilecek
- `threshold=0.5`: Multi-label için confidence threshold

---

## 📈 Beklenen Sonuçlar

### İdeal Senaryo

```
Full Match: ~90%+
  - deterministic_trend detection: ~95%+
  - anomaly detection: ~85%+
```

### Gerçekçi Senaryo

```
Full Match: 60-80%
Partial Match: 15-30%
No Match: 5-10%
```

**Neden Partial Match olabilir?**
1. Model `deterministic_trend`'i bulur ama anomaly'yi kaçırır
2. Model anomaly'yi bulur ama `deterministic_trend` yerine `stochastic_trend` der
3. Model doğru 2 label'ı bulur ama 3. bir label daha ekler (fazla pozitif)

---

## 🔍 Detaylı Analiz

### Hangi label'lar zor?

Eğer `deterministic_trend` detection rate %95+ ama bazı anomaly'ler düşükse:
- O anomaly tipi için model yetersiz
- O anomaly için daha fazla training data gerekebilir

### Hangi kombinasyonlar zor?

Eğer bazı kombinasyonlar sürekli Partial Match veriyorsa:
- O base trend + anomaly kombinasyonu modeli zorluyor
- Feature engineering gerekebilir

### Model fazla label veriyor mu?

Eğer "3+ labels predicted" yüksekse:
- Threshold çok düşük (0.5'ten daha yüksek denenebilir)
- Model false positive veriyor

---

## 🎯 Ana Metrikler

| Metrik | Açıklama | İdeal Değer |
|--------|----------|-------------|
| **Full Match Rate** | Her iki label de doğru | >80% |
| **Partial Match Rate** | En az bir label doğru | <20% |
| **No Match Rate** | Hiçbiri doğru değil | <5% |
| **deterministic_trend Detection** | Trend label'ı bulma | >95% |
| **Anomaly Detection (avg)** | Ortalama anomaly bulma | >85% |
| **Avg Predicted Labels** | Kaç label veriyor | ~2.0 |

---

## 🔗 İlgili Dosyalar

- [combination_mapping.py](combination_mapping.py) - Label mapping'ler
- [test_multilabel_combinations.py](test_multilabel_combinations.py) - Test scripti
- [../config.py](../config.py) - Ensemble config
- [../README_UNCERTAIN.md](../README_UNCERTAIN.md) - Multi-label analizi
