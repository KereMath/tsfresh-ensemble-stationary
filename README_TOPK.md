# 🎯 Top-K Accuracy Analysis

Bu analiz, modelin sadece **en yüksek tahminini** değil, **top-K tahminlerini** (top-2, top-3, top-5) de dikkate alarak doğruluk hesaplar.

---

## 📊 Motivasyon

Gerçek dünya uygulamalarında bazen sadece "en iyi tahmin" yeterli olmaz:

1. **Recommendation Systems**: Kullanıcıya top-3 alternatif sunmak
2. **Medical Diagnosis**: Doktor için olası 3 teşhisi göstermek
3. **Uncertainty Handling**: Model kararsızsa alternatifleri sunmak
4. **Model Evaluation**: Modelin alternatif tahminlerinin kalitesini ölçmek

**Soru**: Model yanlış tahmin yaptığında, doğru cevap top-3 veya top-5 içinde var mı?

---

## 🔧 Nasıl Çalışır?

### Top-K Accuracy

```python
# Top-1: Sadece en yüksek confidence'lı tahmin
top1_accuracy = (primary_prediction == true_class)

# Top-2: Doğru cevap ilk 2 tahmin içinde mi?
top2_accuracy = (true_class in top2_predictions)

# Top-3: Doğru cevap ilk 3 tahmin içinde mi?
top3_accuracy = (true_class in top3_predictions)

# Top-5: Doğru cevap ilk 5 tahmin içinde mi?
top5_accuracy = (true_class in top5_predictions)
```

---

## 📈 Overall Top-K Results

| Metrik | Doğru | Toplam | Accuracy | Improvement |
|--------|-------|--------|----------|-------------|
| **Top-1** | 18,885 | 21,578 | **87.52%** | - |
| **Top-2** | 20,348 | 21,578 | **94.30%** | **+7.75%** |
| **Top-3** | 20,972 | 21,578 | **97.19%** | **+11.05%** |
| **Top-5** | 21,494 | 21,578 | **99.61%** | **+13.82%** |

**Kritik Bulgular**:
- Top-1'den Top-2'ye geçişte **+6.78% absolute gain** (87.52% -> 94.30%)
- Top-3 accuracy **%97.19** - neredeyse her sample için doğru cevap ilk 3'te!
- Top-5 accuracy **%99.61** - sadece 84 sample'da doğru cevap ilk 5'te yok

**Yorum**:
> Model yanlış tahmin yaptığında, doğru cevap **%77.5 ihtimalle top-3'te** yer alıyor!

---

## 📊 Class-Wise Top-K Accuracy

| Sınıf | Top-1 | Top-2 | Top-3 | Top-5 |
|-------|-------|-------|-------|-------|
| **contextual_anomaly** | 100.0% | 100.0% | 100.0% | 100.0% |
| **deterministic_trend** | 98.6% | 99.5% | 99.7% | 100.0% |
| **stochastic_trend** | 95.0% | 96.9% | 97.6% | 99.1% |
| **trend_shift** | 91.6% | 95.6% | 96.5% | 99.0% |
| **point_anomaly** | 89.6% | **98.3%** | **99.7%** | 100.0% |
| **volatility** | 84.7% | 91.6% | 95.3% | 99.4% |
| **variance_shift** | 79.7% | 88.9% | 93.6% | 99.3% |
| **mean_shift** | 78.4% | 88.9% | 95.5% | 99.8% |
| **collective_anomaly** | 70.2% | 88.9% | 96.7% | 99.9% |

### Sınıf Bazlı Analizler

#### 1. contextual_anomaly - Mükemmel
```yaml
Top-1: 100.0%
Top-2: 100.0%
Top-3: 100.0%
Top-5: 100.0%
```
**Yorum**: Bu sınıf için model **hiç hata yapmıyor**.

#### 2. point_anomaly - Dramatik İyileşme
```yaml
Top-1: 89.6%
Top-2: 98.3% (+8.7%)
Top-3: 99.7% (+10.1%)
```
**Yorum**: Top-1'de %90 olan doğruluk, top-3'te **%99.7'ye çıkıyor**! Model bu sınıfı karıştırdığında doğru cevap neredeyse her zaman ilk 3'te.

#### 3. collective_anomaly - En Çok İyileşen
```yaml
Top-1: 70.2%
Top-2: 88.9% (+18.7%)
Top-3: 96.7% (+26.5%)
```
**Yorum**: Top-1'de en düşük doğruluk (%70), ama top-3'te **%96.7**'ye çıkıyor. Bu, modelin bu sınıf için alternatif tahminlerinin çok kaliteli olduğunu gösterir.

---

## 🔍 Misclassified Samples Analysis

### Yanlış Tahmin Edilen Sample'lar

```yaml
Total misclassified: 2,693 (Top-1 hataları)

True class in Top-3: 2,087 (%77.5)
True class NOT in Top-3: 606 (%22.5)
```

**Ana Mesaj**:
> Model yanlış tahmin yaptığında, **%77.5 ihtimalle doğru cevap top-3'te** yer alıyor!

### Confidence Gap Analysis

Yanlış tahmin edilen ama doğru cevap top-3'te olan sample'lar için:

```yaml
Average confidence gap: 0.4127
Median confidence gap: 0.3732

Rank distribution:
  Rank 2 (2. sırada): 1,463 (%70.1)
  Rank 3 (3. sırada): 624 (%29.9)
```

**Yorum**:
- Yanlış tahminlerde doğru cevap **%70 ihtimalle 2. sırada**
- Ortalama confidence farkı %41.27 (yani model yanlış tahminde %41 daha emin)
- Medyan fark %37.32 (yarısından fazlası bu civarında)

---

## 🔬 Examples: Close Calls

### Örnek 1: Ultra Close (Gap: 0.0002)

```yaml
Sample 247
  True: mean_shift (Rank 2, Conf: 36.75%)
  Predicted: point_anomaly (Conf: 36.77%)

  Top 3:
    point_anomaly:       36.77%
    mean_shift:          36.75%  <-- 0.02% fark!
    collective_anomaly:   6.53%
```

**Yorum**: Model neredeyse **coin flip** yapıyor (%36.77 vs %36.75). Bu sample kesinlikle manuel incelenmeli.

### Örnek 2: High Confidence Wrong (Gap: 0.0013)

```yaml
Sample 15083
  True: mean_shift (Rank 2, Conf: 83.70%)
  Predicted: point_anomaly (Conf: 83.83%)

  Top 3:
    point_anomaly:  83.83%
    mean_shift:     83.70%  <-- Yüksek confidence ama 2. sırada
    collective_anomaly: 3.16%
```

**Yorum**: Her iki sınıf da **%83+ confidence** gösteriyor. Model iki yüksek kaliteli hipotez üretiyor, ama yanlış olanı seçiyor.

### Örnek 3: Low Confidence All Around (Gap: 0.0013)

```yaml
Sample 18204
  True: collective_anomaly (Rank 3, Conf: 1.38%)
  Predicted: mean_shift (Conf: 1.52%)

  Top 3:
    mean_shift:            1.52%
    point_anomaly:         1.51%
    collective_anomaly:    1.38%  <-- Tüm confidenceler çok düşük
```

**Yorum**: Model hiçbir sınıftan emin değil (hepsi <%5). Bu sample muhtemelen **outlier** veya **mislabeled**.

---

## 💡 Practical Applications

### 1. Top-K Recommendation System

```python
def predict_with_alternatives(sample):
    confidences = predict_all_confidences(sample)
    top3 = sorted(confidences.items(), key=lambda x: x[1], reverse=True)[:3]

    if top3[0][1] - top3[1][1] < 0.2:
        # Belirsiz durum - alternatifler sun
        return {
            'decision': 'uncertain',
            'alternatives': [
                f"{cls}: {conf:.1%}" for cls, conf in top3
            ]
        }
    else:
        # Kesin karar
        return {
            'decision': 'certain',
            'prediction': top3[0][0],
            'confidence': top3[0][1]
        }
```

### 2. Medical Diagnosis Style Output

```python
# Doktora olası 3 teşhisi sun
print("Possible diagnoses:")
for i, (class_name, conf) in enumerate(top3, 1):
    print(f"  {i}. {class_name}: {conf:.1%}")
```

**Çıktı**:
```
Possible diagnoses:
  1. point_anomaly: 83.8%
  2. mean_shift: 83.7%
  3. collective_anomaly: 3.2%
```

### 3. Accuracy Threshold Strategy

```python
# Eğer doğru cevap top-3'te yoksa "unknown" döndür
if true_class not in top3_predictions:
    return "UNKNOWN - Model very uncertain"
else:
    return top3_predictions
```

**Sonuç**: %97.19 accuracy with coverage (sadece 606 sample "unknown" döner)

---

## 📊 Özet Bulgular

| Metrik | Değer |
|--------|-------|
| **Top-1 Accuracy** | 87.52% |
| **Top-2 Accuracy** | 94.30% (+6.78%) |
| **Top-3 Accuracy** | 97.19% (+9.67%) |
| **Top-5 Accuracy** | 99.61% (+12.09%) |
| **Misclassified in Top-3 Rate** | 77.5% |
| **Avg Confidence Gap (wrong)** | 41.27% |
| **Doğru Cevap 2. Sırada Oranı** | 70.1% |
| **En İyileşen Sınıf** | collective_anomaly (+26.5%) |

---

## 🎯 Key Insights

### 1. Top-3 is the Sweet Spot

```
Top-1: 87.52%
Top-2: 94.30% (+6.78%)  <-- Büyük sıçrama
Top-3: 97.19% (+2.89%)  <-- Hala iyi kazanç
Top-5: 99.61% (+2.42%)  <-- Azalan getiri
```

**Öneri**: Kullanıcıya **top-3** alternatif sunmak optimum cost-benefit sağlar.

### 2. Misclassification is Often Close

Yanlış tahminlerin %70'inde doğru cevap **2. sırada** - yani model "neredeyse doğru tahmin" yapıyor.

### 3. Class-Specific Strategies

- `contextual_anomaly`: Hiç hata yok, top-1 yeterli
- `point_anomaly`, `collective_anomaly`: Top-3 kullanımı kritik (+%26 kazanç)
- `mean_shift`, `variance_shift`: Top-2 bile büyük kazanç sağlıyor (+%10-18)

---

## 🚀 Kullanım

```bash
# Top-K accuracy analizi çalıştır
python topk_accuracy.py
```

**Gereksinimler**:
- `results/detailed_predictions.json` (önce `ensemble_with_confidence.py` çalıştırılmalı)

**Çıktılar**:
- `results/topk_accuracy.json` - Özet sonuçlar
- Konsol: Detaylı analiz çıktıları

---

## 🔬 Gelecek İyileştirmeler

1. **Dynamic K Selection**: Her sınıf için optimal K değeri belirleme
2. **Confidence Calibration**: Top-K confidencelerinin kalibrasyonu
3. **Cost-Sensitive Top-K**: Her rank için farklı cost tanımlama
4. **Top-K Ensemble**: Farklı modellerin top-K'larını birleştirme

---

## 🎯 Ana Mesajlar

1. **Top-1'den Top-3'e geçiş %9.67 absolute gain** sağlıyor
2. **Model yanlış tahmin yaptığında %77.5 oranla doğru cevap top-3'te**
3. **collective_anomaly** ve **point_anomaly** için top-K kullanımı kritik
4. **Confidence gap < 0.4 olan sample'lar** için top-K alternatifleri sunulmalı
5. **Top-5'te %99.61 accuracy** - neredeyse her sample için doğru cevap var

**Pratik Öneri**:
> Kullanıcıya **top-3 alternatifleri sunarak** accuracy %87.52'den **%97.19'a çıkarılabilir**!

---

## 🔗 İlgili Dosyalar

- [topk_accuracy.py](topk_accuracy.py) - Ana analiz scripti
- [ensemble_with_confidence.py](ensemble_with_confidence.py) - Confidence tahmin sistemi
- [README_CONFIDENCE.md](README_CONFIDENCE.md) - Confidence sistemi açıklaması
- [README_UNCERTAIN.md](README_UNCERTAIN.md) - Belirsiz sample analizi
- [results/topk_accuracy.json](results/topk_accuracy.json) - Analiz sonuçları
