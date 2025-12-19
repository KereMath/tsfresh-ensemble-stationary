# 🔍 Uncertain Samples Analysis

Bu analiz, ensemble modelinin **belirsiz kaldığı** (birden fazla sınıfa yüksek confidence verdiği) sample'ları detaylı incelemeyi sağlar.

---

## 📊 Motivasyon

Bir modelin yanlış tahmin yapması iki farklı şekilde olabilir:

1. **Kesin ama yanlış**: Model %95 confidence ile yanlış tahmin yapar
2. **Belirsiz ve yanlış**: Model %52 vs %48 gibi kararsız kalır ve yanlış seçer

İkinci durumdaki sample'lar **öğrenme fırsatı** sunar:
- Bu sample'ları yeniden etiketleme yapılabilir
- Modelin karıştırdığı class pair'leri tespit edilebilir
- Belirsizlik yüksek olan sınıflar için özel stratejiler geliştirilebilir

---

## 🔧 Nasıl Çalışır?

### 1. Multi-Label Distribution

Threshold (varsayılan: 0.5) üzerindeki confidence'lara sahip tüm sınıflar sayılır:

```python
# Örnek 1: point_anomaly: 0.72, collective_anomaly: 0.52 -> Multi-label = 2 (belirsiz)
# Örnek 2: point_anomaly: 0.42, collective_anomaly: 0.38 -> Multi-label = 0 (hiçbiri yeterince yüksek değil)
# Örnek 3: point_anomaly: 0.92 -> Multi-label = 1 (kesin karar)

zero_labels = 1,718 sample (8.0%)      # Hiçbir sınıf threshold'u geçmemiş
single_label = 19,292 sample (89.4%)   # Kesin karar
two_labels = 565 sample (2.6%)         # İkili kararsızlık
three+ labels = 3 sample (0.0%)        # Çoklu kararsızlık
TOTAL = 21,578 sample (100%)
```

**Önemli**: %8 oranında sample'da **hiçbir sınıf %50 threshold'unu geçemiyor**! Bu sample'lar için model çok belirsiz.

### 2. Confidence Gap Analysis

1. ve 2. sıradaki tahminler arasındaki fark hesaplanır:

```python
gap = top1_confidence - top2_confidence

# Küçük gap (<0.2) = Çok belirsiz
# Orta gap (0.2-0.5) = Orta belirsiz
# Büyük gap (>0.5) = Kesin karar
```

**Sonuçlar**:
- **Mean gap**: 0.8410 (model genelde kesin karar veriyor)
- **Median gap**: 0.9978 (yarısından fazlası neredeyse %100 kesin)
- **Small gap (<0.2)**: 1,557 sample (%7.2) - en belirsiz durumlar
- **Large gap (>0.5)**: 18,295 sample (%84.8) - kesin kararlar

### 3. Class-Wise Uncertainty

Her sınıf için belirsizlik oranı hesaplanır:

| Sınıf | Toplam Sample | Belirsiz Sample | Belirsizlik Oranı |
|-------|--------------|-----------------|------------------|
| **collective_anomaly** | 2,397 | 148 | **6.2%** |
| **point_anomaly** | 2,398 | 123 | **5.1%** |
| **variance_shift** | 2,398 | 86 | **3.6%** |
| **volatility** | 2,398 | 79 | **3.3%** |
| **mean_shift** | 2,397 | 68 | **2.8%** |
| **trend_shift** | 2,398 | 39 | **1.6%** |
| **deterministic_trend** | 2,397 | 16 | **0.7%** |
| **stochastic_trend** | 2,398 | 9 | **0.4%** |
| **contextual_anomaly** | 2,397 | 0 | **0.0%** |

**Yorum**:
- `contextual_anomaly`: %100 kesin tahminler - model bu sınıfta hiç kararsız kalmıyor
- `collective_anomaly` ve `point_anomaly`: En belirsiz sınıflar - birbirleriyle sık karıştırılıyor

---

## 🎯 Most Confused Class Pairs

İkili belirsizlik (2 label) durumlarında hangi sınıf çiftleri sık karıştırılıyor:

| Sınıf Çifti | Karıştırma Sayısı |
|------------|------------------|
| **collective_anomaly ↔ point_anomaly** | **213** |
| point_anomaly ↔ variance_shift | 66 |
| point_anomaly ↔ volatility | 55 |
| variance_shift ↔ volatility | 49 |
| mean_shift ↔ trend_shift | 45 |
| mean_shift ↔ point_anomaly | 37 |
| collective_anomaly ↔ mean_shift | 33 |

**Kritik Bulgu**: `collective_anomaly` ve `point_anomaly` 213 kez karıştırılmış! Bu iki sınıf için özel feature engineering yapılabilir.

---

## 📈 Accuracy on Uncertain vs Certain Samples

### Certain Samples (Single Label)
```yaml
Total: 19,292
Accuracy: 92.84% (17,910/19,292)
```

Model kesin karar verdiğinde **%93 doğruluk** gösteriyor.

### Uncertain Samples (Multi-Label)
```yaml
Total: 568
Primary prediction accuracy: 56.16% (319/568)
True class in multi-label: 90.32% (513/568)
```

**Yorum**:
- Model belirsiz kaldığında primary prediction sadece **%56 doğru**
- Ama doğru cevap **%90 ihtimalle multi-label içinde var**!

**Pratik Sonuç**:
> Belirsiz durumlarda (2+ label), kullanıcıya "Bu sample şu sınıflardan biri olabilir: [X, Y]" şeklinde alternatifli cevap vermek %90 başarı sağlar.

---

## 🔬 Example: Most Uncertain Predictions

### Örnek 1: Çok Belirsiz (Gap: 0.0001)

```yaml
Sample 18204 [WRONG]
  True: collective_anomaly
  Predicted: mean_shift (conf: 1.52%)

  Top 3:
    mean_shift:            1.52%
    point_anomaly:         1.51%
    collective_anomaly:    1.38%  <-- Doğru cevap 3. sırada
```

**Yorum**: Model hiçbir sınıfa yüksek confidence veremiyor (hepsi <%5). Bu sample muhtemelen **mislabeled** veya **outlier**.

### Örnek 2: Neredeyse Berabere (Gap: 0.0002)

```yaml
Sample 247 [WRONG]
  True: mean_shift
  Predicted: point_anomaly (conf: 36.77%)

  Top 3:
    point_anomaly:    36.77%
    mean_shift:       36.75%  <-- 0.02% fark ile kaybetti!
    collective_anomaly: 6.53%
```

**Yorum**: Model iki sınıf arasında **neredeyse tam yarıya bölünmüş**. Bu sample **kesinlikle manuel incelenmeli**.

### Örnek 3: Çift Label (Gap: 0.0002)

```yaml
Sample 15441 [WRONG]
  True: collective_anomaly
  Predicted: point_anomaly (conf: 56.59%)

  Top 3:
    point_anomaly:       56.59%
    collective_anomaly:  56.57%  <-- Multi-label içinde!
    volatility:           2.30%

  Multi-label: [point_anomaly, collective_anomaly]
```

**Yorum**: Her iki sınıf da %56 confidence'a sahip (threshold: 0.5). Multi-label sistemi **doğru cevabı yakalamış**.

---

## 💡 Practical Applications

### 1. Active Learning Pipeline

```python
# Belirsiz sample'ları filtrele
uncertain = [p for p in predictions
             if len(p['multi_label']) >= 2]

# Bunları manuel etiketleme için işaretle
for p in uncertain:
    print(f"Sample {p['sample_index']}: "
          f"Model kararsız: {p['multi_label']}")
```

**Sonuç**: 568 sample manuel incelenerek model doğruluğu artırılabilir.

### 2. Confidence-Threshold Strategy

```python
# Eğer gap < 0.2 ise alternatifli cevap ver
if (top1_conf - top2_conf) < 0.2:
    return f"Bu sample {top1} veya {top2} olabilir"
else:
    return f"Bu sample {top1} (kesin)"
```

### 3. Class-Specific Handling

```python
# collective_anomaly ve point_anomaly için özel kontrol
if predicted in ['collective_anomaly', 'point_anomaly']:
    if confidence < 0.7:
        return "Bu iki sınıf sıkça karıştırılıyor, lütfen manuel kontrol edin"
```

---

## 📊 Özet Bulgular

| Metrik | Değer |
|--------|-------|
| **Zero Label Oranı** | **8.0%** (hiçbir sınıf >0.5) |
| **Kesin Karar Oranı** | **89.4%** (single label) |
| **Belirsiz Sample Oranı** | **2.6%** (2+ label) |
| **Kesin Sample Accuracy** | **92.84%** |
| **Belirsiz Sample Accuracy** | **56.16%** |
| **Multi-Label Hit Rate** | **90.32%** |
| **En Belirsiz Sınıflar** | collective_anomaly, point_anomaly |
| **En Kesin Sınıf** | contextual_anomaly (0% belirsiz) |
| **En Karıştırılan Çift** | collective ↔ point (213 kez) |

---

## 🚀 Kullanım

```bash
# Uncertain samples analizi çalıştır
python uncertain_analysis.py
```

**Gereksinimler**:
- `results/detailed_predictions.json` (önce `ensemble_with_confidence.py` çalıştırılmalı)

**Çıktılar**:
- `results/uncertain_analysis.json` - Özet sonuçlar
- Konsol: Detaylı analiz çıktıları

---

## 🎯 Ana Mesajlar

1. **%8 sample'da hiçbir sınıf threshold geçemiyor** (zero label) - çok belirsiz durumlar
2. **Model %89 oranında kesin karar veriyor** (single label)
3. **Sadece %2.6 sample'da iki sınıf arası kararsızlık** var (2+ label)
4. **Kesin kararlarda %93 doğruluk** var
5. **Belirsiz kararlarda primary %56 doğru**, ama **multi-label %90 doğru cevabı içeriyor**
6. **collective_anomaly ↔ point_anomaly** en çok karıştırılan çift (213 kez)
7. **contextual_anomaly** hiç belirsizlik göstermiyor (%100 kesin)

**Pratik Öneri**:
> Confidence gap < 0.2 olan sample'larda **multi-label prediction** kullanılarak %90 başarı elde edilebilir!

---

## 🔗 İlgili Dosyalar

- [uncertain_analysis.py](uncertain_analysis.py) - Ana analiz scripti
- [ensemble_with_confidence.py](ensemble_with_confidence.py) - Confidence tahmin sistemi
- [README_CONFIDENCE.md](README_CONFIDENCE.md) - Confidence sistemi açıklaması
- [results/uncertain_analysis.json](results/uncertain_analysis.json) - Analiz sonuçları
