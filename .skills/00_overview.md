# Türkçe Morfolojik Çözümleyici — Mimari Genel Bakış

> Bu dosya, projeyi anlamak ve geliştirmek için temel başvuru kaynağıdır.

---

## ⚠️ Proje Durumu (Multi-Project)

Bu repo **iki bağımsız projeden** oluşur:

| Proje | Amaç | Durum |
|-------|------|-------|
| **morphology/** | Kural-tabanlı morfolojik çözümleme | ✅ Aktif, ~%88.6 doğruluk |
| **dep_bert/** | BERT+Biaffine dependency parsing | 🔄 Eğitim başladı |

**Farklı hedefler:** morphology = lemma/ek çözümleme, dep_bert = sözdizim ağacı

---

## Proje Özeti (Morphology)

**Amaç:** Türkçe sözcükleri kök + eklerine ayıran kural-tabanlı morfolojik çözümleyici

**Temel Yaklaşımlar:**
- Sağdan sola ek sıyırma (right-to-left suffix stripping)
- Ünlü/ünsüz uyumu kuralları (BÜU, KÜU)
- TDK sözlük desteği
- 4 katmanlı strateji sistemi

**Benchmark:** BOUN Treebank (UD_Turkish-BOUN) — 10,182 token
**Mevcut Doğruluk:** ~%88.6
**Pratik Tavan:** ~%90-92 (kural-tabanlı için)

---

## Dosya Yapısı

```
lemma-rule-based/
├── morphology/                 # Ana paket
│   ├── __init__.py            # Fabrika metodu, dışa aktarım
│   ├── phonology.py           # Ses sınıflandırma (ünlü/ünsüz kümeleri)
│   ├── harmony.py             # Ünlü/ünsüz uyumu kontrolü
│   ├── suffix.py              # Ek tanımları, şablon genişletme
│   ├── dictionary.py          # TDK sözlük, morfofonemik çözümleme
│   ├── analyzer.py            # Ana çözümleme motoru (~1700 satır)
│   ├── morphotactics.py       # Morfotaktik FSM (16 durum)
│   ├── validator.py           # Gövde/kök geçerlilik kontrolü
│   ├── sentence.py            # Cümle düzeyinde bağlamsal düzeltme
│   ├── formatter.py           # Çıktı biçimlendirme
│   └── dependency.py          # Bağımlılık analizi
├── benchmark/
│   ├── evaluate.py            # BOUN Treebank değerlendirme
│   └── test.conllu            # Gold standard test seti
├── turkish_words.txt          # 48,715 TDK madde başı
├── AGENTS.md                  # Agent talimatları (bu dosya)
└── skill.md                   # Detaylı uzman analizi
```

---

## Mimari İlkeler (SOLID)

| Prensip | Durum | Açıklama |
|---------|-------|----------|
| **SRP** | ⚠️ Dikkat | `analyzer.py` hem algoritma hem veri içerir |
| **OCP** | ✅ İyi | Yeni ekler `SuffixRegistry.register()` ile eklenir |
| **DIP** | ✅ İyi | `HarmonyChecker` Protocol — somut sınıflara bağlı değil |
| **LSP** | ✅ İyi | `StrictHarmony` ve `RelaxedHarmony` yer değiştirebilir |

---

## 4 Katmanlı Strateji Sistemi

```
Katman 1: StrictHarmony + Sözlük    → Tam uyumlu Türkçe sözcükler
Katman 2: RelaxedHarmony + Sözlük   → Alıntı sözcükler (saat, otobüs)
Katman 3: StrictHarmony + Sezgisel  → Sözlükte olmayanlar
Katman 4: RelaxedHarmony + Sezgisel → Son çare (fallback)
```

**Bu sıralama kasıtlıdır** — precision'dan recall'a doğru gradient.

---

## Temel Veri Yapıları

### MorphemeAnalysis
```python
@dataclass(frozen=True)
class MorphemeAnalysis:
    stem: str                    # Çözümlenmiş kök
    suffixes: list[tuple[str, str]]  # [(ek_formu, etiket), ...]
    root: str | None = None     # Morfofonemik kök (ünsüz yumuşaması vb.)
    lemma: str | None = None     # Lemma (mastar biçimi vb.)
```

### Şablon Değişkenleri
| Değişken | Karşılıklar | Kural |
|----------|-------------|-------|
| `{A}` | a, e | BÜU (2-yollu) |
| `{I}` | ı, i, u, ü | BÜU + KÜU (4-yollu) |
| `{D}` | d, t | Ötümsüz ünsüzden sonra t |
| `{C}` | c, ç | Ötümsüz ünsüzden sonra ç |

---

## API Kullanımı

```python
from morphology import create_default_analyzer

analyzer = create_default_analyzer(dictionary_path="turkish_words.txt")

# Tek sonuç
result = analyzer.analyze("geliyorum", upos="VERB")
print(result.stem)      # "gel"
print(result.suffixes)   # [("uyor", "ŞİMDİKİ_ZAMAN"), ("um", "KİŞİ_1T")]

# Tüm sonuçlar
results = analyzer.analyze_all("gelirin", upos="VERB", max_results=3)
```

---

## Hızlı Test

```powershell
$env:PYTHONIOENCODING = "utf-8"
python -X utf8 benchmark/evaluate.py
```
