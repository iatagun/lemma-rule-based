# Karpathy-Inspired Guidelines

> Kaynak: https://github.com/multica-ai/andrej-karpathy-skills  
> Temel: Andrej Karpathy'nin LLM coding pitfalls gözlemleri

---

## Problemler

> "The models make wrong assumptions on your behalf and just run along with them without checking. They don't manage their confusion, don't seek clarifications, don't surface inconsistencies, don't present tradeoffs, don't push back when they should."

> "They really like to overcomplicate code and APIs, bloat abstractions, don't clean up dead code... implement a bloated construction over 1000 lines when 100 would do."

> "They still sometimes change/remove comments and code they don't sufficiently understand as side effects, even if orthogonal to the task."

---

## 4 Prensip

### 1.think_before_coding

**Varsayım yapma. Karışıklığı gizleme. Trade-off'ları ortaya çıkar.**

- **Varsayımları açıkça belirt** — Emin değilsen, sor
- **Birden fazla yorum sun** — Belirsizlik varsa sessizce seçme
- **Gerekirse itiraz et** — Daha basit bir yol varsa söyle
- **Karıştığında dur** — Ne belirsiz olduğunu adlandır ve sor

### 2.simplicity_first

**Problemi çözen minimum kod. Spekülatif bir şey yok.**

- İstenen ötesinde özellik yok
- Tek kullanımlık soyutlama yok
- İstenmeyen "esneklik" veya "yapılandırılabilirlik" yok
- İmkansız senaryolar için error handling yok
- 200 satır 50 olabiliyorsa yeniden yaz

**Test:** Kıdemli bir mühendis "aşırı karmaşık" dese mi? Evetse sadeleştir.

### 3.surgical_changes

**Sadece gerekeni değiştir. Kendi pisliğini temizle.**

Mevcut kodu düzenlerken:
- Yan kod, yorumlar veya formatting'i "iyileştirme"
- Kırık olmayan şeyleri refactor etme
- Mevcut stili kullan (farklı yapardın dese bile)
- Önceden var olan kod silme (sorulsun)

Değişikliklerinin yarattığı orphan'ları temizle:
- Değişikliklerin kullanılmayan import/variable/function'larını kaldır
- Önceden var olan dead code'ı sormadan silme

**Test:** Her değişen satır doğrudan kullanıcının isteğine trace edilmeli.

### 4.goal_driven_execution

**Başarı kriterlerini tanımla. Doğrulayana kadar döngüye devam et.**

| Yerine | Dönüştür |
|-------|----------|
| "Validation ekle" | "Test yaz → invalid input'lar için → geçir" |
| "Bug düzelt" | "Reproduce eden test yaz → geçir" |
| "X'i refactor et" | "Test geçsin önce ve sonra" |

Çok adımlı görevler için kısa plan:
```
1. [Adım] → doğrula: [kontrol]
2. [Adım] → doğrula: [kontrol]
3. [Adım] → doğrula: [kontrol]
```

**Güçlü başarı kriteri** = Bağımsız döngü gerektirir  
**Zayıf başarı kriteri** = ("çalıştır") = Sürekli açıklama gerekli

---

## Bu Projede Uygulama

```
Morphology Analiz Görevleri:
- "elma için lemma bul" → "gold standard ile doğrula"
- "benchmark çalıştır" → "sonuçları raporla, UAS/LAS"

Code Değişiklikleri:
- "bug düzelt" → "önce test yaz"
- "yeni ek ekle" → "eklendikten sonra benchmark doğrula"
- "refactor yapma" → "test geçerse bırak"

Sorulacak Sorular:
- "Bu değişikliğin kapsamı tam olarak ne?"
- "Hangi test geçerse başarılı?"
- "Bu abstraction gerekli mi?"
```

---

## Trade-off Notu

Bu prensipler **temkinli > hızlı** yönünde. Trivial görevler için (basit typo düzeltmeler, açık bir-liner) yargı kullan — her değişiklik tam rigor gerektirmez.

Amaç: Non-trivial işlerde pahalı hataları azaltmak, basit işleri yavaşlatmak değil.