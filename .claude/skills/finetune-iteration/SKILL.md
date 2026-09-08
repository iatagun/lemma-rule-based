---
name: finetune-iteration
description: DizgeBERT ince-ayar deney disiplini — kıyas dosyalarını dondurma, apples-to-apples ölçüm, checkpoint hijyeni, hangi benchmark'ın neyi ölçtüğü, küçük sınıflandırıcı overfit oyun kitabı (katman dondurma + seçim metriğini dağıtım metriğine eşitleme), bu repodaki ortam/harness tuzakları. Trigger — yeni bir model varyantı/encoder/hiperparametre deneme, eğitim koşusu başlatma, sonuçları kıyaslama, "vN çalıştır", overfit/eşik/epoch-seçimi sorunu.
allowed-tools: Bash, Read, Edit, Write, Grep, Glob
user-invokable: true
---

# DizgeBERT ince-ayar deney disiplini

Bu repoda 8+ başarısız + 2 başarılı varyant denemesinden çıkan yöntem. Amaç: her deneyi
**karşılaştırılabilir** kılmak ve "kazanç" ile "artefakt"ı ayırmak.

## Kıyas disiplini (her deneyde)

1. **Eval dosyası SABİT.** v5/v6/.../v13 hepsi aynı `idiom_data/tdk_examples_test.json`
   (frozen, 312 deyim) üzerinde ölçüldü. Split shuffle-slice ise deyim sayısı değişince
   permütasyon kayar → frozen-split modu ekle (mevcut dev/test dosyaları varsa sabitle,
   yeni örnekler yalnız train'e).
2. **Reçete SABİT, tek değişken.** Encoder A/B'de yalnız `--encoder`; veri A/B'de yalnız
   veri bayrağı. v5 reçetesi = `--class-weights --tdk-examples --epochs 10`.
3. **Checkpoint hijyeni.** Eğitim `<data>/best_*_tagger.pt`'yi ÜZERİNE YAZAR. Koşu bitince
   HEMEN `cp best_X.pt best_X_vN_<etiket>.pt`; kanoniği geri yükle. Kanonik = `best_*_tagger.pt`
   (yayınlanan), yedeği ayrı isimle.
4. **Her sonucu memory'e yaz** — reddedilenler dahil (`memory/dizgebert-<x>-project.md`).
   "vN: <ne>, <sonuç tablosu>, neden reddedildi". Aynı şeyi iki kez deneme.

## Hangi benchmark neyi ölçer (DizgeBERT-Idiom örneği — genelle)

| benchmark | ölçtüğü | tuzak |
|---|---|---|
| PARSEME test (3304 c.) | **yalnız span bulma** — TÜM gold span'ler idyomatik, literal kullanım YOK | F1 artışı çoğu zaman bir **recall artefaktı**; filtre eklemek burada = yapay false-negative |
| TDK held-out (frozen) | zayıf-etiketli span, gerçek literal negatifi yok | precision düşüşü gerçek sinyal; recall şişmesi şüpheli |
| Çavuşoğlu & Çöltekin (198 çift) | **bağlamda idyomatik/literal AYRIMI** (asıl zor problem) | 198 çift → ±1.5 puan gürültü; 3 metriğe birden bak (duyarlılık, yanlış-poz, doğru-ayırt) |
| CASES / GLU tanı (elle) | nokta-atışı sağlık kontrolü | GLU'nun 9 minimal çifti bazı sürümlerin EĞİTİMİNDE → kontamine |

**Genel kural:** bir benchmark'ta sadece bir eksen (P veya R) değişiyorsa, o "kazanç" muhtemelen
takas. Precision zaafını ölçmek için literal-negatifli, GÖRÜLMEMİŞ-deyim bir set şart.

## Tükenmiş kaldıraçlar (DizgeBERT-Idiom — tekrar deneme)

- **Encoder swap** (ConvBERTurk-mC4): precision tavanını kırmadı, recall'a kaydı.
- **Daha çok pozitif weak-supervision verisi** (TDK crawl, Leipzig madenciliği): her seferinde
  precision→recall takası. Weak-label yüzey-biçim eşleşmesi; idyomatikliği öğretmiyor.
- **Tek-modelde idyomatik/literal ayrımı** (L→hep-O örnekleri; ayrı `-LIT` etiket sınıfı):
  görülmemiş deyimde genellemedi (7-yönlü softmax güçlü B-VID yüzey prior'unu yenemiyor).
- **Recall-boost stage-1** (`--span-weight-mult`): span modeli olarak iyi (TDK F1 +2.4) ama
  iki-aşama boru hattında stage-2'nin filtreleyebileceğinden çok literal-FP ekliyor.

## Çalışan yaklaşım: İKİ AŞAMA (detect → filter)

Yüksek-recall span dedektörü (stage-1) → her aday için ikili sınıflandırıcı (stage-2:
{idyomatik, literal}) → güvenli literal'i ELE. Ayrı görev, ayrı gövde; span modelini bozmaz.
Çavuşoğlu yanlış-poz %25→%16, doğru-ayırt %37→%42. Pakete gömülü (bkz. [[hf-model-publish]]).

## Küçük sınıflandırıcı overfit oyun kitabı

Belirti: ~1k örnekte tam fine-tune → train loss → ~0.0007, held-out sabit, `--stage2-thresh`
ölü (softmax doygun, hep ~0/1).

1. **Alt katmanları dondur.** `--freeze 8` (embeddings + alttan 8 transformer katmanı) →
   110M → 28M trainable. `freeze 6` yetersiz, `freeze 10` underfit. Loss artık ~0.4'te doyar
   → eşik taraması geri gelir.
2. **Seçim metriğini DAĞITIM metriğine eşitle.** `(idyom_F1 + literal_eleme)/2` literal_eleme'yi
   aşırı ödüllendirip aşırı-temkinli epoch seçti (held-out idyom_R %76). Boru hattının "doğru-ayırt"
   metriği = iki sınıfın recall'ı dengeli → `(idyom_R + literal_eleme)/2` (dengeli doğruluk) →
   doğru epoch (idyom_R %90 + literal_eleme %82).
3. Eşik ayarı yalnız softmax doygun DEĞİLSE anlamlı. Doygunsa önce (1)+(2).

## Ortam / harness tuzakları (bu repo)

- **pytest:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q` — çıplak `pytest`
  `opentelemetry` autoload eklentisiyle ImportError veriyor (kod hatası değil).
- **Uzun eğitim:** `nohup python -u training/train_X.py ... > <data>/_vN_train.log 2>&1 &`
  (`-u` şart — nohup redirect'te Python block-buffer'lar). Sonra **Monitor** + until-loop ile
  bitiş satırını bekle (`Best selection score` / `Traceback` / `Killed`). Foreground `sleep` bloklu.
- **Süreç kontrolü:** `tasklist | grep python` asla çıkmıyor (MCP server pid hep açık).
  `powershell -NoProfile -c "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_X*' }).Count"`.
- **Disk:** her tagger checkpoint ~420MB, stage-2 ~440MB, iki-aşama bundle ~880MB. Oturum
  disk'i %100 doldurabilir → reddedilen deney checkpoint'lerini agresif sil (hepsi gitignore,
  bulgular memory'de, eğitim scriptleri commit'li). Kanonik + yedek + son aday'ı tut.
- **Arka plan bash** bellek baskısında öldürülüyor; nohup'lu python orphan olarak yaşar.
- **Scriptler repo kökünden:** `python training/train_X.py` (alt dizin). Yeni script eklerken
  `PROJECT_ROOT = Path(__file__).resolve().parent.parent` + `sys.path.insert(0, str(PROJECT_ROOT))`
  ve bunu `from dizgebert_*` / `from training.` / `from data.` importlarından ÖNCE koy.

## İlgili

[[hf-model-publish]] · [[idiom]] (DizgeBERT-Idiom durumu + GLU karar çerçevesi) · [[dep_parsing]]
