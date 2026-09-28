# Sonraki oturum planı (2026-09-28 sonu)

## Durum
- **En iyi model: v6-dp** = D:/dizgetts/runs/v6_dplin/ep400_dp_mse.pt (v6 ep150 + doğrusal-MSE süre tahmincisi + birikimli yuvarlama).
  Kör AB: v6-dp > v6 19-0 (11 fark yok); v6-dp > v7b 17-6.
- **Yayın ertelendi** (kullanıcı: "daha iyi sürümleri bekleyelim"). HF paketi D:/dizgetts/hf/DizgeTTS-Antalia hâlâ v6 ep150; commit 8641ad6 + etiket
  `dizgetts-antalia-v0` YEREL (push edilmedi). Yayın zamanı gelince: v6-dp (ya da daha iyisi) ile export_tts_hf --check, kartı güncelle, onayla push.
- **Bu oturumun kodu COMMIT EDİLMEDİ** (dizgetts-v2 çalışma ağacı): eval/{oracle,am_events,dp_response,boundary_rules_eval}.py, frontend/boundary_rules.py,
  train/flowdp.py, scripts/{train_flowdp,build_manifest_measured}.py, tools/{make_textgrid,read_breaks}.py, engine.py (BoundaryRuleStage), train/{train,dpfeat}.py
  (flow_dp, long_vowel_scale kancaları), train_dp --manifest, evaluate.py (text alanı, --splits boş), make_ab_test (metin düzeltmesi), docs, experiments.yaml.
  Kullanıcıya commit'i sor.

## Karar kuralı (kullanıcı onaylı)
Süre/ezgi deneylerinde karar = KÖR AB (tek soru "hangisi daha doğal"). CER/WER gerçekçi zamanlamayı cezalandırır (kahin B WER 14 vs A 8,3 ama 29-1 kazandı);
yalnız büyük bozulma izleme. **AB'den önce farkın duyulabilir olduğunu ölç** (dp_response / süre farkı); ince farklı AB'lerle kullanıcıyı yorma.

## Sıradaki seçenekler (kullanıcı seçecek)
1. **Arapça/Farsça alıntı sözcük eksikleri** (kullanıcı istedi, "her şey sırasıyla"): somut, kural/sözlük işi; önce kullanıcıdan örnek sözcükler al.
2. **Daha doğru deterministik süre modeli**: hedef log-süre korelasyonunu (val/test 0,49) artırmak. Girdi: sözcük içi konum, hece yapısı (açık/kapalı),
   vurgu, sözcük uzunluğu, sınır; daha büyük/bağlamlı ağ (şu an proj_w 2 katman). Karar öncesi iç ölçüm: korelasyon belirgin artmazsa AB yok.
3. **ğ uzun ünlüleri**: uzun ünlü (ː) gerçek 69 ms, v6-dp 59 (%85; train'de bile %87). long_vowel_scale kancası hazır (cfg.model.long_vowel_scale; 1,3
   train'de ölçüldü) ama v6-dp'ye UYGULANMADI/test edilmedi. Kullanıcı -dığında/doğa "hızlı" dedi. Dinleme seti D:/dizgetts/samples/v7b_digi/.

## Kullanıcı ezgi verisi
D:/dizgetts/user_prosody: 40 cümle (01-24 Antalia, 25-40 yeni) telefon kaydı + Praat `duraklama` etiketleri (0/1/2). Bu 40 = GELİŞTİRME seti.
Kullanıcının 1'i sessizliksiz (uzama/perde), 2'si ~250 ms. Antalia okuyucusuyla sınırların çoğu örtüşüyor. 41-48 okunmadı (nihai kör sınır ölçümü için ayrılabilir).
Praat yardımcısı: ac.praat (numara sor, aç, "Kaydet ve kapat").
