"""Söyleyiş istisna sözlüğü (frontend/pronounce.py, resources/pronunciation_exceptions.tsv).
  python -X utf8 -m dizgetts.tests.test_pronounce
Neden: dizge `lexicon.loans` yalnız TAM sözcüğü düzeltiyor; kağıt düzeltmesi kağıdı'da kayboluyordu (kʰ ɑː ɨ d ɨ). Sözlük kök + ek zinciriyle eşler."""
import warnings

from dizgetts.engine import Engine
from dizgetts.frontend.phonemize import Phonemizer
from dizgetts.frontend.pronounce import Exceptions
from dizgetts.frontend.symbols import PHONES, tokenize

raw = Phonemizer(bert_fallback=False)                                    # istisnasız (varsayılan)
oz = Phonemizer(bert_fallback=False, pron_exceptions=True)               # özenli
gu = Phonemizer(bert_fallback=False, pron_exceptions=True, register="gündelik")
A = lambda ph, w: " ".join(tokenize(ph.word(w)))

# 0) kapalıyken dizge çıktısı BİREBİR aynı (eski checkpoint'ler / donmuş manifestler)
assert A(raw, "kağıdı") == "kʰ ɑː ɨ d ɨ" and A(raw, "hakim") == "x ɑː c I m"

# 1) kağıt: kök ve TÜM çekimli biçimler aynı ön a (dizge'de yalnız kök doğruydu)
for w in ("kağıt", "kâğıt", "kağıdı", "kâğıdı", "kağıda", "kağıdın", "kağıtlar", "kağıtları", "kağıtta", "kağıthane", "kağıtçı", "kâğıd"):
    assert A(oz, w).startswith("cʰ aː ɨ"), (w, A(oz, w))
assert A(oz, "kağıdı") == "cʰ aː ɨ d ɨ" and A(oz, "kağıt") == A(raw, "kağıt")  # kök zaten doğruydu: dokunulmaz

# 2) hakim / hâkim: uzun ama arka değil (aː); ek ne olursa
for w in ("hakim", "hâkim", "hakimi", "hakime", "hakimler", "hakimiyet", "hâkimiyet", "hakimlik"):
    assert A(oz, w).startswith("x aː c I m"), (w, A(oz, w))
assert A(oz, "hakim") == "x aː c I m"

# 3) yanlış pozitif yok: benzer başlayan ilgisiz sözcükler dizge çıktısıyla aynı
for w in ("hakem", "hakan", "hala", "kağan", "kitap", "kilim", "klavye", "mahzur", "kral"):
    assert oz.word(w) == raw.word(w), w
assert A(oz, "kağıdı") != A(raw, "kağıdı")

# 4) özenli okumada iddia/klinik dizge'nin okuması; gündelik okumada kullanıcı söylenişi
assert oz.word("iddia") == raw.word("iddia") and oz.word("klinik") == raw.word("klinik")
assert A(gu, "iddia") == "I d d ɑː" and A(gu, "iddiası") == "I d d ɑː s ɨ" and A(gu, "iddialar").startswith("I d d ɑː ł")
assert A(gu, "klinik") == "kʰ ɨ l I n I c" and A(gu, "kliniği").startswith("kʰ ɨ l I n")
assert A(gu, "kağıt") == "cʰ aː t" and A(gu, "kağıdı") == "cʰ aː d ɨ"   # kaat: özenli düzeltme + a ı'yi benzetir
assert A(gu, "hakim") == "x aː c I m"                                    # gündelik = özenli + gündelik satırları

# 5) sözlük dosyası tutarlı: her atom geçerli, satır eşleşme kalıbını dizge çıktısında BULUYOR (dizge sürümü değişince test düşer)
ex = Exceptions("gündelik")
assert ex.rows, "sözlük boş"
for stems, reg, frm, to, _ in ex.rows:
    assert reg in ("özenli", "gündelik") and frm and to and all(a in PHONES for a in frm + to), (stems, frm, to)
with warnings.catch_warnings():
    warnings.simplefilter("error")  # beklenen kalıp uyuşmazlığı uyarısı çıkmamalı
    for w in ("kağıt", "kağıdı", "hakim", "hâkimiyet", "iddia", "iddialar", "klinik", "kliniği"):
        gu2 = Phonemizer(bert_fallback=False, pron_exceptions=True, register="gündelik"); gu2.word(w)
# uyuşmayan (dizge değişmiş / desteklenmeyen çekim) -> uyarı: iddiaya (ay -> ɑːI) gündelik satırıyla eşleşmez
with warnings.catch_warnings(record=True) as W:
    warnings.simplefilter("always")
    Phonemizer(bert_fallback=False, pron_exceptions=True, register="gündelik").word("iddiaya")
assert any(issubclass(w.category, RuntimeWarning) and "iddia" in str(w.message) for w in W), [str(w.message) for w in W]

# 6) engine: bayrak varsayılan KAPALI; açıkken token'lar değişir ve sürüm dizgesi sözlüğü içerir
e0, e1 = Engine(bert_fallback=False), Engine(bert_fallback=False, pron_exceptions=True)
assert e0.frontend("Kağıdı hakime verdi.").tokens != e1.frontend("Kağıdı hakime verdi.").tokens
assert e0.frontend("Merhaba dünya.").tokens == e1.frontend("Merhaba dünya.").tokens
assert "istisna" not in e0.versions()["phonemize"] and "istisna" in e1.versions()["phonemize"]
assert Engine(bert_fallback=False, pron_exceptions=True, register="gündelik").versions()["phonemize"] != e1.versions()["phonemize"]
# 7) atom kapsama denetimi: sözlüğün ürettiği atom eğitimde yoksa/azsa işaretlenir (aː Antalia'da 1 kez: model o atomu öğrenemez)
import json, os, tempfile
from dizgetts.frontend.pronounce import coverage

with tempfile.TemporaryDirectory() as d:
    mf = os.path.join(d, "train_phon.jsonl")
    with open(mf, "w", encoding="utf8") as f:
        f.write(json.dumps(dict(tokens=["cʰ"] * 500 + ["aː"] * 3)) + chr(10))
    low = dict((a, n) for a, n, _ in coverage(min_count=100, manifest=mf))
    assert low["aː"] == 3 and "cʰ" not in low, low          # cʰ (500) yeterli, aː (3) yetersiz (sözlükteki diğer atomlar bu sahte manifestte 0)
    assert coverage(manifest=os.path.join(d, "yok.jsonl")) is None   # manifest yoksa atlanır
print("OK")

# 7) Arapça/Farsça/Batı alıntıları (kullanıcı 2026-09-29, docs/loanword_research.md) + şapka (dizge â/î'yi atıyordu)
L = Phonemizer(bert_fallback=False, pron_exceptions=True, length_rules=True)
for w, want in {"saat": "s aː t", "saatler": "s aː t l ɛ ɣ", "tabii": "tʰ ɑ b iː", "zaten": "z aː t ɛ n", "hal": "x aː l", "halde": "x aː l d ɛ",
                "hâlâ": "x aː l aː", "kalbi": "cʰ a l b I", "rolü": "r œ l Y", "kontrol": "kʰ ɔ n t ɾ œ l", "dükkânı": "d Y c c a n ɨ", "hikaye": "ç I c a I ɛ",
                "kâr": "cʰ a ɣ", "resmî": "r e s m iː", "hayalî": "x ɑ I a l iː", "âdet": "aː d ɛ t", "dergâh": "d e ɾ ɟ a x"}.items():
    assert A(L, w) == want, (w, A(L, w), want)
for w in ("halı", "hala", "halk", "halil", "kar", "kalpak", "kalpağı", "adilik", "mekanik", "resmi", "kral"):  # yanlış pozitif yok
    assert L.word(w) == Phonemizer(bert_fallback=False, length_rules=True).word(w), w

# 8) son hecesi ince alıntı kökleri (resources/loan_roots.tsv; ek uyumu madenciliği + kullanıcı grup kararları 2026-09-29)
for w, want in {"normal": "n ɔ ɾ m a l", "normalde": "n ɔ ɾ m a l d ɛ", "golde": "ɟ œ l d ɛ", "protokol": "pʰ ɨ ɾ ɔ t ɔ c œ l", "mahsulü": "m ɑ x s Y l Y",
                "dikkatli": "d I c c a t l I", "hakikat": "x ɑ c I c a t", "itaat": "I t aː t", "menfaatler": "m e n f aː t l ɛ ɣ", "kristali": "cʰ I ɾ I s t a l I"}.items():
    assert A(L, w) == want, (w, A(L, w), want)
for w in ("golden", "tuvalet", "mahalle", "lokanta", "holding", "metallica", "sualtı", "hayaları", "dahiler", "mekaniği", "program", "plan"):  # dışlamalar / kapsam dışı
    assert L.word(w) == Phonemizer(bert_fallback=False, length_rules=True).word(w), w
