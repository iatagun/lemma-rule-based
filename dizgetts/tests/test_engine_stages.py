"""Engine aşamalarının sessiz düşüşleri (2026-09-26 code review regresyonları). Checkpoint/ağ gerekmez (sahte tagger).
  python -X utf8 -m dizgetts.tests.test_engine_stages
"""
import os
import tempfile
import warnings

from dizgetts.engine import Engine, G2PTTSStage, MorphStage, Utterance, Word

# 1) g2ptts sözcük listesi uyuşmazsa: SESSİZ vurgusuz/sınırsız konuşma yerine uyarı + kural tabanlı (M1a) geri dönüş
class _Rules:
    @staticmethod
    def version():
        return "kurallar"


class _BadTagger:
    rules, ckpt = _Rules(), None

    def tag_norm(self, norm):
        return {"words": [dict(word="başka", stress_src="model", boundary="0", stress_from_end=0)]}


st = G2PTTSStage.__new__(G2PTTSStage)
st.t, st.ckpt, st._sha, st._fallback = _BadTagger(), None, None, None
u = Engine(bert_fallback=False).frontend("Merhaba, dünya!")  # kural yolu (referans)
ref_tokens = list(u.tokens)
u2 = Utterance(raw="Merhaba, dünya!", norm=u.norm, words=[Word(text=w.text, phones=list(w.phones), punct=list(w.punct)) for w in u.words])
with warnings.catch_warnings(record=True) as W:
    warnings.simplefilter("always")
    st(u2)
assert u2.meta.get("g2ptts_hizalama_hatasi") is True
assert any(issubclass(w.category, RuntimeWarning) and "g2ptts" in str(w.message) for w in W), [str(w.message) for w in W]
assert all(w.stress_src for w in u2.words) and [w.boundary for w in u2.words] == ["ip", "cümle"], [(w.stress_src, w.boundary) for w in u2.words]
assert any(w.stress is not None for w in u2.words)  # vurgu düşmedi

# 2) Morph hizalama hatası da uyarır
ms = MorphStage(cache={"Merhaba , dünya !": [("X", {})] * 4})
u3 = Utterance(raw="", norm="Merhaba , dünya !", words=[Word(text="Farklı"), Word(text="dünya")])
with warnings.catch_warnings(record=True) as W:
    warnings.simplefilter("always")
    ms(u3)
assert u3.meta.get("morph_hizalama_hatasi") is True and any(issubclass(w.category, RuntimeWarning) for w in W)

# 3) g2ptts sürümü checkpoint İÇERİĞİNE bağlı (yolun değil): aynı yolda içerik değişirse sürüm değişir
with tempfile.TemporaryDirectory() as d:
    f = os.path.join(d, "best.pt")
    ver = []
    for blob in (b"a" * 100, b"b" * 100):
        open(f, "wb").write(blob)
        s = G2PTTSStage.__new__(G2PTTSStage)
        s.t, s.ckpt, s._sha, s._fallback = _BadTagger(), f, None, None
        ver.append(s.version())
    assert ver[0] != ver[1], ver

# 4) fonemsiz sözcük meta'ya yazılır
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    u4 = Engine(bert_fallback=False).frontend("Merhaba ж dünya")
assert u4.meta.get("fonemsiz_sözcük") == ["ж"], u4.meta  # normalize harf sayar, dizge tanımaz -> sözcük konuşmadan düşer, kayıt tutulur
# 5) hazır etiketleyici enjeksiyonu (HF modeli: Space'te yerel checkpoint yok): Engine(g2ptts_tagger=...) -> vurgu + sınır; sürüm ckpt yolu olmadan da çalışır
class _Cfg:
    _name_or_path, rules_version, train_info = "iatagun/DizgeBERT-G2PTTS", "abc12345", {"ckpt_sha1": "deadbeef0000"}


class _HFLike:
    """HF DizgeBertG2ptts arayüzü: tag_norm(norm) + config + rules (ckpt özniteliği YOK)."""
    config, rules = _Cfg(), _Rules()

    def tag_norm(self, norm, text="", tokenizer=None):
        ws = [t for t in norm.split() if t.isalpha()]
        return {"words": [dict(word=w, stress_from_end=0, stress_src="model", boundary="ip" if i == 0 else "0", p_break=0.9) for i, w in enumerate(ws)][:-1]
                + [dict(word=ws[-1], stress_from_end=0, stress_src="model", boundary="cümle", p_break=0.9)]}


eg = Engine(bert_fallback=False, g2ptts=True, g2ptts_tagger=_HFLike(), g2ptts_breaks=True, pron_exceptions=True, length_rules=True)
ug = eg.frontend("Kağıdı verdi ama gitti.")
assert [w.boundary for w in ug.words] == ["ip", "0", "0", "cümle"] and all(w.stress is not None for w in ug.words), [(w.text, w.stress, w.boundary) for w in ug.words]
assert "ˈ" in ug.tokens and "hf:iatagun/DizgeBERT-G2PTTS@deadbeef0000" in eg.versions()["g2ptts"] and "abc12345" in eg.versions()["g2ptts"], eg.versions()["g2ptts"]
print("OK")
