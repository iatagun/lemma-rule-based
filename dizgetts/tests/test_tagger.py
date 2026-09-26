"""g2ptts Tagger: uzun girdi kesilmemeli (2026-09-26 code review regresyonu). Tokenizer için HF önbelleği/ağ gerekir (yoksa atlanır);
ikinci bölüm checkpoint ister (yoksa atlanır).
  python -X utf8 -m dizgetts.tests.test_tagger
Hata: max_length=256 aşılınca kesilen sözcüklerin `first` indeksi 0 ([CLS]) kalıyor, son ~50 sözcüğün tamamı 'ip' sınırı alıyordu."""
import os
import sys
import warnings

from dizgetts import paths
from dizgetts.frontend.normalize import normalize

SENT = "Bugün hava çok güzel olduğu için arkadaşlarımla birlikte parka gidip uzun uzun yürüyüş yaptık, sonra da eve döndük. "

from transformers import AutoTokenizer

from dizgetts.frontend.dep import MODEL_ID, MODEL_REV
from dizgetts.g2ptts.tagger import MAX_SUBWORDS, Tagger

try:
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
except OSError as e:  # ağ/önbellek yok (yalnız bu atlanır; içe aktarma hataları testi düşürmeli)
    print("tokenizer yok, atlandı:", type(e).__name__)
    sys.exit(0)

tg = Tagger.__new__(Tagger)  # yalnız parçalama mantığı: checkpoint yüklenmez
tg.tok = tok
for n in (1, 3, 14, 40):
    toks = normalize(SENT * n).split()
    ch = tg._chunks(toks)
    assert ch[0][0] == 0 and ch[-1][1] == len(toks) and all(a[1] == b[0] for a, b in zip(ch, ch[1:])), ch  # kesintisiz bölüntü
    for a, b in ch:
        enc = tok(toks[a:b], is_split_into_words=True, truncation=False)
        assert len(enc["input_ids"]) <= MAX_SUBWORDS + 2, (n, a, b, len(enc["input_ids"]))  # [CLS] [SEP] dahil 256 içinde
        assert b == len(toks) or toks[b - 1] in ".?!", (n, toks[b - 3:b])  # cümle sınırından kesilir
assert len(tg._chunks(normalize(SENT).split())) == 1  # kısa metin tek parça
assert tg._chunks([]) == []
# tek başına sığmayan dev bir cümle: noktalama/sözcük sınırında bölünür, yine de hiçbir sözcük düşmez
long_sent = (SENT.replace(".", "") * 20).split()
ch = tg._chunks(long_sent)
assert ch[0][0] == 0 and ch[-1][1] == len(long_sent) and len(ch) > 1

ck = f"{paths.RUNS}/g2ptts_v3/best.pt"
if not os.path.exists(ck):
    print("OK (yalnız parçalama; checkpoint yok)")
    sys.exit(0)
tg = Tagger(ck)
with warnings.catch_warnings():
    warnings.simplefilter("error")  # kesilme artık uyarısız/sessiz olmamalı
    w = tg(SENT * 14)["words"]
head, tail = w[:50], w[-50:]
frac = lambda ws: sum(x["boundary"] in ("ip", "IP") for x in ws) / len(ws)
assert abs(frac(tail) - frac(head)) < 0.25 and frac(tail) < 0.6, (frac(head), frac(tail))  # hata: son 50 sözcük %100 'ip'
print("OK")
