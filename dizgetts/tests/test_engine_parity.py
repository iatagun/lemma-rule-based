"""Engine.frontend(metin).tokens == eğitim manifestindeki `tokens` (TÜM klipler) — eğitim/çıkarım eşitliği.
  D:/dizgetts/venv/Scripts/python.exe -X utf8 -m dizgetts.tests.test_engine_parity
Manifestler (D:/dizgetts/data/processed/antalia/*_phon.jsonl) yoksa atlanır."""
import json, os, sys

from dizgetts.engine import Engine

ROOT = "D:/dizgetts/data/processed/antalia"

# birim: sabit cümle
e = Engine(bert_fallback=False)
u = e.frontend("Merhaba, 3'te buluşalım!")
assert u.norm == "Merhaba , üçte buluşalım !", u.norm
assert u.tokens.count(" ") == 2 and u.words[0].punct == [","] and u.words[-1].punct == ["!"], u.tokens

if not os.path.exists(f"{ROOT}/test_phon.jsonl"):
    print("manifest yok, parity atlandı"); sys.exit(0)
bad, n = [], 0
for sp in ("train", "val", "test"):
    for l in open(f"{ROOT}/{sp}_phon.jsonl", encoding="utf8"):
        r = json.loads(l)
        n += 1
        if e.frontend(r["text"]).tokens != r["tokens"]:
            bad.append(r["id"][-8:])
assert not bad, f"{len(bad)}/{n} klipte token farkı: {bad[:5]}"
print(f"OK: {n}/{n} klipte engine tokens == manifest tokens")
