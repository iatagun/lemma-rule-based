"""HF paketi GİDİŞ-DÖNÜŞ testi (push ÖNCESİ ZORUNLU): dışa aktarılan paket, yerel Tagger/checkpoint ile BİREBİR aynı sonucu vermeli.
  python -X utf8 -m dizgetts.tests.test_hf_g2ptts
Denetimler: (1) yüklemede eksik/fazla ağırlık yok, (2) logit'ler orijinal G2PTTS ile aynı, (3) tag() çıktısı Tagger.tag_norm ile cümle cümle aynı, (4) vurgu doğruluğu
(kör test 90/97, gold 33/35) ve sınır F1'i (0,496) HF yolundan da aynı, (5) sözlükler pakette ve HF yolundan kullanılıyor. Checkpoint/veri yoksa atlanır.
Ağ gerekir (dbmdz ELECTRA iskeleti + DizgeBERT-Dep tokenizer önbellekte olmalı)."""
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

from dizgetts import paths

CK = f"{paths.RUNS}/g2ptts_v3/best.pt"
if not os.path.exists(CK) or not os.path.exists(f"{paths.G2PTTS_DATA}/test.jsonl"):
    print("checkpoint/veri yok, atlandı")
    sys.exit(0)

from transformers import AutoModel, AutoTokenizer

from dizgetts.eval.stress_intrinsic import GOLD
from dizgetts.frontend.dep import MODEL_ID as DEP_ID, MODEL_REV as DEP_REV
from dizgetts.frontend.normalize import normalize, tr_lower
from dizgetts.g2ptts.export_hf import export
from dizgetts.g2ptts.tagger import Tagger
from dizgetts.g2ptts.train import G2PTTS, batchify, evaluate, load

torch.manual_seed(0)
tmp = tempfile.TemporaryDirectory()
out = Path(tmp.name) / "pkg"
export(CK, out)
files = {p.name for p in out.iterdir()}
assert {"model.safetensors", "config.json", "stress.py", "normalize.py", "symbols.py", "phonemize.py", "pronounce.py", "g2p.py", "modeling_dizgebert_g2ptts.py",
        "configuration_dizgebert_g2ptts.py", "README.md", "resources", "tokenizer_config.json"} <= files, files
assert {p.name for p in (out / "resources").iterdir()} == {"stress_roots.tsv", "clitics.tsv", "adj_lemmas.txt", "pronunciation_exceptions.tsv", "loan_roots.tsv"}
# paket kendi başına çalışmalı: dizgetts'e bağımlılık SIZMAMIŞ olmalı (HF kullanıcısında dizgetts kurulu değil)
for f in out.glob("*.py"):
    src = f.read_text(encoding="utf8")
    assert "import dizgetts" not in src and "from dizgetts" not in src, f.name   # docstring anmaları serbest, içe aktarma yasak

# (1) yükleme: eksik/fazla anahtar YOK
hf, info = AutoModel.from_pretrained(str(out), trust_remote_code=True, output_loading_info=True)
hf.eval()
assert not info["missing_keys"] and not info["unexpected_keys"] and not info.get("mismatched_keys"), info
assert abs(hf.config.tau - 0.25) < 1e-6 and hf.config.train_info["ckpt_sha1"]

# (2) logit eşitliği (aynı ağırlık, aynı batch)
ck = torch.load(CK, map_location="cpu", weights_only=False)
orig = G2PTTS()
orig.load_state_dict(ck["state"])
orig.eval()
tok = AutoTokenizer.from_pretrained(DEP_ID, revision=DEP_REV)
te = load("test")
ant = [r for r in te if r["src"] == "antalia"]
rows = ant + [r for r in te if r["src"] == "ud"][:200]
worst = 0.0
for i in range(0, len(rows), 16):
    ids, am, first, _, _ = batchify(rows[i:i + 16], tok, "cpu")
    with torch.no_grad():
        ls, lb = orig(ids, am, first)
        o = hf(ids, am, first)
    worst = max(worst, float((ls - o.logits_stress).abs().max()), float((lb - o.logits_boundary).abs().max()))
assert worst < 1e-5, f"logit farkı {worst}"
print(f"(2) logit eşitliği: {len(rows)} cümle, max fark {worst:.1e}")

# (3) tag() == Tagger.tag_norm, cümle cümle (vurgu, kaynak, sınır, olasılık)
tg = Tagger(CK)
texts = [json.loads(l)["text"] for sp in ("test", "val") for l in open(f"{paths.ANTALIA}/{sp}.jsonl", encoding="utf8")]
texts += [l.strip() for l in open(Path(__file__).parents[1] / "eval" / "extra_sentences_ud.txt", encoding="utf8") if l.strip() and not l.startswith("#")][:120]
KEYS = ("word", "stress_from_end", "stress_src", "p_break", "boundary")
bad = 0
for t in texts:
    a = [{k: w[k] for k in KEYS} for w in tg.tag_norm(normalize(t), t)["words"]]
    b = [{k: w[k] for k in KEYS} for w in hf.tag(t, tokenizer=tok)["words"]]
    bad += a != b
assert bad == 0, f"{bad}/{len(texts)} cümlede HF ≠ Tagger"
print(f"(3) tag() = Tagger: {len(texts)} cümle birebir")
long_text = " ".join(texts[:40])  # parçalama yolu (256 alt-sözcük aşılır)
a = [{k: w[k] for k in KEYS} for w in tg.tag_norm(normalize(long_text), long_text)["words"]]
b = [{k: w[k] for k in KEYS} for w in hf.tag(long_text, tokenizer=tok)["words"]]
assert a == b and len(a) > 300, len(a)

# (4a) vurgu doğruluğu: HF yolundan, yerel Tagger ile aynı ve beklenen sayılar
def acc(gold_file, predict):
    gold = [l.split("\t") for l in Path(gold_file).read_text(encoding="utf8").splitlines() if l.strip() and not l.startswith("#")]
    ok = 0
    for row in gold:
        w, g, cat = row[0], None if row[1] == "-" else int(row[1]), row[2]
        text = w[:1].replace("i", "İ").upper() + w[1:] if cat.startswith("yer adı") else w
        ok += predict(text) == g
    return ok, len(gold)

for name, gold_file, want in (("kör test", GOLD.parent / "stress_gold_test.tsv", (90, 97)), ("gold-35", GOLD, (33, 35))):
    h = acc(gold_file, lambda t: hf.tag(t, tokenizer=tok)["words"][0]["stress_from_end"])
    l = acc(gold_file, lambda t: tg(t)["words"][0]["stress_from_end"])
    assert h == l == want, (name, h, l, want)
    print(f"(4a) {name}: HF {h[0]}/{h[1]} = yerel")

# (4b) sınır F1: HF forward ile, yerel modelle aynı (GPU'da kayıtlı 0,49609 idi; CPU'da aynı kalmalı)
class Adapter:
    def __init__(self, m): self.m = m
    def eval(self): self.m.eval(); return self
    def __call__(self, ids, am, first):
        o = self.m(ids, am, first)
        return o.logits_stress, o.logits_boundary

r_hf, _ = evaluate(Adapter(hf), rows, tok, "cpu", tau=hf.config.tau)
r_lo, _ = evaluate(orig, rows, tok, "cpu", tau=hf.config.tau)
assert abs(r_hf["sınır_F1"] - r_lo["sınır_F1"]) < 1e-9 and r_hf["vurgu_uyum_antalia"] == r_lo["vurgu_uyum_antalia"], (r_hf, r_lo)
assert abs(r_hf["sınır_F1"] - 0.49609375) < 0.01, r_hf["sınır_F1"]   # 84 Antalia test klibi: kayıtlı F1 (82 klip, GPU) ile tutarlı
print(f"(4b) sınır F1 HF {r_hf['sınır_F1']:.4f} = yerel {r_lo['sınır_F1']:.4f} (kayıtlı 0,4961)")

# (5) sözlükler HF yolundan KULLANILIYOR: paketteki sözlüğü değiştir -> sonuç değişmeli
assert hf.rules.roots and hf.rules.clitics and hf.rules.adj
alt = Path(tmp.name) / "pkg2"
import shutil
shutil.copytree(out, alt)
(alt / "resources" / "clitics.tsv").write_text("# boş\nzzz\tdenemesi\n", encoding="utf8")
hf2 = AutoModel.from_pretrained(str(alt), trust_remote_code=True).eval()
assert "da" in hf.rules.clitics and "da" not in hf2.rules.clitics
print("(5) sözlükler pakette ve kullanılıyor")

# (6) SESBİRİM: tag()["phonemes"] == dizgetts Engine(g2ptts, aynı etiketleyici, söyleyiş sözlüğü + uzun ünlü kuralları) token'ları, cümle cümle
from dizgetts.engine import Engine
assert hf.config.phonemes and hf.config.pron_exceptions and hf.config.length_rules and hf.config.register == "özenli"
eng = Engine(g2ptts=True, g2ptts_tagger=hf, pron_exceptions=True, length_rules=True, register="özenli")
bad = [t for t in texts if "".join(eng.frontend(t).tokens) != hf.tag(t, tokenizer=tok)["phonemes"]]
assert not bad, f"{len(bad)}/{len(texts)} cümlede sesbirim farkı: {bad[:3]}"
o = hf.tag("Kâğıdı hâlâ dükkânda bıraktım, saat dokuzda alırım.", tokenizer=tok)
ph = {w["word"]: "".join(w["phones"]) for w in o["words"]}
assert ph["Kâğıdı"] == "cʰaɨdɨ" and ph["hâlâ"] == "xaːlaː" and ph["saat"] == "saːt" and ph["dükkânda"] == "dyccandɑ", ph
assert "cʰaɨdˈɨ" in o["phonemes"] and " xˈaːlaː " in o["phonemes"] and "sˈaːt" in o["phonemes"], o["phonemes"]  # hâlâ: ilk hece vurgusu
# söyleyiş sözlüğü de HF yolundan KULLANILIYOR (res_dir): paketteki sözlüğü boşalt -> saat dizge'nin okumasına döner
(alt / "resources" / "pronunciation_exceptions.tsv").write_text("# boş\n", encoding="utf8")
hf3 = AutoModel.from_pretrained(str(alt), trust_remote_code=True).eval()
assert [w["phones"] for w in hf3.tag("saat", tokenizer=tok)["words"]] == [["s", "ɑ", "ɑ", "t"]]
print(f"(6) sesbirim = Engine: {len(texts)} cümle birebir; alıntı/şapka HF yolundan")
tmp.cleanup()
print("OK")
