"""Vurgu iç değerlendirmesi (TTS eğitmeden): gold sözcük listesinde vurgulu seslem doğruluğu, sistem sistem.
  python -X utf8 -m dizgetts.eval.stress_intrinsic [--gold dizgetts/tests/stress_gold.tsv] [--no-morph] [--g2ptts]

Sistemler: son_hece (varsayılan kural), m1a (kök sözlüğü + clitic), m1b (m1a + morfolojik katmanlar; DizgeBERT-Morph TEK sözcük üzerinde,
bağlamsız -> cümle içindekinden zayıf olabilir), espeak (tr, with_stress). Karşılaştırma vurgulu seslemin SONDAN sırasıyladır (0 = son).
DİKKAT: tests/stress_gold.tsv'deki kök sözcükleri resources/stress_roots.tsv'de de var (kurallar bu örneklerden yazıldı) -> m1a/m1b skoru orada
DÖNGÜSEL, yalnız regresyon kontrolüdür. Bağımsız ölçüm için kullanıcının etiketlediği rastgele sözcükler (stress_annotation_sheet -> --gold) gerekir.
Gold biçimi: sözcük<TAB>sondan_sıra ("-" = vurgusuz)<TAB>kategori[<TAB>not]; kategori "yer adı..." ise sözcük büyük harfle başlatılır (Ordu/ordu ayrımı).
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from dizgetts.frontend.normalize import tr_lower
from dizgetts.frontend.stress import TIERS, StressRules, _n_vowels

GOLD = Path(__file__).resolve().parent.parent / "tests" / "stress_gold.tsv"
_IPA_V = re.compile(r"[aeiouyɯœøɛɪʊʏəɑɔɐæ]+ː?")


def espeak_from_end(word: str) -> int | None:
    from dizgetts.frontend.espeak import phonemize

    ipa = phonemize(word)
    if "ˈ" not in ipa:
        return None
    return len(_IPA_V.findall(ipa.split("ˈ", 1)[1])) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(GOLD))
    ap.add_argument("--no-morph", action="store_true")
    ap.add_argument("--g2ptts", action="store_true", help="dizge-g2p-tts karma vurgusu (sözlük öncelikli + model; g2ptts/tagger.py)")
    a = ap.parse_args()

    gold = [l.split("\t") for l in Path(a.gold).read_text(encoding="utf8").splitlines() if l.strip() and not l.startswith("#")]
    rules = StressRules()
    an = None
    if not a.no_morph:
        from dizgetts.frontend.morph import MorphAnalyzer
        an = MorphAnalyzer("cpu")

    tg = None
    if a.g2ptts:
        from dizgetts.g2ptts.tagger import Tagger
        tg = Tagger()
    systems = ["son_hece", "m1a"] + ([] if an is None else ["m1b"]) + ([] if tg is None else ["g2ptts"]) + ["espeak"]
    hits = defaultdict(int)
    print("sözcük\tgold\t" + "\t".join(systems) + "\tkategori")
    for row in gold:
        w, g, cat = row[0], None if row[1] == "-" else int(row[1]), row[2]  # "-" = vurgusuz (kör etiket arayüzü)
        text = w[:1].replace("i", "İ").upper() + w[1:] if cat.startswith("yer adı") else w  # "i".upper() == "I" (Türkçe değil)
        n = _n_vowels(tr_lower(w))
        from_end = lambda k: None if k is None else n - 1 - k
        pred = {"son_hece": 0, "m1a": from_end(rules.syllable(text)[0])}
        if an is not None:
            upos, feats = an.analyze([text])[0]
            pred["m1b"] = from_end(rules.syllable(text, upos, feats, TIERS)[0])
        if tg is not None:
            pred["g2ptts"] = tg(text)["words"][0]["stress_from_end"]
        pred["espeak"] = espeak_from_end(text)
        for s in systems:
            ok = pred[s] == g
            hits[s] += ok
        print(f"{text}\t{g}\t" + "\t".join(f"{pred[s]}{'' if pred[s] == g else '✗'}" for s in systems) + f"\t{cat}")

    print(f"\nDOĞRULUK (n={len(gold)}):")
    for s in systems:
        print(f"  {s:9s} {hits[s]:3d}/{len(gold)} = {hits[s] / len(gold):.1%}")


if __name__ == "__main__":
    main()
