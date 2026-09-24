"""M1b (morfolojik vurgu katmanları) kuralları korpusta ne yapıyor? M1a'ya göre neyi değiştiriyor, espeak ile ne kadar örtüşüyor.

  python -X utf8 -m dizgetts.scripts.stress_m1b_report        (önce: python -m dizgetts.scripts.morph_corpus)
ÖNEMLİ: espeak doğruluk ölçütü DEĞİL (kullanıcı kurallarında 9/28); yalnız "iki bağımsız sistem nerede ayrışıyor" bilgisidir. Gerçek ölçüt kullanıcı etiketi.
Çıktı: reports/stress_m1b_report.json
"""
import collections, json, os

import yaml

from dizgetts.frontend import espeak
from dizgetts.frontend.stress import TIERS, VOWEL_LETTERS, StressRules
from dizgetts.tools.espeak_stress_transfer import espeak_stress_index

HERE = os.path.join(os.path.dirname(__file__), "..")


def main():
    root = yaml.safe_load(open(os.path.join(HERE, "configs", "data.yaml"), encoding="utf8"))["out_root"]
    R = StressRules()
    tot = collections.Counter()
    changed = collections.defaultdict(collections.Counter)  # etiket -> {(sözcük, upos): n}
    agree = collections.defaultdict(lambda: [0, 0, 0])      # etiket -> [n, M1a==espeak, M1b==espeak]
    esp_cache = {}
    for l in open(os.path.join(root, "morph_feats.jsonl"), encoding="utf8"):
        r = json.loads(l)
        for t, up, f in zip(r["tokens"], r["upos"], r["feats"]):
            if not t.isalpha():
                continue
            k0, tag0 = R.syllable(t)
            k1, tag1 = R.syllable(t, up, f, TIERS)
            tot[tag1] += 1
            if (k0, tag0) == (k1, tag1):
                continue
            changed[tag1][(t.lower(), up)] += 1
            lw = t.lower()
            if lw not in esp_cache:
                e = espeak.phonemize(lw).split()[0]
                s, n = espeak_stress_index(e)
                esp_cache[lw] = (s, n)
            s, n = esp_cache[lw]
            nl = sum(c in VOWEL_LETTERS for c in lw)
            if s is not None and n == nl:  # ünlü sayıları eşitse karşılaştır
                a = agree[tag1]
                a[0] += 1; a[1] += (s == k0); a[2] += (s == k1)
    rep = dict(etiket_sayıları=dict(tot),
               değişen={t: dict(sözcük_türü=len(c), toplam=sum(c.values()), en_sık=[(f"{w}/{u}", n) for (w, u), n in c.most_common(25)]) for t, c in changed.items()},
               espeak_örtüşme={t: dict(n=a[0], M1a=round(a[1] / a[0], 3), M1b=round(a[2] / a[0], 3)) for t, a in agree.items() if a[0]})
    json.dump(rep, open(os.path.join(HERE, "reports", "stress_m1b_report.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1)[:7000])


if __name__ == "__main__":
    main()
