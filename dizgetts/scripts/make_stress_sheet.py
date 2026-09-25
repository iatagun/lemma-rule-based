"""Vurgu etiketleme tablosu (gold vurgu için): Antalia sözcüklerinden sıklığa göre katmanlı 250 sözcük.

  python -X utf8 -m dizgetts.scripts.make_stress_sheet
Çıktı: reports/stress_annotation_sheet.tsv. Kullanıcı yalnız YANLIŞ olanları düzeltir: `sizin_vurgu` sütununa doğru vurgulu seslemi
BÜYÜK harfle yazın (ör. şİMdi -> ŞİM-di için "ŞİM-di"); model doğruysa boş bırakın. Sonuç tests/stress_gold_random.tsv olur.
"""
import collections, json, os, random, re

import yaml

from dizgetts.frontend.normalize import tr_lower
from dizgetts.frontend.stress import VOWEL_LETTERS, StressRules, syllabify

HERE = os.path.join(os.path.dirname(__file__), "..")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", default="100,100,50", help="sıklık katmanı başına sözcük: ilk 300 / 300-2000 / kalan")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=[], help="bu gold dosyalarındaki sözcükler örneklenmez")
    ap.add_argument("--out", default="stress_annotation_sheet.tsv", help="reports/ altında")
    args = ap.parse_args()
    root = yaml.safe_load(open(os.path.join(HERE, "configs", "data.yaml"), encoding="utf8"))["out_root"]
    freq, example, cap_mid = collections.Counter(), {}, collections.Counter()
    for sp in ("train", "val", "test"):
        for l in open(os.path.join(root, f"{sp}_phon.jsonl"), encoding="utf8"):
            norm = json.loads(l)["text_norm"]
            toks = norm.split()
            for i, t in enumerate(toks):
                if not t.isalpha():
                    continue
                w = tr_lower(t)
                freq[w] += 1
                example.setdefault(w, " ".join(toks[max(0, i - 4): i + 4]))
                if t[:1].isupper() and i > 0 and toks[i - 1] not in (".", "?", "!"):
                    cap_mid[w] += 1
    ranked = [w for w, _ in freq.most_common() if sum(c in VOWEL_LETTERS for c in w) >= 1]
    excl = {l.split("\t")[0] for f in args.exclude for l in open(f, encoding="utf8") if l.strip() and not l.startswith("#")}
    ranked = [w for w in ranked if w not in excl]
    rng = random.Random(args.seed)
    nb = [int(x) for x in args.bins.split(",")]
    bins = [(0, 300, nb[0]), (300, 2000, nb[1]), (2000, len(ranked), nb[2])]
    picked = []
    for a, b, n in bins:
        picked += rng.sample(ranked[a:b], min(n, len(ranked[a:b])))
    R = StressRules()
    rows = ["sözcük\tsıklık\tözel_ad_olası\tmodel_hece\tmodel_kural\tsizin_vurgu\tnot\törnek_cümle"]
    for w in picked:
        k, tag = R.syllable(w.capitalize() if cap_mid[w] else w)
        syl = syllabify(w)
        if k is not None and k < len(syl):
            syl[k] = syl[k].replace("i", "İ").upper()  # Türkçe büyük harf: i -> İ, ı -> I
        rows.append("\t".join([w, str(freq[w]), "evet" if cap_mid[w] else "", "-".join(syl), tag, "", "", example[w]]))
    out = os.path.join(HERE, "reports", args.out)
    open(out, "w", encoding="utf8").write("\n".join(rows) + "\n")
    print(f"{len(picked)} sözcük -> {out}")
    print("\n".join(rows[:12]))


if __name__ == "__main__":
    main()
