"""dizge-g2p etiket seti / symbols.py doğrulaması (Aşama 1).

1. 87 etiketin hepsi symbols.tokenize ile atomlara ayrılıyor mu?
2. dizge.g2p (öğretmen kural sistemi) çıktısı 48k sözcükte bilinmeyen sembol içeriyor mu? sayımlar.
3. dizge-g2p modeli (öğrenci) öğretmenle ne kadar uyuşuyor? (sözcük düzeyi; ayrıca uyuşmayanlardan örnek)
Çıktı: reports/g2p_probe.json

Çalıştırma (sistem Python: dizge + transformers + torch gerekli):
  python -X utf8 dizgetts/scripts/g2p_probe.py --words D:/playground/turkish_words.txt
"""
import argparse, collections, glob, json, os, sys
from dizgetts.frontend.g2p import G2P  # noqa: E402
from dizgetts.frontend.symbols import PHONES, tokenize, UnknownSymbol  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", default="D:/playground/turkish_words.txt")
    ap.add_argument("--n", type=int, default=0, help="0 = tüm liste")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "reports", "g2p_probe.json"))
    a = ap.parse_args()
    import dizge

    g = G2P()
    labels = g.label_list
    bad_labels = []
    for l in labels:
        if l in ("ʰ", "̥"):  # yalnız değiştirici etiketler: birleştirmede önceki atoma yapışır (kʰ, z̥)
            continue
        try:
            tokenize(l)
        except UnknownSymbol as e:
            bad_labels.append(str(e))
    print(f"[1] etiket: {len(labels)}, ayrışamayan: {len(bad_labels)} {bad_labels}")

    words = [w.strip() for w in open(a.words, encoding="utf8") if w.strip()]
    if a.n:
        words = words[: a.n]
    teacher, variants, unk = {}, 0, collections.Counter()
    atom_cnt = collections.Counter()
    for w in words:
        r = dizge.g2p(w)
        if not isinstance(r, str):
            variants += 1
            r = r[0]
        teacher[w] = r
        for t in tokenize(r, strict=False):
            atom_cnt[t] += 1
        for ch in r:
            if not any(ch in p for p in PHONES) and ch not in " ˈ,.?!;":
                unk[ch] += 1
    unused = sorted(set(PHONES) - set(atom_cnt))
    print(f"[2] sözcük: {len(words)}, çok-varyantlı: {variants}, bilinmeyen karakter: {dict(unk)}")
    print(f"    kullanılmayan atomlar: {unused}")

    pred = g.g2p_batch(words)
    agree = sum(pred[w] == teacher[w] for w in words)
    diffs = [(w, teacher[w], pred[w]) for w in words if pred[w] != teacher[w]]
    pred_unk = collections.Counter()
    for w in words:
        for t in tokenize(pred[w], strict=False):
            pass
        for ch in pred[w]:
            if not any(ch in p for p in PHONES):
                pred_unk[ch] += 1
    print(f"[3] model==öğretmen: {agree}/{len(words)} = {agree/len(words):.4f}; model bilinmeyen karakter: {dict(pred_unk)}")
    for d in diffs[:15]:
        print("   ", d)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(
        {
            "n_labels": len(labels), "bad_labels": bad_labels, "n_words": len(words), "multi_variant_words": variants,
            "teacher_unknown_chars": dict(unk), "unused_atoms": unused,
            "atom_counts": atom_cnt.most_common(), "model_teacher_word_agreement": agree / len(words),
            "model_unknown_chars": dict(pred_unk), "n_diffs": len(diffs), "diff_sample": diffs[:100],
        },
        open(a.out, "w", encoding="utf8"), ensure_ascii=False, indent=1,
    )


if __name__ == "__main__":
    main()
