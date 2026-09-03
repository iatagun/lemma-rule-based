"""
Kural-tabanlı çözümleyicinin hata yaptığı desenler için sentetik cümleler.
Her cümle gold POS/head/deprel içerir.

Hata kaynağı: benchmark/evaluate.py çıktısı (x3+ tekrarlayan hatalar)
Çıktı: dep_data/bert/synthetic.json
"""

import json
from pathlib import Path


def synthetize():
    sentences = []
    sid = 0

    def add(words, upos, heads, deprels, target_word=None):
        nonlocal sid
        sid += 1
        entry = {
            "sent_id": str(sid),
            "words": words,
            "upos": upos,
            "heads": [str(h) for h in heads],
            "deprels": deprels,
        }
        if target_word:
            entry["target"] = target_word
        sentences.append(entry)

    # ═══════════════════════════════════════════════════════════════
    # 1. Edilgen çatı (gold: al, tahmin: alın) x5
    # ═══════════════════════════════════════════════════════════════
    add(
        words=["Bu", "kitap", "çok", "alınır"],
        upos=["DET", "NOUN", "ADV", "VERB"],
        heads=[4, 4, 4, 0],
        deprels=["det", "nsubj", "advmod", "root"],
        target_word="alınır",
    )
    add(
        words=["Toplantıda", "karar", "alındı"],
        upos=["NOUN", "NOUN", "VERB"],
        heads=[3, 3, 0],
        deprels=["obl", "obj", "root"],
        target_word="alındı",
    )
    add(
        words=["Her", "gün", "süt", "içilir"],
        upos=["DET", "NOUN", "NOUN", "VERB"],
        heads=[4, 4, 4, 0],
        deprels=["det", "obl:tmod", "obj", "root"],
        target_word="içilir",
    )
    add(
        words=["Burada", "çay", "içilmez"],
        upos=["ADV", "NOUN", "VERB"],
        heads=[3, 3, 0],
        deprels=["advmod", "obj", "root"],
        target_word="içilmez",
    )
    add(
        words=["Yemekten", "sonra", "uyunur"],
        upos=["NOUN", "ADP", "VERB"],
        heads=[3, 3, 0],
        deprels=["obl", "case", "root"],
        target_word="uyunur",
    )

    # ═══════════════════════════════════════════════════════════════
    # 2. Ettirgen çatı (gold: çıkar, tahmin: çık) x4
    # ═══════════════════════════════════════════════════════════════
    add(
        words=["Ali", "odayı", "çıkarmış"],
        upos=["PROPN", "NOUN", "VERB"],
        heads=[3, 3, 0],
        deprels=["nsubj", "obj", "root"],
        target_word="çıkarmış",
    )
    add(
        words=["O", "sınavı", "geçirmiş"],
        upos=["PRON", "NOUN", "VERB"],
        heads=[3, 3, 0],
        deprels=["nsubj", "obj", "root"],
        target_word="geçirmiş",
    )
    add(
        words=["Annem", "elbiseyi", "giydirdi"],
        upos=["NOUN", "NOUN", "VERB"],
        heads=[3, 3, 0],
        deprels=["nsubj", "obj", "root"],
        target_word="giydirdi",
    )
    add(
        words=["Çocuklar", "koşturuyor"],
        upos=["NOUN", "VERB"],
        heads=[2, 0],
        deprels=["nsubj", "root"],
        target_word="koşturuyor",
    )

    # ═══════════════════════════════════════════════════════════════
    # 3. Çoklu ettirgen (gold: çık, tahmin: çıkar) x4
    # ═══════════════════════════════════════════════════════════════
    add(
        words=["Bu", "işi", "ona", "çıkartmaya", "çalıştı"],
        upos=["DET", "NOUN", "PRON", "VERB", "VERB"],
        heads=[5, 5, 5, 5, 0],
        deprels=["det", "obj", "obl", "xcomp", "root"],
        target_word="çıkartmaya",
    )
    add(
        words=["Bizi", "geçirttiler"],
        upos=["PRON", "VERB"],
        heads=[2, 0],
        deprels=["obj", "root"],
        target_word="geçirttiler",
    )
    add(
        words=["Aşçı", "yemeği", "pişirtti"],
        upos=["NOUN", "NOUN", "VERB"],
        heads=[3, 3, 0],
        deprels=["nsubj", "obj", "root"],
        target_word="pişirtti",
    )

    # ═══════════════════════════════════════════════════════════════
    # 4. İyelikli isim uzun gövde (gold: ev, tahmin: evin) x4
    # ═══════════════════════════════════════════════════════════════
    add(
        words=["Ali", "evinden", "çıktı"],
        upos=["PROPN", "NOUN", "VERB"],
        heads=[3, 3, 0],
        deprels=["nsubj", "obl", "root"],
        target_word="evinden",
    )
    add(
        words=["Şimdi", "elimde", "para", "yok"],
        upos=["ADV", "NOUN", "NOUN", "ADJ"],
        heads=[4, 4, 4, 0],
        deprels=["advmod", "obl", "nsubj", "root"],
        target_word="elimde",
    )
    add(
        words=["Dünyada", "barış", "olsun"],
        upos=["NOUN", "NOUN", "VERB"],
        heads=[3, 3, 0],
        deprels=["obl", "nsubj", "root"],
        target_word="dünyada",
    )
    add(
        words=["Okuldan", "evine", "gitti"],
        upos=["NOUN", "NOUN", "VERB"],
        heads=[3, 3, 0],
        deprels=["obl", "obl", "root"],
        target_word="evine",
    )
    add(
        words=["Köpeğimizi", "bahçede", "gezdirdik"],
        upos=["NOUN", "NOUN", "VERB"],
        heads=[3, 3, 0],
        deprels=["obj", "obl", "root"],
        target_word="Köpeğimizi",
    )

    # ═══════════════════════════════════════════════════════════════
    # 5. Zaman zarflığı (gold: gün, tahmin: günü) x4
    # ═══════════════════════════════════════════════════════════════
    add(
        words=["Bayram", "gününde", "herkes", "eve", "gitti"],
        upos=["PROPN", "NOUN", "PRON", "NOUN", "VERB"],
        heads=[5, 5, 5, 5, 0],
        deprels=["nmod", "obl:tmod", "nsubj", "obl", "root"],
        target_word="gününde",
    )
    add(
        words=["Yılın", "son", "gününde", "kar", "yağdı"],
        upos=["NOUN", "ADJ", "NOUN", "NOUN", "VERB"],
        heads=[5, 3, 5, 5, 0],
        deprels=["nmod:poss", "amod", "obl:tmod", "nsubj", "root"],
        target_word="gününde",
    )
    add(
        words=["O", "gününde", "hastaymış"],
        upos=["PRON", "NOUN", "ADJ"],
        heads=[3, 3, 0],
        deprels=["nsubj", "obl:tmod", "root"],
        target_word="gününde",
    )
    add(
        words=["Doğum", "gününde", "herkes", "gelmiş"],
        upos=["NOUN", "NOUN", "PRON", "VERB"],
        heads=[4, 4, 4, 0],
        deprels=["nmod", "obl:tmod", "nsubj", "root"],
        target_word="gününde",
    )

    # ═══════════════════════════════════════════════════════════════
    # 6. Zaman belirteci (gold: zaman, tahmin: zamanı) x3
    # ═══════════════════════════════════════════════════════════════
    add(
        words=["Zamanı", "gelince", "gideriz"],
        upos=["NOUN", "VERB", "VERB"],
        heads=[2, 0, 2],
        deprels=["obl", "advcl", "root"],
        target_word="zamanı",
    )
    add(
        words=["Yemek", "zamanı", "sofraya", "otur"],
        upos=["NOUN", "NOUN", "NOUN", "VERB"],
        heads=[4, 4, 4, 0],
        deprels=["nmod", "obl:tmod", "obl", "root"],
        target_word="zamanı",
    )
    add(
        words=["Uyku", "zamanı", "gelmiş"],
        upos=["NOUN", "NOUN", "VERB"],
        heads=[3, 3, 0],
        deprels=["nmod", "obl:tmod", "root"],
        target_word="zamanı",
    )

    # ═══════════════════════════════════════════════════════════════
    # 7. Türetim zinciri (gerçekleştirebiliyor → gerçek) x5
    # ═══════════════════════════════════════════════════════════════
    add(
        words=["O", "bunu", "gerçekleştirebiliyor"],
        upos=["PRON", "PRON", "VERB"],
        heads=[3, 3, 0],
        deprels=["nsubj", "obj", "root"],
        target_word="gerçekleştirebiliyor",
    )
    add(
        words=["Firma", "ürünü", "değerlendiriliyor"],
        upos=["NOUN", "NOUN", "VERB"],
        heads=[3, 3, 0],
        deprels=["nsubj", "obj", "root"],
        target_word="değerlendiriliyor",
    )
    add(
        words=["Göçmenler", "şehre", "yerleşmişler"],
        upos=["NOUN", "NOUN", "VERB"],
        heads=[3, 3, 0],
        deprels=["nsubj", "obl", "root"],
        target_word="yerleşmişler",
    )
    add(
        words=["Öğrenciler", "sınıfa", "yerleştirildi"],
        upos=["NOUN", "NOUN", "VERB"],
        heads=[3, 3, 0],
        deprels=["nsubj", "obl", "root"],
        target_word="yerleştirildi",
    )

    # ═══════════════════════════════════════════════════════════════
    # 8. Sözcüksel (işte→PART, askeri→ADJ, öncelikle→ADV) x3-4
    # ═══════════════════════════════════════════════════════════════
    add(
        words=["İşte", "burada", "oturuyorum"],
        upos=["PART", "ADV", "VERB"],
        heads=[3, 3, 0],
        deprels=["discourse", "advmod", "root"],
        target_word="işte",
    )
    add(
        words=["Askeri", "birlik", "harekete", "geçti"],
        upos=["ADJ", "NOUN", "NOUN", "VERB"],
        heads=[4, 4, 4, 0],
        deprels=["amod", "nsubj", "obl", "root"],
        target_word="askeri",
    )
    add(
        words=["Öncelikle", "eve", "gitmeliyim"],
        upos=["ADV", "NOUN", "VERB"],
        heads=[3, 3, 0],
        deprels=["advmod", "obl", "root"],
        target_word="Öncelikle",
    )
    add(
        words=["Bu", "konuda", "farklı", "düşünüyorum"],
        upos=["DET", "NOUN", "ADJ", "VERB"],
        heads=[4, 4, 4, 0],
        deprels=["det", "obl", "xcomp", "root"],
        target_word="farklı",
    )

    # ═══════════════════════════════════════════════════════════════
    # 9. İsim-fiil / çekimli fiil (gold: yap, tahmin: yapacağ) x3
    # ═══════════════════════════════════════════════════════════════
    add(
        words=["Ali", "ne", "yapacağı", "belli", "değil"],
        upos=["PROPN", "PRON", "NOUN", "ADJ", "VERB"],
        heads=[3, 3, 5, 5, 0],
        deprels=["nmod", "nmod", "nsubj", "root", "cop"],
        target_word="yapacağı",
    )
    add(
        words=["Yarın", "işe", "gideceğim"],
        upos=["ADV", "NOUN", "VERB"],
        heads=[3, 3, 0],
        deprels=["advmod", "obl", "root"],
        target_word="gideceğim",
    )

    # ═══════════════════════════════════════════════════════════════
    # 10. İyelik 3T / copula (gold: din(ı), tahmin: din) x3
    # ═══════════════════════════════════════════════════════════════
    add(
        words=["O", "dinini", "özgürce", "yaşıyor"],
        upos=["PRON", "NOUN", "ADV", "VERB"],
        heads=[4, 4, 4, 0],
        deprels=["nsubj", "obj", "advmod", "root"],
        target_word="dinini",
    )
    add(
        words=["Halkın", "dinine", "saygı", "duyarız"],
        upos=["NOUN", "NOUN", "NOUN", "VERB"],
        heads=[4, 4, 4, 0],
        deprels=["nmod:poss", "obj", "obl", "root"],
        target_word="dinine",
    )

    return sentences


def main():
    output_dir = Path(__file__).resolve().parent / "dep_data" / "bert"
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences = synthetize()

    output_path = output_dir / "synthetic.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)

    print(f"Sentetik veri: {len(sentences)} cümle")
    print(f"Kaydedildi: {output_path}")

    print("\nHata deseni dağılımı:")
    targets = {}
    for s in sentences:
        t = s.get("target", "none")
        targets[t] = targets.get(t, 0) + 1
    for t, c in sorted(targets.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")

    print("\nÖrnekler:")
    for s in sentences[:3]:
        print(f"  [{s.get('target','?')}] {' '.join(s['words'])}")
        print(f"    POS: {s['upos']}")
        print(f"    Head: {s['heads']}")
        print(f"    Deprel: {s['deprels']}")


if __name__ == "__main__":
    main()
