"""
BOUN Treebank (Universal Dependencies) ile karşılaştırmalı değerlendirme.

Gold standard: UD_Turkish-BOUN test seti
Format: CoNLL-U (word → lemma eşleşmesi)
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

# Proje kökünü path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from morphology import create_default_analyzer
from morphology.phonology import turkish_lower


# ── CoNLL-U Ayrıştırıcı ──────────────────────────────────────


def parse_conllu(path: str) -> list[dict]:
    """CoNLL-U dosyasından token bilgilerini çıkarır."""
    tokens = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 10:
                continue
            # Çok sözcüklü token satırlarını atla (1-2 format)
            if "-" in cols[0] or "." in cols[0]:
                continue
            tokens.append(
                {
                    "word": cols[1],
                    "lemma": cols[2],
                    "upos": cols[3],
                    "feats": cols[5],
                }
            )
    return tokens


# ── Değerlendirme ────────────────────────────────────────────


def get_predicted_lemma(analyzer, word: str, upos: str = None) -> str:
    """Sistemimizin tahmin ettiği lemmayı döndürür."""
    result = analyzer.analyze(word, upos=upos)
    return result.root if result.root else result.stem


def evaluate(tokens: list[dict], analyzer) -> dict:
    """Gold lemmalarla karşılaştırma yapar."""
    results = {
        "total": 0,
        "correct": 0,
        "wrong": 0,
        "by_pos": {},       # POS bazlı sonuçlar
        "errors": [],       # Hata örnekleri
        "identity_correct": 0,  # word==lemma ve doğru tahmin
        "changed_correct": 0,   # word!=lemma ve doğru tahmin
        "changed_total": 0,     # word!=lemma toplam
    }

    skip_pos = {"PUNCT", "SYM", "X"}

    for tok in tokens:
        upos = tok["upos"]
        if upos in skip_pos:
            continue

        word = tok["word"]
        gold_lemma = turkish_lower(tok["lemma"])
        predicted = get_predicted_lemma(analyzer, word, upos=upos).lower()

        results["total"] += 1
        is_changed = turkish_lower(word) != gold_lemma

        if is_changed:
            results["changed_total"] += 1

        # POS takibi
        if upos not in results["by_pos"]:
            results["by_pos"][upos] = {"total": 0, "correct": 0}
        results["by_pos"][upos]["total"] += 1

        if predicted == gold_lemma:
            results["correct"] += 1
            results["by_pos"][upos]["correct"] += 1
            if is_changed:
                results["changed_correct"] += 1
            else:
                results["identity_correct"] += 1
        else:
            results["wrong"] += 1
            results["errors"].append(
                {
                    "word": word,
                    "gold": gold_lemma,
                    "predicted": predicted,
                    "pos": upos,
                }
            )

    return results


def print_report(results: dict) -> None:
    """Değerlendirme raporunu yazdırır."""
    total = results["total"]
    correct = results["correct"]
    acc = 100 * correct / total if total else 0

    changed_total = results["changed_total"]
    changed_correct = results["changed_correct"]
    changed_acc = (
        100 * changed_correct / changed_total if changed_total else 0
    )

    identity_total = total - changed_total
    identity_correct = results["identity_correct"]
    identity_acc = (
        100 * identity_correct / identity_total if identity_total else 0
    )

    print("=" * 65)
    print("  BOUN Treebank Karşılaştırmalı Değerlendirme")
    print("=" * 65)
    print()
    print(f"  Toplam token (PUNCT hariç) : {total}")
    print(f"  Doğru                      : {correct}")
    print(f"  Yanlış                     : {results['wrong']}")
    print(f"  Genel doğruluk             : {acc:.1f}%")
    print()
    print(f"  ── Birebir (word==lemma) ──")
    print(f"  Toplam: {identity_total}  Doğru: {identity_correct}"
          f"  Oran: {identity_acc:.1f}%")
    print()
    print(f"  ── Değişen (word!=lemma) ──")
    print(f"  Toplam: {changed_total}  Doğru: {changed_correct}"
          f"  Oran: {changed_acc:.1f}%")
    print()

    # POS bazlı sonuçlar
    print("  ── POS Bazlı Doğruluk ──")
    print(f"  {'POS':8s} {'Toplam':>8s} {'Doğru':>8s} {'Oran':>8s}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for pos, data in sorted(
        results["by_pos"].items(),
        key=lambda x: x[1]["total"],
        reverse=True,
    ):
        t = data["total"]
        c = data["correct"]
        r = 100 * c / t if t else 0
        print(f"  {pos:8s} {t:8d} {c:8d} {r:7.1f}%")
    print()

    # En sık hatalar
    error_counts = Counter(
        (e["gold"], e["predicted"], e["pos"]) for e in results["errors"]
    )
    print("  ── En Sık Hata Örnekleri (gold → tahmin) ──")
    for (gold, pred, pos), cnt in error_counts.most_common(25):
        # Hatanın bir örneğini bul
        example = next(
            e["word"]
            for e in results["errors"]
            if e["gold"] == gold and e["predicted"] == pred
        )
        print(f"  {example:20s} → {pred:15s} (gold: {gold:15s}) "
              f"[{pos}] x{cnt}")
    print()

    # Hata kategorileri analizi
    verb_errors = [e for e in results["errors"] if e["pos"] == "VERB"]
    noun_errors = [e for e in results["errors"] if e["pos"] == "NOUN"]
    adj_errors = [e for e in results["errors"] if e["pos"] == "ADJ"]
    propn_errors = [e for e in results["errors"] if e["pos"] == "PROPN"]

    print("  ── Hata Kategorileri ──")
    for label, errs in [
        ("VERB hataları", verb_errors),
        ("NOUN hataları", noun_errors),
        ("ADJ hataları", adj_errors),
        ("PROPN hataları", propn_errors),
    ]:
        if errs:
            print(f"  {label}: {len(errs)}")


# ── Ana Akış ─────────────────────────────────────────────────

if __name__ == "__main__":
    dict_path = PROJECT_ROOT / "turkish_words.txt"
    test_path = Path(__file__).resolve().parent / "test.conllu"

    if not test_path.exists():
        print(f"HATA: {test_path} bulunamadı!")
        sys.exit(1)

    print("Sözlük yükleniyor...")
    analyzer = create_default_analyzer(str(dict_path))

    print("Test seti ayrıştırılıyor...")
    tokens = parse_conllu(str(test_path))

    print("Değerlendirme yapılıyor...\n")
    results = evaluate(tokens, analyzer)
    print_report(results)
