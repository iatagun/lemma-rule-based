"""
Kural-tabanlı morfolojik çözümleyicinin doğru bildiği token'ları
CoNLL-U'dan filtreleyerek dependency parser eğitimi için augmented veri oluşturur.

Çıktı: prepare_morph_data.py ile aynı JSON formatında
  - augment_train.json (doğruluk paylaştırma ile genişletilmiş eğitim)
  - augment_dev.json
  - augment_test.json
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from morphology import create_default_analyzer


def parse_conllu(path: str) -> list[list[dict]]:
    """CoNLL-U dosyasını cümle cümle ayrıştırır."""
    sentences = []
    current = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if line == "" and current:
                    sentences.append(current)
                    current = []
                continue
            cols = line.split("\t")
            if len(cols) < 10:
                continue
            if "-" in cols[0] or "." in cols[0]:
                continue
            current.append({
                "id": int(cols[0]),
                "form": cols[1],
                "lemma": cols[2],
                "upos": cols[3],
                "head": int(cols[6]),
                "deprel": cols[7],
            })
        if current:
            sentences.append(current)
    return sentences


def get_predicted_lemma(analyzer, word: str, upos: str) -> str:
    """Sistemin tahmin ettiği lemmayı döndürür."""
    result = analyzer.analyze(word, upos=upos)
    if result is None:
        return word
    return result.root if result.root else result.stem


def filter_sentence(sent_tokens: list[dict], analyzer, min_accuracy: float = 0.8) -> list[dict] | None:
    """Cümledeki token'ları doğruluk paylaştırma ile filtreler."""
    correct = 0
    total = 0
    results = []

    for tok in sent_tokens:
        if tok["upos"] == "PUNCT":
            continue
        total += 1
        pred = get_predicted_lemma(analyzer, tok["form"], tok["upos"])
        if pred == tok["lemma"]:
            correct += 1

    accuracy = correct / max(total, 1)
    if accuracy < min_accuracy:
        return None

    words = [tok["form"] for tok in sent_tokens]
    upos_list = [tok["upos"] for tok in sent_tokens]
    heads = [str(tok["head"]) for tok in sent_tokens]
    deprels = [tok["deprel"] for tok in sent_tokens]

    return {
        "words": words,
        "upos": upos_list,
        "heads": heads,
        "deprels": deprels,
        "accuracy": round(accuracy, 3),
    }


def main():
    dictionary_path = PROJECT_ROOT / "turkish_words.txt"
    analyzer = create_default_analyzer(dictionary_path=str(dictionary_path))

    kenet_dir = PROJECT_ROOT / "ngram_pos" / "UD_Turkish-Kenet-master"

    splits = {
        "train": kenet_dir / "tr_kenet-ud-train.conllu",
        "dev": kenet_dir / "tr_kenet-ud-dev.conllu",
        "test": kenet_dir / "tr_kenet-ud-test.conllu",
    }

    output_dir = PROJECT_ROOT / "dep_data" / "bert"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}

    for split_name, conllu_path in splits.items():
        if not conllu_path.exists():
            print(f"[{split_name}] Dosya bulunamadı: {conllu_path}")
            continue

        sentences = parse_conllu(str(conllu_path))
        print(f"[{split_name}] {len(sentences)} cümle okundu.")

        filtered = [filter_sentence(s, analyzer) for s in sentences]
        filtered = [s for s in filtered if s is not None]

        stats = defaultdict(int)
        for s in filtered:
            acc_key = f"{s['accuracy']:.2f}"
            stats[acc_key] += 1

        output_path = output_dir / f"augment_{split_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)

        print(f"[{split_name}] {len(filtered)}/{len(sentences)} cümle filtreyi geçti.")
        print(f"  Doğruluk dağılımı: {dict(sorted(stats.items()))}")
        print(f"  Kaydedildi: {output_path}")

        all_stats[split_name] = {"toplam": len(sentences), "filtrelenmiş": len(filtered)}

    print("\n=== Özet ===")
    for split_name, stats in all_stats.items():
        print(f"{split_name}: {stats['filtrelenmiş']}/{stats['toplam']} cümle korundu.")

    print("\nİlk filtrelenmiş cümle örneği:")
    for split_name in ("train", "dev", "test"):
        aug_path = output_dir / f"augment_{split_name}.json"
        if aug_path.exists():
            with open(aug_path, encoding="utf-8") as f:
                data = json.load(f)
            if data:
                ex = data[0]
                print(f"[{split_name}] {' '.join(ex['words'][:6])}... (acc={ex['accuracy']})")
                print(f"  POS: {ex['upos'][:6]}")
                print(f"  Head: {ex['heads'][:6]}")
                print(f"  Deprel: {ex['deprels'][:6]}")
                break


if __name__ == "__main__":
    main()
