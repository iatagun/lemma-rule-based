#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bağımlılık çözümleme (dependency parsing) benchmark.

BOUN Treebank CoNLL-U dosyasını gold standard olarak kullanır.
Her cümle için:
  1. Morfolojik çözümleme (SentenceAnalyzer)
  2. Bağımlılık ayrıştırma (DependencyParser)
  3. Gold HEAD/DEPREL ile karşılaştırma

Metrikler:
  - UAS (Unlabeled Attachment Score): Head doğruluğu
  - LAS (Labeled Attachment Score): Head + deprel doğruluğu
  - Deprel doğruluğu (sadece etiket)
  - UPOS doğruluğu

Kullanım:
    python benchmark/eval_dep.py                         # varsayılan test.conllu
    python benchmark/eval_dep.py --file benchmark/dev.conllu
    python benchmark/eval_dep.py --max-sentences 100     # ilk 100 cümle
    python benchmark/eval_dep.py --errors 20             # ilk 20 hata
    python benchmark/eval_dep.py --deprel-matrix          # deprel karışıklık tablosu
"""

import sys
import argparse
import time
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

# ── Proje kök dizini ────────────────────────────────────────────
_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from morphology import create_default_analyzer
from morphology.sentence import SentenceAnalyzer
from morphology.dependency import DependencyParser


# ═══════════════════════════════════════════════════════════════════
#  CoNLL-U Parser
# ═══════════════════════════════════════════════════════════════════

def parse_conllu(path: Path):
    """CoNLL-U dosyasını cümle cümle okur.

    Yields:
        (sent_id, text, tokens) where tokens is list of dicts with:
          id, form, lemma, upos, head, deprel
    """
    sent_id = ""
    text = ""
    tokens = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# sent_id"):
                sent_id = line.split("=", 1)[1].strip()
            elif line.startswith("# text"):
                text = line.split("=", 1)[1].strip()
            elif line == "":
                if tokens:
                    yield sent_id, text, tokens
                    tokens = []
            elif line.startswith("#"):
                continue
            else:
                cols = line.split("\t")
                if len(cols) < 8:
                    continue
                # Multi-word token (1-2 gibi) veya empty node (1.1) atla
                if "-" in cols[0] or "." in cols[0]:
                    continue
                tokens.append({
                    "id": int(cols[0]),
                    "form": cols[1],
                    "lemma": cols[2],
                    "upos": cols[3],
                    "head": int(cols[6]),
                    "deprel": cols[7].split(":")[0] if ":" not in cols[7]
                             else cols[7],  # obl:tmod gibi alt-tipleri koru
                })

    if tokens:
        yield sent_id, text, tokens


# ═══════════════════════════════════════════════════════════════════
#  Değerlendirme
# ═══════════════════════════════════════════════════════════════════

def _strip_punct_and_reindex(gold_tokens):
    """PUNCT token'ları kaldır ve ID'leri yeniden numarala.

    BOUN Treebank noktalama işaretlerini token olarak içerir ama
    bizim sistem içermez. Adil karşılaştırma için PUNCT'ları çıkarıp
    head referanslarını yeniden eşleriz.

    Returns:
        list[dict]: PUNCT-hariç, yeniden indekslenmiş gold token'lar.
    """
    # Orijinal ID → yeni ID eşleme
    old_to_new = {0: 0}  # root→root
    new_tokens = []
    for g in gold_tokens:
        if g["upos"] == "PUNCT":
            continue
        new_id = len(new_tokens) + 1
        old_to_new[g["id"]] = new_id
        new_tokens.append({**g, "id": new_id})

    # Head referanslarını yeniden eşle
    for t in new_tokens:
        old_head = t["head"]
        t["head"] = old_to_new.get(old_head, 0)
        # Head PUNCT'a işaret ediyorsa → atlanmış, 0 (root) yap
        # Ancak bu bilgi kaybına yol açar, loglayalım

    return new_tokens


def evaluate(
    conllu_path: Path,
    sa: SentenceAnalyzer,
    dp: DependencyParser,
    *,
    max_sentences: int = 0,
    collect_errors: int = 0,
    deprel_matrix: bool = False,
):
    """Gold CoNLL-U ile dependency parser çıktısını karşılaştırır.

    Strateji: PUNCT token'ları gold'dan çıkarılır ve ID'ler yeniden
    numaralanır. Böylece bizim PUNCT-üretmeyen sistem ile adil
    pozisyon bazlı hizalama yapılır.
    """

    total_tokens = 0
    uas_correct = 0
    las_correct = 0
    deprel_correct = 0
    upos_correct = 0

    # Deprel bazlı istatistik
    deprel_counts = Counter()       # gold deprel dağılımı
    deprel_hits = Counter()         # gold deprel'e göre LAS hit
    deprel_confusion = defaultdict(Counter)  # gold→pred confusion

    # UPOS bazlı
    upos_counts = Counter()
    upos_hits = Counter()

    errors = []
    sent_count = 0
    sent_perfect_las = 0
    token_mismatch_sents = 0

    for sent_id, text, gold_tokens_raw in parse_conllu(conllu_path):
        if max_sentences and sent_count >= max_sentences:
            break
        sent_count += 1

        # PUNCT çıkar ve yeniden indeksle
        gold_tokens = _strip_punct_and_reindex(gold_tokens_raw)

        # Morfolojik çözümleme → Bağımlılık ayrıştırma
        try:
            st_tokens = sa.analyze(text)
            pred_tokens = dp.parse(st_tokens)
        except Exception:
            total_tokens += len(gold_tokens)
            continue

        # Token sayısı uyuşmazlığı kontrolü
        if len(pred_tokens) != len(gold_tokens):
            token_mismatch_sents += 1

        # Pozisyon bazlı hizalama (min uzunluk kadar karşılaştır)
        n = min(len(gold_tokens), len(pred_tokens))
        sent_las_ok = True
        # Fazla/eksik token'lar doğrudan hata
        extra = abs(len(gold_tokens) - len(pred_tokens))

        for k in range(n):
            g = gold_tokens[k]
            p = pred_tokens[k]
            total_tokens += 1

            deprel_counts[g["deprel"]] += 1
            upos_counts[g["upos"]] += 1

            # UPOS
            if p.upos == g["upos"]:
                upos_correct += 1
                upos_hits[g["upos"]] += 1

            # UAS (head)
            head_ok = p.head == g["head"]
            if head_ok:
                uas_correct += 1

            # Deprel
            deprel_ok = p.deprel == g["deprel"]
            if deprel_ok:
                deprel_correct += 1

            # LAS (head + deprel)
            las_ok = head_ok and deprel_ok
            if las_ok:
                las_correct += 1
                deprel_hits[g["deprel"]] += 1
            else:
                sent_las_ok = False
                deprel_confusion[g["deprel"]][p.deprel] += 1

                if collect_errors and len(errors) < collect_errors:
                    errors.append({
                        "sent_id": sent_id,
                        "text": text[:60],
                        "gold_form": g["form"],
                        "pred_form": p.form,
                        "gold_head": g["head"],
                        "gold_deprel": g["deprel"],
                        "pred_head": p.head,
                        "pred_deprel": p.deprel,
                    })

        # Hizalanmayan kalan token'lar hata olarak say
        if len(gold_tokens) > n:
            for k in range(n, len(gold_tokens)):
                total_tokens += 1
                g = gold_tokens[k]
                deprel_counts[g["deprel"]] += 1
                upos_counts[g["upos"]] += 1
                deprel_confusion[g["deprel"]]["_MISSING_"] += 1
                sent_las_ok = False

        if sent_las_ok and extra == 0:
            sent_perfect_las += 1

    return {
        "sent_count": sent_count,
        "sent_perfect_las": sent_perfect_las,
        "total_tokens": total_tokens,
        "token_mismatch_sents": token_mismatch_sents,
        "uas": uas_correct,
        "las": las_correct,
        "deprel_correct": deprel_correct,
        "upos_correct": upos_correct,
        "deprel_counts": deprel_counts,
        "deprel_hits": deprel_hits,
        "deprel_confusion": deprel_confusion,
        "upos_counts": upos_counts,
        "upos_hits": upos_hits,
        "errors": errors,
        "deprel_matrix": deprel_matrix,
    }


# ═══════════════════════════════════════════════════════════════════
#  Raporlama
# ═══════════════════════════════════════════════════════════════════

def _pct(n, d):
    return f"{100 * n / d:.1f}%" if d > 0 else "—"


def print_report(r: dict):
    W = 70
    print("═" * W)
    print("  BAĞIMLILIK ÇÖZÜMLEYİCİ BENCHMARK — BOUN Treebank (PUNCT hariç)")
    print("═" * W)

    print(f"\n  Cümle sayısı     : {r['sent_count']}")
    print(f"  Token sayısı     : {r['total_tokens']}  (PUNCT çıkarılıp yeniden hizalandı)")
    print(f"  Token uyuşmazlık : {r['token_mismatch_sents']}/{r['sent_count']} cümlede farklı token sayısı")
    print(f"  Perfect LAS cümle: {r['sent_perfect_las']}/{r['sent_count']}  "
          f"({_pct(r['sent_perfect_las'], r['sent_count'])})")

    print(f"\n{'─' * W}")
    print(f"  {'Metrik':<40} {'Değer':>12}")
    print(f"{'─' * W}")
    print(f"  {'UAS (Head doğruluğu)':<40} "
          f"{_pct(r['uas'], r['total_tokens']):>12}")
    print(f"  {'LAS (Head + Deprel)':<40} "
          f"{_pct(r['las'], r['total_tokens']):>12}")
    print(f"  {'Deprel doğruluğu (deprel eşleşme)':<40} "
          f"{_pct(r['deprel_correct'], r['total_tokens']):>12}")
    print(f"  {'UPOS doğruluğu':<40} "
          f"{_pct(r['upos_correct'], r['total_tokens']):>12}")

    # ── Deprel bazlı detay ──
    print(f"\n{'─' * W}")
    print(f"  {'Deprel':<18} {'Gold':>6} {'LAS Hit':>8} {'Oran':>8}")
    print(f"{'─' * W}")
    for dep, cnt in r["deprel_counts"].most_common():
        hits = r["deprel_hits"].get(dep, 0)
        print(f"  {dep:<18} {cnt:>6} {hits:>8} {_pct(hits, cnt):>8}")

    # ── UPOS bazlı detay ──
    print(f"\n{'─' * W}")
    print(f"  {'UPOS':<10} {'Gold':>6} {'Hit':>6} {'Oran':>8}")
    print(f"{'─' * W}")
    for upos, cnt in r["upos_counts"].most_common():
        hits = r["upos_hits"].get(upos, 0)
        print(f"  {upos:<10} {cnt:>6} {hits:>6} {_pct(hits, cnt):>8}")

    # ── Hata örnekleri ──
    if r["errors"]:
        print(f"\n{'─' * W}")
        print(f"  İlk {len(r['errors'])} hata örneği:")
        print(f"{'─' * W}")
        for e in r["errors"]:
            gf = e.get("gold_form", "?")
            pf = e.get("pred_form", "?")
            form_info = f'"{gf}"' if gf == pf else f'gold="{gf}" pred="{pf}"'
            print(f"  [{e['sent_id']}] {form_info}")
            print(f"    Gold: head={e['gold_head']} deprel={e['gold_deprel']}")
            print(f"    Pred: head={e['pred_head']} deprel={e['pred_deprel']}")

    # ── Deprel confusion matrix (en sık karışıklıklar) ──
    if r.get("deprel_matrix"):
        print(f"\n{'─' * W}")
        print(f"  Deprel Karışıklık Tablosu (en sık 30):")
        print(f"{'─' * W}")
        pairs = []
        for gold_dep, preds in r["deprel_confusion"].items():
            for pred_dep, cnt in preds.items():
                pairs.append((cnt, gold_dep, pred_dep))
        pairs.sort(reverse=True)
        print(f"  {'Gold':<18} {'Pred':<18} {'Sayı':>6}")
        for cnt, gold_dep, pred_dep in pairs[:30]:
            print(f"  {gold_dep:<18} {pred_dep:<18} {cnt:>6}")

    print(f"\n{'═' * W}")


# ═══════════════════════════════════════════════════════════════════
#  Ana Giriş
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Dependency parser benchmark — BOUN Treebank",
    )
    parser.add_argument(
        "--file", "-f",
        default=str(_PROJECT / "benchmark" / "test.conllu"),
        help="CoNLL-U dosya yolu (varsayılan: benchmark/test.conllu)",
    )
    parser.add_argument(
        "--max-sentences", "-n", type=int, default=0,
        help="Maksimum cümle sayısı (0 = hepsi)",
    )
    parser.add_argument(
        "--errors", "-e", type=int, default=20,
        help="Gösterilecek hata örneği sayısı",
    )
    parser.add_argument(
        "--deprel-matrix", "-m", action="store_true",
        help="Deprel karışıklık tablosunu göster",
    )
    args = parser.parse_args()

    conllu_path = Path(args.file)
    if not conllu_path.exists():
        print(f"HATA: {conllu_path} bulunamadı!", file=sys.stderr)
        sys.exit(1)

    print(f"Yükleniyor: {conllu_path.name}")
    dict_path = _PROJECT / "turkish_words.txt"
    analyzer = create_default_analyzer(
        dictionary_path=dict_path if dict_path.exists() else None,
    )
    sa = SentenceAnalyzer(analyzer)
    dp = DependencyParser()

    print(f"Değerlendirme başlıyor...")
    t0 = time.time()

    results = evaluate(
        conllu_path, sa, dp,
        max_sentences=args.max_sentences,
        collect_errors=args.errors,
        deprel_matrix=args.deprel_matrix,
    )

    elapsed = time.time() - t0
    print(f"Süre: {elapsed:.1f}s  ({results['sent_count'] / elapsed:.0f} cümle/s)\n")

    print_report(results)


if __name__ == "__main__":
    main()
