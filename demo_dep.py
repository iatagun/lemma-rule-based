#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bağımlılık çözümleme (dependency parsing) demo.

Kural-tabanlı morfolojik çözümleme üzerine inşa edilen bağımlılık
ayrıştırıcısının çıktılarını gösterir:
  • CoNLL-U formatı  (Universal Dependencies standardı)
  • ASCII ağaç görselleştirmesi
  • Kural uygulama izleme (hangi kural hangi ilişkiyi atadı)

Kullanım:
    python demo_dep.py                   # tüm test cümleleri
    python demo_dep.py --conllu          # sadece CoNLL-U çıktısı
    python demo_dep.py --tree            # sadece ağaç görselleştirmesi
    python demo_dep.py --sentence "..."  # tek cümle analizi
"""

import sys
import argparse
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from morphology import create_default_analyzer
from morphology.sentence import SentenceAnalyzer
from morphology.dependency import DependencyParser

# ═══════════════════════════════════════════════════════════════════
#  Kurulum
# ═══════════════════════════════════════════════════════════════════

_DICT = Path(__file__).resolve().parent / "turkish_words.txt"
analyzer = create_default_analyzer(
    dictionary_path=_DICT if _DICT.exists() else None,
)
sa = SentenceAnalyzer(analyzer)
dp = DependencyParser()

W = 82  # terminal genişliği

# ═══════════════════════════════════════════════════════════════════
#  Test Cümleleri — Kategorize
# ═══════════════════════════════════════════════════════════════════

SENTENCES = [
    # ── Temel Cümle Yapıları ──────────────────────────────────
    ("Temel S-V",          "Ali okudu"),
    ("Temel S-O-V",        "Ali kitabı okudu"),
    ("S-IO-O-V",           "Çocuk kediye süt verdi"),

    # ── İsim Öbeği (NP) Yapıları ─────────────────────────────
    ("Det + N",            "Bu kitabı aldım"),
    ("Det + Adj + N",      "Bu güzel kitabı okudum"),
    ("Belirtili tamlama",  "Ali'nin kedisi uyudu"),

    # ── Hal Ekleri ve Edatlar ────────────────────────────────
    ("Bulunma hali",       "Çocuk okulda ders çalıştı"),
    ("Edat yapısı",        "Ev için para biriktiriyoruz"),

    # ── Zarf ve Bağlaç ───────────────────────────────────────
    ("Zarf",               "Çocuk çok hızlı koştu"),
    ("Bağlaç",             "Ali ve Ayşe geldi"),

    # ── Birleşik Yapılar ─────────────────────────────────────
    ("Zarf-fiil (advcl)",  "Koşarak eve geldi"),
    ("Sıfat-fiil (acl)",   "Okunan kitap güzeldi"),
    ("Fiilsiz cümle",      "Türkiye büyük bir ülkedir"),

    # ── v2: Hafif Fiil (compound:lvc) ────────────────────────
    ("Hafif fiil",         "Yardım etti"),
    ("Hafif fiil 2",       "Dans etmeyi seviyorum"),

    # ── v2: Sayı (nummod) ───────────────────────────────────
    ("Sayı + isim",        "Üç kitap okudum"),
    ("Rakam + isim",       "Yüz kişi toplandı"),

    # ── v2: Özel İsim (flat) ───────────────────────────────
    ("Özel isim",          "Mustafa Kemal geldi"),
    ("Özel isim + hal",    "Ali Veli okula gitti"),

    # ── v2: Zaman İsmi (obl:tmod) ───────────────────────────
    ("Zaman ismi",         "Akşam geldim"),
    ("Çoklu zaman",        "Dün akşam eve geldim"),

    # ── v2: Sıfat-Zarf Belirsizliği ─────────────────────────
    ("Sıfat→amod",         "Güzel araba geldi"),
    ("Sıfat→advmod",       "Hızlı koştu"),

    # ── v2: Scope-Aware Sıfat-fiil ──────────────────────────
    ("RC scope",           "Okula giden çocuk güldü"),
]


# ═══════════════════════════════════════════════════════════════════
#  Çıktı Fonksiyonları
# ═══════════════════════════════════════════════════════════════════

def _box(text: str, char: str = "═") -> None:
    bw = W - 4
    print(f"  ╔{char * bw}╗")
    print(f"  ║ {text:<{bw - 1}}║")
    print(f"  ╚{char * bw}╝")


def _separator() -> None:
    print("  " + "─" * (W - 2))


def _print_table(dep_tokens) -> None:
    """Bağımlılık tablosunu UD formatında yazdırır."""
    hdr = f"  {'ID':>3}  {'Form':<18} {'Lemma':<15} {'UPOS':<6} {'Feats':<28} {'Head':>4}  {'Deprel':<12}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for t in dep_tokens:
        feats = t.feats_str if t.feats_str != "_" else "—"
        print(
            f"  {t.id:>3}  {t.form:<18} {t.lemma:<15} {t.upos:<6} "
            f"{feats:<28} {t.head:>4}  {t.deprel:<12}"
        )


def _print_relations(dep_tokens) -> None:
    """Sözdizimsel ilişkileri doğal dilde gösterir."""
    root_tok = None
    for t in dep_tokens:
        if t.deprel == "root":
            root_tok = t
            break
    if not root_tok:
        return

    id_map = {t.id: t for t in dep_tokens}
    for t in dep_tokens:
        if t.deprel == "root":
            print(f"    ● {t.form} ← root (yüklem)")
        elif t.head in id_map:
            head = id_map[t.head]
            arrow = f"{t.form} ──{t.deprel}──▶ {head.form}"
            print(f"    {arrow}")


def analyze_sentence(text: str, label: str = "",
                     show_conllu: bool = True,
                     show_tree: bool = True,
                     show_table: bool = True) -> list:
    """Tek cümleyi analiz edip çıktıları yazdırır."""
    tokens = sa.analyze(text)
    dep_tokens = dp.parse(tokens)

    # Başlık
    print()
    if label:
        _separator()
        print(f"  ┃ [{label}]  {text}")
        _separator()
    else:
        _separator()
        print(f"  ┃ {text}")
        _separator()

    # Tablo
    if show_table:
        print()
        _print_table(dep_tokens)

    # CoNLL-U
    if show_conllu:
        print()
        print("  ┌─ CoNLL-U ─────────────────────────────────────────────┐")
        conllu = dp.to_conllu(dep_tokens, text=text)
        for line in conllu.strip().splitlines():
            print(f"  │ {line}")
        print("  └───────────────────────────────────────────────────────┘")

    # Ağaç
    if show_tree:
        print()
        print("  ┌─ Ağaç ───────────────────────────────────────────────┐")
        tree = dp.to_tree(dep_tokens)
        for line in tree.splitlines():
            print(f"  │ {line}")
        print("  └───────────────────────────────────────────────────────┘")

    # İlişkiler
    print()
    _print_relations(dep_tokens)

    return dep_tokens


# ═══════════════════════════════════════════════════════════════════
#  İstatistik Özeti
# ═══════════════════════════════════════════════════════════════════

def _print_summary(all_results: list[tuple[str, list]]) -> None:
    """Tüm cümlelerin deprel dağılımını gösterir."""
    deprel_counts: dict[str, int] = {}
    upos_counts: dict[str, int] = {}
    total_tokens = 0
    fallback_count = 0

    for _, dep_tokens in all_results:
        for t in dep_tokens:
            total_tokens += 1
            deprel_counts[t.deprel] = deprel_counts.get(t.deprel, 0) + 1
            upos_counts[t.upos] = upos_counts.get(t.upos, 0) + 1
            if t.deprel == "dep":
                fallback_count += 1

    assigned = total_tokens - fallback_count
    pct = (assigned / total_tokens * 100) if total_tokens else 0

    print()
    _box("  İstatistik Özeti")
    print()

    # Deprel dağılımı
    print("  Bağımlılık İlişkileri (deprel):")
    for rel, cnt in sorted(deprel_counts.items(), key=lambda x: -x[1]):
        bar = "█" * cnt
        print(f"    {rel:<14} {cnt:>3}  {bar}")

    print()

    # UPOS dağılımı
    print("  Sözcük Türleri (UPOS):")
    for pos, cnt in sorted(upos_counts.items(), key=lambda x: -x[1]):
        bar = "▓" * cnt
        print(f"    {pos:<8} {cnt:>3}  {bar}")

    print()
    print(f"  Toplam sözcük    : {total_tokens}")
    print(f"  Atanan ilişki    : {assigned} / {total_tokens}  ({pct:.1f}%)")
    if fallback_count:
        print(f"  Fallback (dep)   : {fallback_count}")
    print()


# ═══════════════════════════════════════════════════════════════════
#  Ana Program
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Kural-tabanlı bağımlılık çözümleme demo"
    )
    parser.add_argument(
        "--sentence", "-s", type=str, default=None,
        help="Tek cümle analiz et"
    )
    parser.add_argument(
        "--conllu", action="store_true",
        help="Sadece CoNLL-U çıktısı göster"
    )
    parser.add_argument(
        "--tree", action="store_true",
        help="Sadece ağaç görselleştirmesi göster"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Sadece tablo ve ağaç (CoNLL-U yok)"
    )
    args = parser.parse_args()

    # Mod seçimi
    show_conllu = not args.tree and not args.quiet
    show_tree = not args.conllu
    show_table = not args.conllu and not args.tree

    if args.conllu:
        show_conllu = True
        show_tree = False
        show_table = False
    if args.tree:
        show_conllu = False
        show_tree = True
        show_table = False

    # Başlık
    print()
    _box("Kural-Tabanlı Bağımlılık Çözümleme  ·  Rule-Based Dependency Parser")
    print("  20 kural  ·  SOLID mimarisi  ·  CoNLL-U uyumlu  ·  UD standartları")
    print()

    # Tek cümle modu
    if args.sentence:
        analyze_sentence(
            args.sentence, label="Kullanıcı",
            show_conllu=show_conllu, show_tree=show_tree,
            show_table=show_table,
        )
        return

    # Tüm test cümleleri
    all_results = []
    for label, text in SENTENCES:
        dep_tokens = analyze_sentence(
            text, label=label,
            show_conllu=show_conllu, show_tree=show_tree,
            show_table=show_table,
        )
        all_results.append((text, dep_tokens))

    # İstatistik
    if show_table:
        _print_summary(all_results)


if __name__ == "__main__":
    main()
