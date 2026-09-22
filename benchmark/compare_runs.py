#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""İki koşunun Çavuşoğlu çift-düzeyi sonuçlarını EŞLEŞTİRİLMİŞ bootstrap'la kıyaslar.

Proje disiplini: bir stage-2 adayı promote edilmeden önce farkın istatistiksel anlamlılığı
ölçülür (Deney AA'da 4-gövde ensemble adayı +0.5pp ile GA sıfırı içerdiği için reddedilmişti).
Eşleştirilmiş olması şart — aynı 198 çift iki modelde de koşuluyor, bağımsız oran GA'ları
gereksiz geniş çıkar.

Kullanım:
    python benchmark/eval_idiom.py --local --ensemble ... --stage2 A.pt --mode external \
        --dump-hits /tmp/a.json
    python benchmark/eval_idiom.py --local --ensemble ... --stage2 B.pt --mode external \
        --dump-hits /tmp/b.json
    python benchmark/compare_runs.py /tmp/a.json /tmp/b.json --names v8 AB-1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.stats_utils import paired_diff_ci, proportion_ci  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--field", default="both",
                    choices=["both", "sample", "literal"],
                    help="both=doğru-ayırt (varsayılan), sample=duyarlılık, literal=yanlış-poz")
    args = ap.parse_args()

    A = json.loads(Path(args.a).read_text(encoding="utf-8"))
    B = json.loads(Path(args.b).read_text(encoding="utf-8"))
    if [r["idiom"] for r in A] != [r["idiom"] for r in B]:
        sys.exit("çiftler hizalanmıyor — iki koşu AYNI tsv ve AYNI filtreyle yapılmalı")

    ha = [bool(r[args.field]) for r in A]
    hb = [bool(r[args.field]) for r in B]
    na, nb = args.names
    for nm, h in ((na, ha), (nb, hb)):
        p, lo, hi = proportion_ci(h)
        print(f"  {nm:10s} {args.field}: %{100*p:.1f}  (95% GA %{100*lo:.1f}–%{100*hi:.1f})")
    diff, lo, hi = paired_diff_ci(ha, hb)
    sig = "ANLAMLI" if (lo > 0 or hi < 0) else "ANLAMSIZ (GA sıfırı içeriyor)"
    print(f"  eşleştirilmiş fark ({nb} − {na}): {100*diff:+.1f}pp  "
          f"(95% GA {100*lo:+.1f}/{100*hi:+.1f})  → {sig}")
    # nereye kaydığını göster
    gain = sum(1 for x, y in zip(ha, hb) if not x and y)
    loss = sum(1 for x, y in zip(ha, hb) if x and not y)
    print(f"  {nb} kazandığı: {gain} çift, kaybettiği: {loss} çift (net {gain - loss:+d})")


if __name__ == "__main__":
    main()
