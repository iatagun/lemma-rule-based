#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leipzig Corpora Collection — Türkçe cümle dosyalarını indirir.

DizgeBERT-Idiom bağlamı: TDK deyim sözlüğündeki deyimlerin (`kind==idiom`) gerçek bağlam
cümlelerini bulmak için ham metin kaynağı. TDK'nin kendi gömülü örnek cümleleri deyim başına
genelde tek ve sınırlı; büyük bir derlemde sıkı gövde-eşleştirmesiyle taranınca deyim başına
onlarca *temiz* örnek çıkar (bkz. prepare_tdk_corpus_examples.py).

Leipzig dosyaları ZATEN cümle-başına-satır, temizlenmiş, tekilleştirilmiş — wikiextractor /
cümle-segmentasyonu gerekmez. Format: her satır `<id>\\t<cümle>`. Lisans: CC BY (atıf:
D. Goldhahn, T. Eckart & U. Quasthoff, "Building Large Monolingual Dictionaries at the Leipzig
Corpora Collection", LREC 2012).

downloads.wortschatz-leipzig.de/corpora/ üzerinde Anubis (anti-scraping) YOK — düz curl/GET
çalışıyor (ana site wortschatz.uni-leipzig.de'de var).

Kullanım:
    python fetch_leipzig_tr.py                 # varsayılan 3 derlem (~590 MB, ~3M cümle)
    python fetch_leipzig_tr.py --corpora tur_wikipedia_2021_1M tur_news_2024_1M
"""
from __future__ import annotations

import argparse
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "idiom_data" / "raw" / "leipzig"
BASE_URL = "https://downloads.wortschatz-leipzig.de/corpora/{}.tar.gz"

# Wikipedia + News + Web karışımı (kullanıcı seçimi). Hepsi 1M boyutu.
DEFAULT_CORPORA = [
    "tur_wikipedia_2021_1M",
    "tur_news_2024_1M",
    "tur-tr_web_2019_1M",
]


def fetch_one(corpus_id: str) -> Path:
    out_txt = OUT_DIR / f"{corpus_id}-sentences.txt"
    if out_txt.exists() and out_txt.stat().st_size > 0:
        print(f"  atlandı (var): {out_txt.name}  ({out_txt.stat().st_size / 1048576:.0f} MB)")
        return out_txt

    url = BASE_URL.format(corpus_id)
    print(f"  indiriliyor: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        blob = resp.read()
    print(f"  {len(blob) / 1048576:.0f} MB indirildi, açılıyor…")

    # tar içindeki tek gerekli üye: <corpus_id>/<corpus_id>-sentences.txt
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        member = next((m for m in tf.getmembers() if m.name.endswith("-sentences.txt")), None)
        if member is None:
            sys.exit(f"  HATA: {corpus_id} içinde -sentences.txt yok")
        with tf.extractfile(member) as src:
            out_txt.write_bytes(src.read())
    n = sum(1 for _ in out_txt.open(encoding="utf-8"))
    print(f"  yazıldı: {out_txt.relative_to(PROJECT_ROOT)}  ({n:,} cümle, "
          f"{out_txt.stat().st_size / 1048576:.0f} MB)")
    return out_txt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+", default=DEFAULT_CORPORA)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== Leipzig Türkçe derlem indirme → {OUT_DIR.relative_to(PROJECT_ROOT)} ===")
    total = 0
    for cid in args.corpora:
        p = fetch_one(cid)
        total += sum(1 for _ in p.open(encoding="utf-8"))
    print(f"\nToplam {total:,} cümle. Sonraki: python prepare_tdk_corpus_examples.py")


if __name__ == "__main__":
    main()
