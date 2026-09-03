"""
Toplu CoNLL-U lemmatizasyon aracı.

Girdi: CoNLL-U formatında dosya (veya stdin)
Çıktı: LEMMA sütunu doldurulmuş CoNLL-U

Kullanım:
  python batch_conllu.py input.conllu > output.conllu
  python batch_conllu.py --stdin < input.conllu > output.conllu

POS etiketi olmayan basit token listesi için:
  python batch_conllu.py --plain tokens.txt > output.conllu
  (Her satır bir token, boş satır cümle ayracı)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from morphology import create_default_analyzer
from morphology.phonology import turkish_lower


def lemmatize_conllu(input_path: str, output_path: str | None = None) -> None:
    """CoNLL-U dosyasını okuyup LEMMA sütununu doldurur."""
    dict_path = PROJECT_ROOT / "turkish_words.txt"
    print("Sözlük yükleniyor...", file=sys.stderr, flush=True)
    analyzer = create_default_analyzer(str(dict_path))

    out = open(output_path, "w", encoding="utf-8") if output_path else sys.stdout

    lines = []
    total = 0
    lemmatized = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                out.write(line + "\n")
                continue
            cols = line.split("\t")
            if len(cols) < 2:
                out.write(line + "\n")
                continue
            # Çok sözcüklü token satırı veya boş düğüm
            if "-" in cols[0] or "." in cols[0]:
                out.write(line + "\n")
                continue

            word = cols[1]
            upos = cols[3] if len(cols) > 3 and cols[3] != "_" else None

            try:
                result = analyzer.analyze(word, upos=upos)
                lemma = result.root if result.root else result.stem
                lemma = turkish_lower(lemma)
            except Exception:
                lemma = turkish_lower(word)

            # LEMMA sütununu güncelle (3. sütun, indeks 2)
            if len(cols) <= 2:
                cols.extend(["_"] * (10 - len(cols)))
            cols[2] = lemma
            out.write("\t".join(cols) + "\n")

            total += 1
            if lemma != turkish_lower(word):
                lemmatized += 1

    if output_path:
        out.close()

    print(f"Toplam token: {total}  Lemmatize edilen: {lemmatized}  "
          f"Değişmeyen: {total - lemmatized}", file=sys.stderr, flush=True)


def lemmatize_plain(input_path: str, output_path: str | None = None) -> None:
    """Basit token listesini CoNLL-U formatına çevirir.

    Girdi: her satırda bir token, boş satır cümle ayracı.
    Çıktı: CoNLL-U formatı.
    """
    dict_path = PROJECT_ROOT / "turkish_words.txt"
    print("Sözlük yükleniyor...", file=sys.stderr, flush=True)
    analyzer = create_default_analyzer(str(dict_path))

    out = open(output_path, "w", encoding="utf-8") if output_path else sys.stdout

    sent_id = 0
    token_id = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if token_id > 0:
                    out.write("\n")
                sent_id += 1
                token_id = 0
                continue

            word = line
            token_id += 1

            try:
                result = analyzer.analyze(word)
                lemma = result.root if result.root else result.stem
                lemma = turkish_lower(lemma)
            except Exception:
                lemma = turkish_lower(word)

            out.write(f"{token_id}\t{word}\t{lemma}\t_\t_\t_\t_\t_\t_\t_\n")

    if output_path:
        out.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CoNLL-U lemmatizasyon aracı"
    )
    parser.add_argument(
        "input", nargs="?", help="Girdi dosyası (verilmezse stdin)"
    )
    parser.add_argument(
        "-o", "--output", help="Çıktı dosyası (varsayılan: stdout)"
    )
    parser.add_argument(
        "--plain", action="store_true",
        help="Girdi basit token listesi (CoNLL-U değil)"
    )
    args = parser.parse_args()

    if args.plain:
        if args.input:
            lemmatize_plain(args.input, args.output)
        else:
            print("HATA: --plain modunda girdi dosyası gerekli", file=sys.stderr)
            sys.exit(1)
    else:
        if args.input:
            lemmatize_conllu(args.input, args.output)
        else:
            print("HATA: Girdi dosyası gerekli", file=sys.stderr)
            sys.exit(1)
