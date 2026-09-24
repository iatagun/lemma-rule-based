"""Sıfat taban sözlüğü -> resources/adj_lemmas.txt (pekiştirme ve -CIk sıfat vurgu kurallarının kapısı; frontend/stress.py).
  python -X utf8 -m dizgetts.scripts.make_adj_lexicon
Kaynak: UD Turkish BOUN + IMST + Kenet (CC BY-SA 4.0). Lemma, geçişlerinin >= %40'ı ADJ/ADV ve >= 2 kez ise (en az bir ADJ) alınır:
UD ara sıra adları ADJ etiketliyor (ana, parça, kitap, oda) -> saflık eşiği olmadan -CIk kuralı anacığım/kitapçık'ı sıfat sayıyordu.
"""
import collections
from pathlib import Path

from dizgetts.frontend.normalize import tr_lower
from dizgetts.g2ptts.data import REPO, UD

OUT = Path(__file__).resolve().parents[1] / "resources" / "adj_lemmas.txt"


def main():
    cnt = collections.defaultdict(collections.Counter)
    for paths in UD.values():
        for p in paths:
            for line in open(REPO / p, encoding="utf8"):
                c = line.split("\t")
                if len(c) > 3 and c[2].isalpha() and "-" not in c[0]:
                    cnt[tr_lower(c[2])][c[3]] += 1
    adj = {l for l, c in cnt.items() if c["ADJ"] and c["ADJ"] + c["ADV"] >= 2 and (c["ADJ"] + c["ADV"]) / sum(c.values()) >= 0.4}
    OUT.write_text("# UD Turkish BOUN+IMST+Kenet (CC BY-SA 4.0) ADJ lemmaları (ADJ/ADV payı >= %40, >= 2 kez); üretici: scripts/make_adj_lexicon.py\n"
                   + "\n".join(sorted(adj)) + "\n", encoding="utf8")
    print(len(adj), "->", OUT)


if __name__ == "__main__":
    main()
