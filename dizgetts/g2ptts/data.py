"""dizge-g2p-tts eğitim verisi: belirteç başına VURGU (kural damıtması) + SINIR (sesten ölçülen) etiketleri -> <out>/{train,val,test}.jsonl
  python -X utf8 -m dizgetts.g2ptts.data

Satır: {src, id, tokens, stress, boundary}; etiket -100 = kayıp dışı.
  stress  : 0 = vurgusuz (clitic/ünlüsüz), 1+r = vurgulu seslem sondan r (r>=3 -> 4). Kaynak: StressRules.syllable(..., TIERS) — UD'de GOLD UPOS/FEATS,
            Antalia'da DizgeBERT-Morph önbelleği (morph_feats.jsonl). Model ham metinden görür -> biçimbilimi kendisi çıkarmalı.
  boundary: yalnız Antalia; sözcük k'den sonraki sessizlik (breaks.jsonl): 0 (<60 ms) / 1 ip (60-250) / 2 IP (>=250). Klibin son sözcüğü -100 (hep cümle).
UD: BOUN+IMST (morph_data/raw) + Kenet (data/treebanks); çok-parçalı belirteçli cümleler (haldeyim = halde+yim) atlanır: yüzey biçimine tek
biçimbilim etiketi yok. Kesme işareti atılır (normalize() gibi: Nisan'da -> Nisanda).
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import yaml

from dizgetts.frontend.morph import parse_feats
from dizgetts.frontend.stress import TIERS, StressRules, _n_vowels
from dizgetts.frontend.normalize import tr_lower

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1]
UD = {"train": ["morph_data/raw/tr_boun-ud-train.conllu", "morph_data/raw/tr_imst-ud-train.conllu", "data/treebanks/UD_Turkish-Kenet/tr_kenet-ud-train.conllu"],
      "val": ["morph_data/raw/tr_boun-ud-dev.conllu", "morph_data/raw/tr_imst-ud-dev.conllu", "data/treebanks/UD_Turkish-Kenet/tr_kenet-ud-dev.conllu"],
      "test": ["morph_data/raw/tr_boun-ud-test.conllu", "morph_data/raw/tr_imst-ud-test.conllu", "data/treebanks/UD_Turkish-Kenet/tr_kenet-ud-test.conllu"]}
OUT = "D:/dizgetts/data/g2ptts"
R = StressRules()


def stress_label(tok: str, upos: str | None, feats: dict | None) -> int:
    if not tok.isalpha():
        return -100
    k, _ = R.syllable(tok, upos, feats, TIERS)
    if k is None:
        return 0
    return 1 + min(_n_vowels(tr_lower(tok)) - 1 - k, 3)


def read_ud(path: str):
    sent, sid, mwt = [], None, False
    for line in open(REPO / path, encoding="utf8"):
        line = line.rstrip("\n")
        if line.startswith("# sent_id"):
            sid = line.split("=", 1)[1].strip()
        elif not line:
            if sent and not mwt:
                yield sid, sent
            sent, mwt = [], False
        elif not line.startswith("#"):
            c = line.split("\t")
            if "-" in c[0]:
                mwt = True
            elif "." not in c[0]:
                sent.append((c[1].replace("'", "").replace("’", ""), c[3], parse_feats(c[5])))


def main():
    root = yaml.safe_load(open(HERE / "configs" / "data.yaml", encoding="utf8"))["out_root"]
    morph = {d["id"]: d for d in map(json.loads, open(os.path.join(root, "morph_feats.jsonl"), encoding="utf8"))}
    breaks = {d["id"]: d for d in map(json.loads, open(os.path.join(root, "breaks.jsonl"), encoding="utf8"))}
    os.makedirs(OUT, exist_ok=True)
    stats = {}
    for sp in ("train", "val", "test"):
        n = {"ud": 0, "antalia": 0, "antalia_sınırsız": 0}
        with open(f"{OUT}/{sp}.jsonl", "w", encoding="utf8") as out:
            for path in UD[sp]:
                for sid, sent in read_ud(path):
                    toks = [t for t, _, _ in sent]
                    out.write(json.dumps(dict(src="ud", id=f"{Path(path).stem}/{sid}", tokens=toks,
                                              stress=[stress_label(t, u, f) for t, u, f in sent], boundary=[-100] * len(toks)), ensure_ascii=False) + "\n")
                    n["ud"] += 1
            for l in open(os.path.join(root, f"{sp}_phon.jsonl"), encoding="utf8"):
                r = json.loads(l)
                m = morph[r["id"]]
                toks = m["tokens"]
                bnd = [-100] * len(toks)
                b = breaks.get(r["id"])
                widx = [i for i, t in enumerate(toks) if t.isalpha()]
                if b and len(widx) == b["n_words"] == b["n_words_dizge"]:
                    for bd in b["boundaries"]:
                        if bd["k"] + 1 < len(widx):
                            s = bd["silence_ms"]
                            bnd[widx[bd["k"]]] = 0 if s < 60 else 1 if s < 250 else 2
                else:
                    n["antalia_sınırsız"] += 1
                out.write(json.dumps(dict(src="antalia", id=r["id"], tokens=toks, stress=[stress_label(t, u, f) for t, u, f in zip(toks, m["upos"], m["feats"])],
                                          boundary=bnd), ensure_ascii=False) + "\n")
                n["antalia"] += 1
        stats[sp] = n
    stats["stress_rules_version"] = R.version()
    stats["tiers"] = TIERS
    json.dump(stats, open(f"{OUT}/stats.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
