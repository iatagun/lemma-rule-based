#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DizgeBERT-Morph çıkarım — CoNLL-U / düz metin → UPOS + XPOS + FEATS dolu CoNLL-U.

Girdi biçimleri:
  --in x.conllu          : var olan CoNLL-U; FORM/LEMMA korunur, UPOS(4)/XPOS(5)/FEATS(6) yazılır
  --in x.txt --plain     : cümle başına bir satır, kelimeler boşlukla ayrık (ön-token'lanmış)

Kullanım:
    python predict_morph.py --in benchmark/test.conllu --out morph_data/boun_pred.conllu --scheme boun
    python predict_morph.py --in cumleler.txt --plain --out cikti.conllu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo kökü (script bir alt dizinde)
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoTokenizer

from training.train_morph_bert import DATA_DIR, LabelSpace, MorphTagger, TB_TO_ID


def load_model(checkpoint: str, device):
    ck = torch.load(checkpoint, map_location=device)
    ls = LabelSpace(ck["label_space"])
    tok = AutoTokenizer.from_pretrained(ls.encoder_model)
    model = MorphTagger(ls, ls.encoder_model).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, tok, ls


@torch.no_grad()
def tag_sentence(words: list[str], model, tok, ls: LabelSpace, scheme: str, device):
    enc = tok(words, is_split_into_words=True, return_tensors="pt",
              truncation=True, max_length=512).to(device)
    first, last = {}, {}
    for i, wid in enumerate(enc.word_ids()):
        if wid is None:
            continue
        first.setdefault(wid, i)
        last[wid] = i
    kept = sorted(first)
    fpos = torch.tensor([[first[w] for w in kept]], device=device)
    lpos = torch.tensor([[last[w] for w in kept]], device=device)
    tb = torch.tensor([TB_TO_ID[scheme]], device=device)
    out = model(enc["input_ids"], enc["attention_mask"], tb, fpos, lpos)
    up = out["upos"].argmax(-1)[0]
    xp = out["xpos"].argmax(-1)[0]
    fp = {n: out["feats"][n].argmax(-1)[0] for n in ls.feat_names}

    rows = []
    for k in range(len(kept)):
        pairs = {n: ls.feat_values[n][int(fp[n][k])] for n in ls.feat_names}
        rows.append((ls.upos[up[k]], ls.xpos[xp[k]], ls.feats_to_string(pairs)))
    while len(rows) < len(words):  # truncation olduysa
        rows.append(("X", "_", "_"))
    return rows


def run_conllu(in_path: Path, out_path: Path, model, tok, ls, scheme, device):
    import conllu

    lines_out = []
    with open(in_path, encoding="utf-8") as f:
        for sent in conllu.parse_incr(f):
            toks = [t for t in sent if isinstance(t["id"], int)]
            preds = tag_sentence([t["form"] for t in toks], model, tok, ls, scheme, device)
            pi = 0
            for meta in sent.metadata:
                lines_out.append(f"# {meta} = {sent.metadata[meta]}")
            for t in sent:
                if not isinstance(t["id"], int):
                    # MWT / boş düğüm satırını olduğu gibi bırak
                    lines_out.append(_raw(t))
                    continue
                upos, xpos, feats = preds[pi]
                pi += 1
                lines_out.append("\t".join([
                    str(t["id"]), t["form"], t.get("lemma") or "_",
                    upos, xpos, feats,
                    "_" if t.get("head") is None else str(t["head"]),
                    t.get("deprel") or "_", "_", "_",
                ]))
            lines_out.append("")
    out_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")


def _raw(tok) -> str:
    tid = tok["id"]
    tid = f"{tid[0]}-{tid[2]}" if isinstance(tid, tuple) and tid[1] == "-" else \
          (f"{tid[0]}.{tid[2]}" if isinstance(tid, tuple) else str(tid))
    return "\t".join([
        tid, tok["form"], tok.get("lemma") or "_", tok.get("upos") or "_",
        tok.get("xpos") or "_", "_", "_", "_", "_", "_",
    ])


def run_plain(in_path: Path, out_path: Path, model, tok, ls, scheme, device):
    lines_out = []
    for ln, line in enumerate(in_path.read_text(encoding="utf-8").splitlines(), 1):
        words = line.split()
        if not words:
            continue
        preds = tag_sentence(words, model, tok, ls, scheme, device)
        lines_out.append(f"# sent_id = {ln}")
        lines_out.append(f"# text = {line.strip()}")
        for i, (w, (upos, xpos, feats)) in enumerate(zip(words, preds), 1):
            lines_out.append("\t".join([str(i), w, "_", upos, xpos, feats, "_", "_", "_", "_"]))
        lines_out.append("")
    out_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plain", action="store_true", help="girdi düz metin (cümle/satır)")
    ap.add_argument("--scheme", choices=["kenet", "boun"], default="kenet")
    ap.add_argument("--checkpoint", default=str(DATA_DIR / "best_morph_tagger.pt"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, ls = load_model(args.checkpoint, device)
    runner = run_plain if args.plain else run_conllu
    runner(Path(args.inp), Path(args.out), model, tok, ls, args.scheme, device)
    print(f"yazıldı: {args.out}  (scheme={args.scheme})")


if __name__ == "__main__":
    main()
