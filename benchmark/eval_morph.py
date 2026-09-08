#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DizgeBERT-Morph benchmark — UD morfolojik etiketleme (UPOS + XPOS + FEATS).

Gold: bir CoNLL-U dosyası (varsayılan `benchmark/test.conllu` = BOUN test, held-out).
Modlar:
  neural   (varsayılan) — eğitilmiş DizgeBERT-Morph checkpoint'i
  majority              — train.json'dan (treebank, form) → en sık (upos,xpos,feats)

Metrikler (treebank başına): UPOS acc, XPOS acc, UFeats micro-F1 (CoNLL-18),
FEATS exact-match, full-tag acc, özellik-başına F1. PUNCT'lu / PUNCT'suz ayrı.

Kullanım:
    python benchmark/eval_morph.py                                    # BOUN test, neural
    python benchmark/eval_morph.py --file data/treebanks/UD_Turkish-Kenet/tr_kenet-ud-test.conllu --scheme kenet
    python benchmark/eval_morph.py --mode majority
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

import torch
from torch.utils.data import DataLoader

from data.prepare_morph_data_ud import canon_feats, iter_sentences, sentence_to_record
from training.train_morph_bert import (
    DATA_DIR,
    TREEBANKS,
    LabelSpace,
    MorphDataset,
    MorphTagger,
    evaluate,
    make_collate,
    print_eval,
)

REFERENCE = (
    "Yayınlanmış referanslar (tr UFeats F1): UDPipe-2 ~94, Stanza ~92-94, "
    "Trankit ~93-95 (in-domain)."
)


def records_from_conllu(path: Path, treebank: str) -> list[dict]:
    return [sentence_to_record(toks, treebank) for toks in iter_sentences(path)]


# ─────────────────────────────────────────────────────────────────────────────
#  Baseline'lar → aynı metrik fonksiyonuyla ölçmek için tahmin kayıtları üretir
# ─────────────────────────────────────────────────────────────────────────────
def eval_from_predictions(gold: list[dict], pred: list[dict]) -> dict:
    """Basit token-düzeyi metrik (neural evaluate ile aynı formülleri kullanır)."""

    def f1(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    def feat_set(s: str) -> set[str]:
        return set() if s == "_" else set(s.split("|"))

    agg = defaultdict(lambda: dict(
        n=0, upos_ok=0, xpos_ok=0, xpos_n=0, tp=0, fp=0, fn=0, exact=0, full=0,
        pf=defaultdict(lambda: dict(tp=0, fp=0, fn=0)),
    ))
    for g, p in zip(gold, pred):
        a = agg[g["treebank"]]
        for i in range(len(g["words"])):
            a["n"] += 1
            a["upos_ok"] += int(g["upos"][i] == p["upos"][i])
            if g["xpos"][i] != "_":
                a["xpos_n"] += 1
                a["xpos_ok"] += int(g["xpos"][i] == p["xpos"][i])
            gs, ps = feat_set(g["feats"][i]), feat_set(p["feats"][i])
            a["tp"] += len(gs & ps)
            a["fp"] += len(ps - gs)
            a["fn"] += len(gs - ps)
            a["exact"] += int(gs == ps)
            a["full"] += int(gs == ps and g["upos"][i] == p["upos"][i])
            cats = {x.split("=")[0] for x in gs | ps}
            for c in cats:
                gv = next((x for x in gs if x.startswith(c + "=")), None)
                pv = next((x for x in ps if x.startswith(c + "=")), None)
                d = a["pf"][c]
                if gv == pv:
                    d["tp"] += 1
                else:
                    if pv:
                        d["fp"] += 1
                    if gv:
                        d["fn"] += 1

    res = {}
    for tb, a in agg.items():
        if not a["n"]:
            continue
        res[tb] = {
            "n": a["n"],
            "upos_acc": round(100 * a["upos_ok"] / a["n"], 2),
            "xpos_acc": round(100 * a["xpos_ok"] / a["xpos_n"], 2) if a["xpos_n"] else None,
            "ufeats_f1": round(100 * f1(a["tp"], a["fp"], a["fn"]), 2),
            "feats_exact": round(100 * a["exact"] / a["n"], 2),
            "full_tag_acc": round(100 * a["full"] / a["n"], 2),
            "per_feature_f1": {
                c: round(100 * f1(d["tp"], d["fp"], d["fn"]), 2)
                for c, d in sorted(a["pf"].items())
            },
        }
    return res


def predict_majority(gold: list[dict]) -> list[dict]:
    import json as _j

    data = _j.loads((DATA_DIR / "train.json").read_text(encoding="utf-8"))
    table: dict[tuple, Counter] = defaultdict(Counter)
    glob = Counter()
    for r in data:
        tb = r["treebank"]
        for w, u, x, f in zip(r["words"], r["upos"], r["xpos"], r["feats"]):
            table[(tb, w.lower())][(u, x, f)] += 1
            glob[(u, x, f)] += 1
    default = glob.most_common(1)[0][0]

    out = []
    for r in gold:
        tb = r["treebank"]
        pu, px, pf = [], [], []
        for w in r["words"]:
            key = (tb, w.lower())
            u, x, f = table[key].most_common(1)[0][0] if key in table else default
            pu.append(u); px.append(x); pf.append(f)
        out.append({"treebank": tb, "words": r["words"], "upos": pu, "xpos": px, "feats": pf})
    return out


def strip_punct(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        keep = [i for i, u in enumerate(r["upos"]) if u not in ("PUNCT", "SYM")]
        out.append({k: ([r[k][i] for i in keep] if isinstance(r[k], list) else r[k])
                    for k in r})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    # NOT: benchmark/test.conllu eski bir BOUN snapshot'ı (PronType vb. eksik, 528/979 cümle
    # güncel UD master'dan farklı). Eğitim verisiyle tutarlı olması için indirilen sürümü kullan.
    ap.add_argument("--file", default=str(DATA_DIR / "raw" / "tr_boun-ud-test.conllu"))
    ap.add_argument("--treebank", choices=TREEBANKS, default="boun",
                    help="gold dosyanın şeması (metrik gruplaması + neural scheme)")
    ap.add_argument("--scheme", choices=TREEBANKS, default=None,
                    help="neural çıkarım şeması (varsayılan: --treebank)")
    ap.add_argument("--mode", choices=["neural", "majority"], default="neural")
    ap.add_argument("--checkpoint", default=str(DATA_DIR / "best_morph_tagger.pt"))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    gold = records_from_conllu(Path(args.file), args.treebank)
    print(f"Dosya: {args.file}  ({len(gold)} cümle, {sum(len(r['words']) for r in gold)} token)")
    print(f"Mod: {args.mode}   Treebank/şema: {args.treebank}\n")

    if args.mode == "majority":
        res_all = eval_from_predictions(gold, predict_majority(gold))
        res_np = eval_from_predictions(strip_punct(gold), predict_majority(strip_punct(gold)))

    else:  # neural
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(args.checkpoint, map_location=device)
        ls = LabelSpace(ck["label_space"])
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(ls.encoder_model)
        model = MorphTagger(ls, ls.encoder_model).to(device)
        model.load_state_dict(ck["model"])
        scheme = args.scheme or args.treebank
        # scheme'i zorlamak için kayıtların treebank alanını override et
        forced = [{**r, "treebank": scheme} for r in gold]
        # ama metrik gruplaması args.treebank altında görünsün diye evaluate sonrası yeniden adlandır
        col = make_collate(ls, tok.pad_token_id or 0)
        dl = DataLoader(MorphDataset(forced, tok, ls), batch_size=args.batch_size, collate_fn=col)
        res_scheme = evaluate(model, dl, device, ls)
        res_all = {args.treebank: res_scheme[scheme]} if scheme in res_scheme else res_scheme
        print(f"(neural scheme = {scheme})")
        res_np = None

    print("=" * 60)
    print("TÜM TOKEN'LAR")
    print_eval(res_all)
    if res_np is not None:
        print("\n" + "=" * 60)
        print("PUNCT/SYM HARİÇ")
        print_eval(res_np)
    print("\n" + REFERENCE)


if __name__ == "__main__":
    main()
