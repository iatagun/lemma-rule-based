#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DizgeBERT-Idiom çıkarım — düz metin (cümle/satır) → deyim (VID) / eşdizim (LVC) span'leri.

Kullanım:
    python inference/predict_idiom.py --demo                                   # örnek cümleler
    python inference/predict_idiom.py --in cumleler.txt --checkpoint idiom_data/best_idiom_tagger.pt
    python inference/predict_idiom.py --in cumleler.txt --hf                   # yayınlanmış HF modeli (iki-aşama içeride)
    python inference/predict_idiom.py --demo --stage2 idiom_data/best_idiomaticity_clf_v3.pt  # yerel + stage-2

Not: `--hf` yolu iki-aşama (stage-2 idyomatiklik filtresi) YAYINLANAN modele gömülü olduğu için
otomatik uygular. Yerel `--checkpoint` yolu VARSAYILAN ham stage-1'dir; stage-2 için `--stage2`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo kökü (script bir alt dizinde)
sys.path.insert(0, str(PROJECT_ROOT))

DEMO_SENTENCES = [
    "Yalanları ortaya çıkınca patron gözden düştü .",
    "Komisyon bu konuyu ele aldı .",
    "Aileyle konuşup karar verdi .",
    "Çocuk okula gitti .",
]


def load_local(checkpoint: str, device):
    import torch
    from transformers import AutoTokenizer

    from training.train_idiom_bert import IdiomLabelSpace, IdiomTagger

    ck = torch.load(checkpoint, map_location=device)
    ls = IdiomLabelSpace(ck["label_space"])
    tok = AutoTokenizer.from_pretrained(ls.encoder_model)
    model = IdiomTagger(ls, ls.encoder_model).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, tok, ls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=None, help="cümle başına bir satır")
    ap.add_argument("--demo", action="store_true", help="gömülü örnek cümleler")
    ap.add_argument("--hf", action="store_true", help="yerel .pt yerine yayınlanmış HF modeli")
    ap.add_argument("--hf-repo", default="iatagun/DizgeBERT-Idiom")
    ap.add_argument("--checkpoint", default=str(PROJECT_ROOT / "idiom_data" / "best_idiom_tagger.pt"))
    ap.add_argument("--stage2", default=None,
                    help="yerel yolda idyomatiklik sınıflandırıcı checkpoint'i (stage-2 filtresi)")
    ap.add_argument("--stage2-thresh", type=float, default=0.5)
    args = ap.parse_args()

    if not args.demo and not args.inp:
        sys.exit("--in <dosya> veya --demo ver.")
    sentences = DEMO_SENTENCES if args.demo else [
        ln.strip() for ln in Path(args.inp).read_text(encoding="utf-8").splitlines() if ln.strip()
    ]

    if args.hf:
        import torch
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(args.hf_repo, trust_remote_code=True).eval()
        tok = AutoTokenizer.from_pretrained(args.hf_repo)
        predict_spans = lambda ws: model.predict_spans(ws, tokenizer=tok)  # noqa: E731
    else:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, tok, ls = load_local(args.checkpoint, device)

        from dizgebert_idiom.modeling_dizgebert_idiom import (
            align_words, decode_bigappy_spans, spans_from_bigappy, viterbi_decode)
        from training.train_idiom_bert import MAX_LEN

        @torch.no_grad()
        def predict_spans(ws):
            enc, kept, fp, lp = align_words(tok, ws, MAX_LEN, device)
            out = model(enc["input_ids"], enc["attention_mask"], fp, lp)
            tags1 = viterbi_decode(out["tags"][0], ls.tags)
            tags2 = viterbi_decode(out["tags2"][0], ls.tags2)
            return spans_from_bigappy(decode_bigappy_spans(tags1, tags2), ws)

        if args.stage2:
            from training.train_idiomaticity_clf import wrap_stage2
            predict_spans = wrap_stage2(predict_spans, args.stage2, args.stage2_thresh)
            print(f"[iki-aşama] stage-2 filtresi: {args.stage2} (thresh {args.stage2_thresh})")

    for sent in sentences:
        words = sent.split()
        spans = predict_spans(words)
        print(f"\n{sent}")
        if not spans:
            print("  (deyim/eşdizim span'i bulunamadı)")
        for sp in spans:
            loc = f"{sp['start']}-{sp['end']}" + (f" + {sp['start2']}-{sp['end2']}" if sp.get("gappy") else "")
            print(f"  [{sp['category']}] {sp['text']}  (kelime {loc})")


if __name__ == "__main__":
    main()
