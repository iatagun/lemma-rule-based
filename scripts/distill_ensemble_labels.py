#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Öneri #1 — ensemble distilasyonu: Deney O'nun union(vE,vL) ensemble'ının bilgisini TEK
gövdeli bir öğrenciye aktarmak için, weak-supervision eğitim kayıtlarına (tdk_examples.json,
corpus_examples_glu.json) vE+vL'nin ORTALAMA per-kelime softmax'ını ("soft_tags"/"soft_tags2")
ekler. PARSEME altın verisine (train.json) DOKUNULMAZ — yalnız zaten zayıf-etiketli kaynaklar.

Truncation'a takılan (MAX_LEN'i aşan) kayıtlar soft_tags'siz bırakılır (has_soft=False,
IdiomDataset'te sabit-etiket-yalnız olarak eğitilir) — hizalama karmaşıklığından kaçınmak için.

Kullanım:
    python scripts/distill_ensemble_labels.py \\
        --teacher-a idiom_data/best_idiom_tagger_vE_leipzigglu.pt \\
        --teacher-b idiom_data/best_idiom_tagger_vL_idiomcoverage.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "idiom_data"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from training.train_idiom_bert import IdiomDataset, IdiomLabelSpace, IdiomTagger, MAX_LEN


def load_teacher(ckpt_path: str, device):
    ck = torch.load(ckpt_path, map_location=device)
    ls = IdiomLabelSpace(ck["label_space"])
    model = IdiomTagger(ls, ls.encoder_model).to(device).eval()
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    return model, ls


@torch.no_grad()
def soft_label_file(src_path: Path, out_path: Path, model_a, model_b, ls, tokenizer, device):
    data = json.loads(src_path.read_text(encoding="utf-8"))
    ds = IdiomDataset(src_path, tokenizer, ls)
    assert len(ds) == len(data)
    n_full, n_trunc = 0, 0
    for rec, item in zip(data, ds.items):
        n_kept = len(item["first_pos"])
        if n_kept != len(rec["words"]):
            n_trunc += 1
            continue  # truncation'a takıldı — hizalama net değil, soft_tags eklenmez
        n_full += 1
        input_ids = torch.tensor([item["input_ids"]], device=device)
        attn = torch.tensor([item["attention_mask"]], device=device)
        fp = torch.tensor([item["first_pos"]], device=device)
        lp = torch.tensor([item["last_pos"]], device=device)
        pos_ids = torch.tensor([item["pos_ids"]], device=device)

        out_a = model_a(input_ids, attn, fp, lp, pos_ids)
        out_b = model_b(input_ids, attn, fp, lp, pos_ids)
        pa1, pb1 = F.softmax(out_a["tags"][0], dim=-1), F.softmax(out_b["tags"][0], dim=-1)
        pa2, pb2 = F.softmax(out_a["tags2"][0], dim=-1), F.softmax(out_b["tags2"][0], dim=-1)
        p1 = (pa1 + pb1) / 2
        p2 = (pa2 + pb2) / 2
        rec["soft_tags"] = p1.cpu().tolist()
        rec["soft_tags2"] = p2.cpu().tolist()
        # Deney W — anlaşmazlık-ağırlıklı distilasyon: Deney R'nin damıtımı TÜM kayıtlara
        # eşit uygulaması ("Agree to Disagree", NeurIPS 2020'nin teşhis ettiği ortalama-eşleme
        # sorunu — öğretmenlerin ÇEŞİTLİLİĞİni silip yalnız ortalamasını öğretiyor) yerine, iki
        # öğretmenin (vE/vL) per-token softmax'ının toplam-varyasyon uzaklığı (0=birebir aynı
        # fikirde, 1=tam zıt) her token için ayrıca saklanır — eğitimde KL kaybını bu ağırlıkla
        # çarpmak, damıtım kapasitesini öğretmenlerin GERÇEKTEN anlaştığı (zaten gereksiz sinyal)
        # değil AYRIŞTIĞI (tamamlayıcı kapsamın olduğu) bölgelere yönlendirir.
        rec["disagree_tags"] = (0.5 * (pa1 - pb1).abs().sum(-1)).cpu().tolist()
        rec["disagree_tags2"] = (0.5 * (pa2 - pb2).abs().sum(-1)).cpu().tolist()

    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"{src_path.name} → {out_path.name}: {n_full} soft-etiketli, {n_trunc} truncation'a "
          f"takıldı (sabit-etiket-yalnız kaldı)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-a", default=str(DATA_DIR / "best_idiom_tagger_vE_leipzigglu.pt"))
    ap.add_argument("--teacher-b", default=str(DATA_DIR / "best_idiom_tagger_vL_idiomcoverage.pt"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model_a, ls = load_teacher(args.teacher_a, device)
    model_b, ls_b = load_teacher(args.teacher_b, device)
    assert ls.tags == ls_b.tags and ls.tags2 == ls_b.tags2, "öğretmenlerin etiket uzayı uyuşmuyor"
    tokenizer = AutoTokenizer.from_pretrained(ls.encoder_model)

    for name in ("tdk_examples.json", "corpus_examples_glu.json"):
        src = DATA_DIR / name
        if not src.exists():
            print(f"atlandı (yok): {src}")
            continue
        out = DATA_DIR / name.replace(".json", "_distill.json")
        soft_label_file(src, out, model_a, model_b, ls, tokenizer, device)


if __name__ == "__main__":
    main()
