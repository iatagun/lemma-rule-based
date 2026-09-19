#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Öneri #1 — ensemble distilasyonu: N-öğretmenli ensemble'ın bilgisini TEK gövdeli bir
öğrenciye aktarmak için, weak-supervision eğitim kayıtlarına (tdk_examples.json,
corpus_examples_glu.json) öğretmenlerin ORTALAMA per-kelime softmax'ını
("soft_tags"/"soft_tags2") ekler. PARSEME altın verisine (train.json) DOKUNULMAZ — yalnız
zaten zayıf-etiketli kaynaklar.

Truncation'a takılan (MAX_LEN'i aşan) kayıtlar soft_tags'siz bırakılır (has_soft=False,
IdiomDataset'te sabit-etiket-yalnız olarak eğitilir) — hizalama karmaşıklığından kaçınmak için.

Deney X (2026-09-19): 2 öğretmenden N öğretmene genelleştirildi. Anlaşmazlık ağırlığı artık
tüm öğretmen ÇİFTLERİ arasındaki ortalama toplam-varyasyon uzaklığı (N=2 için eski 0.5*|pa-pb|
formülüyle birebir aynı sonucu verir — geriye uyumlu).

Kullanım:
    python scripts/distill_ensemble_labels.py \\
        --teachers idiom_data/best_idiom_tagger_vE_leipzigglu.pt,idiom_data/best_idiom_tagger_vL_idiomcoverage.pt,idiom_data/best_idiom_tagger_vX3_slice3.pt
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


def _mean_pairwise_tv(probs: list[torch.Tensor]) -> torch.Tensor:
    """Öğretmen çiftlerinin toplam-varyasyon uzaklığının ortalaması (N=2 için eski
    0.5*|pa-pb| formülüyle birebir aynı)."""
    n = len(probs)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    tv = sum(0.5 * (probs[i] - probs[j]).abs().sum(-1) for i, j in pairs)
    return tv / len(pairs)


@torch.no_grad()
def soft_label_file(src_path: Path, out_path: Path, models: list, ls, tokenizer, device):
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

        outs = [m(input_ids, attn, fp, lp, pos_ids) for m in models]
        p1s = [F.softmax(o["tags"][0], dim=-1) for o in outs]
        p2s = [F.softmax(o["tags2"][0], dim=-1) for o in outs]
        p1 = sum(p1s) / len(p1s)
        p2 = sum(p2s) / len(p2s)
        rec["soft_tags"] = p1.cpu().tolist()
        rec["soft_tags2"] = p2.cpu().tolist()
        # Deney W — anlaşmazlık-ağırlıklı distilasyon: damıtımı TÜM kayıtlara eşit uygulamak
        # ("Agree to Disagree", NeurIPS 2020'nin teşhis ettiği ortalama-eşleme sorunu —
        # öğretmenlerin ÇEŞİTLİLİĞİni silip yalnız ortalamasını öğretiyor) yerine, öğretmen
        # ÇİFTLERİNİN per-token softmax'ının ortalama toplam-varyasyon uzaklığı (0=hepsi aynı
        # fikirde, 1=tam zıt) her token için ayrıca saklanır — eğitimde KL kaybını bu ağırlıkla
        # çarpmak, damıtım kapasitesini öğretmenlerin GERÇEKTEN anlaştığı (zaten gereksiz sinyal)
        # değil AYRIŞTIĞI (tamamlayıcı kapsamın olduğu) bölgelere yönlendirir. Deney X (2026-09-19):
        # N=3 öğretmene genelleştirildi (ortalama ikili TV uzaklığı).
        rec["disagree_tags"] = _mean_pairwise_tv(p1s).cpu().tolist()
        rec["disagree_tags2"] = _mean_pairwise_tv(p2s).cpu().tolist()

    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"{src_path.name} → {out_path.name}: {n_full} soft-etiketli, {n_trunc} truncation'a "
          f"takıldı (sabit-etiket-yalnız kaldı)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", default=",".join([
        str(DATA_DIR / "best_idiom_tagger_vE_leipzigglu.pt"),
        str(DATA_DIR / "best_idiom_tagger_vL_idiomcoverage.pt"),
    ]), help="virgülle ayrılmış 2+ öğretmen checkpoint'i")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    ck_paths = [c.strip() for c in args.teachers.split(",") if c.strip()]
    assert len(ck_paths) >= 2, "--teachers en az 2 checkpoint gerektirir"
    loaded = [load_teacher(p, device) for p in ck_paths]
    models = [m for m, _ in loaded]
    ls = loaded[0][1]
    for _, ls_i in loaded[1:]:
        assert ls.tags == ls_i.tags and ls.tags2 == ls_i.tags2, "öğretmenlerin etiket uzayı uyuşmuyor"
    print(f"öğretmenler ({len(models)}): {ck_paths}")
    tokenizer = AutoTokenizer.from_pretrained(ls.encoder_model)

    for name in ("tdk_examples.json", "corpus_examples_glu.json"):
        src = DATA_DIR / name
        if not src.exists():
            print(f"atlandı (yok): {src}")
            continue
        out = DATA_DIR / name.replace(".json", "_distill.json")
        soft_label_file(src, out, models, ls, tokenizer, device)


if __name__ == "__main__":
    main()
