#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Öneri #2 — ortak-gövde çok-görevli eğitim (multi-task, hiç denenmemiş eksen).

Şu ana kadar stage-1 (BIO span) ve stage-2 (idiomatiklik sınıflandırma) TAMAMEN AYRI iki
ELECTRA gövdesi taşıyordu (~880MB-1.3GB paket). Bu script TEK paylaşılan gövde üzerine hem
BIO tagging head'lerini (`tag_head`/`tag_head2`, v5'ten değişmedi) hem idiomatiklik head'ini
(`stage2_head`, `IdiomaticityClf` ile aynı mimari: span ilk⊕son subword → Linear(2H,2))
kurup İKİ görevi AYNI ADIMDA, PAYLAŞILAN gradyanla eğitir — ne ensemble (Deney O, tahminleri
birleştirme) ne distilasyon (Deney R, bir modelin çıktısını diğerine öğretme), üçüncü bir
mekanizma: iki gözetim sinyali aynı temsili birlikte şekillendiriyor.

Veri: stage-1 tarafı vL'nin BİREBİR aynı reçetesi (train.json + tdk_examples.json +
corpus_examples_glu.json). stage-2 tarafı `train_idiomaticity_clf.py::load_pairs()`'ın
AYNI frozen havuzu (PARSEME altın ile hiç karışmaz, ayrı bir DataLoader). Her adımda stage-1
batch'i + cycle edilmiş bir stage-2 batch'i AYNI şekilde ileri geçirilir, kayıplar toplanır
(loss1 + mu*loss2), TEK backward/optimizer adımı.

Değerlendirme kolaylığı için (mevcut eval altyapısını DEĞİŞTİRMEDEN kullanmak amacıyla) en iyi
epoch'ta paylaşılan gövde iki AYRI, mevcut format uyumlu checkpoint'e "bölünür":
  - `best_idiom_tagger_vJoint.pt`      → düz `IdiomTagger` state_dict (stage2_head hariç),
                                          `benchmark/eval_idiom.py --checkpoint` ile uyumlu.
  - `best_idiomaticity_clf_vJoint.pt`  → `IdiomaticityClf`-uyumlu state_dict (aynı encoder
                                          ağırlıkları + stage2_head→head), `--stage2` ile uyumlu.
Bu yalnız DEĞERLENDİRME kolaylığı — eğitim GERÇEKTEN paylaşılan tek gövdeyle yapılıyor.

Kullanım:
    python scripts/train_joint_stage2.py --epochs 10 --mu 1.0
    python benchmark/eval_idiom.py --local --checkpoint idiom_data/best_idiom_tagger_vJoint.pt \\
        --stage2 idiom_data/best_idiomaticity_clf_vJoint.pt --gap 2 --mode external
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from training.train_idiom_bert import (
    DATA_DIR, ENCODER_MODEL, IdiomDataset, IdiomLabelSpace, IdiomTagger,
    build_class_weights, build_class_weights2, compute_loss, evaluate,
    make_collate, print_eval, selection_score,
)
from training.train_idiomaticity_clf import ClfDS, load_pairs
from training.train_idiomaticity_clf import collate as clf_collate

WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
BATCH_SIZE = 16
LR = 2e-5


class JointIdiomTagger(IdiomTagger):
    """`IdiomTagger` + ikinci bir head (`stage2_head`) AYNI `self.encoder`'ı paylaşır.
    `forward()` DEĞİŞMEDİ (stage-1 davranışı birebir aynı, tek başına yüklenebilir — bkz.
    dosya-başı docstring'teki checkpoint bölme). `forward_stage2` span ilk⊕son subword →
    idiomatiklik logiti, `IdiomaticityClf.forward` ile AYNI hesap, gövde PAYLAŞILAN."""

    def __init__(self, ls: IdiomLabelSpace, encoder_model: str = ENCODER_MODEL,
                stage2_dropout: float = 0.15):
        super().__init__(ls, encoder_model)
        h = self.encoder.config.hidden_size
        self.stage2_head = nn.Linear(2 * h, 2)
        self.stage2_dropout_layer = nn.Dropout(stage2_dropout)

    def forward_stage2(self, input_ids, attention_mask, sf, sl):
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        B = hs.shape[0]
        f = hs[torch.arange(B, device=hs.device), sf]
        g = hs[torch.arange(B, device=hs.device), sl]
        return self.stage2_head(self.stage2_dropout_layer(torch.cat([f, g], -1)))


def split_checkpoints(model: JointIdiomTagger, ls: IdiomLabelSpace, meta: dict,
                       suffix: str = "vJoint") -> None:
    """En iyi epoch'ta paylaşılan gövdeyi mevcut eval altyapısıyla uyumlu iki dosyaya böler
    (dosya-başı docstring). Yalnız değerlendirme kolaylığı, eğitimi etkilemez."""
    # GPU tensörlerini TEK seferde CPU'ya kopyala — aksi halde iki ayrı torch.save() çağrısı
    # encoder ağırlıklarını (paylaşılan, iki dosyada da var) GPU'dan İKİ KEZ D2H kopyalar,
    # bu da 4GB'lık kartta allocator fragmentasyonuna yol açıp sonraki epoch'u yavaşlatıyordu.
    full = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    tagger_sd = {k: v for k, v in full.items()
                if not k.startswith("stage2_head") and not k.startswith("stage2_dropout_layer")}
    torch.save({**meta, "model": tagger_sd, "label_space": ls.as_dict(),
                "encoder_model": ls.encoder_model},
               DATA_DIR / f"best_idiom_tagger_{suffix}.pt")

    clf_sd = {k: v for k, v in full.items() if k.startswith("encoder.")}
    clf_sd["head.weight"] = full["stage2_head.weight"]
    clf_sd["head.bias"] = full["stage2_head.bias"]
    torch.save({"model": clf_sd, "encoder": ls.encoder_model, "metrics": meta.get("metrics")},
               DATA_DIR / f"best_idiomaticity_clf_{suffix}.pt")


def pcgrad_step(model: JointIdiomTagger, loss1: torch.Tensor, loss2: torch.Tensor) -> bool:
    """Deney T — PCGrad (Yu et al., NeurIPS 2020) gradyan cerrahisi, yalnız PAYLAŞILAN
    `encoder` parametrelerinde. Deney S'in kök nedeni buydu: ortak gövde korumasız paylaşıldığı
    için stage-1'in çok daha büyük hacimli gradyanı stage-2'nin ince sinyalini boğuyordu.
    Head'ler (`tag_head`/`tag_head2` ↔ loss1, `stage2_head` ↔ loss2) zaten göreve özel,
    çakışma yalnız paylaşılan gövdede olabilir — orada iki görev gradyanının kosinüsü negatifse
    (çakışıyorsa) her birinin diğerine çakışan bileşeni silinir, DEĞİLSE davranış toplam
    (loss1+loss2) ile birebir aynıdır. `--pcgrad` kapalıyken bu fonksiyon hiç çağrılmaz."""
    shared = [p for p in model.encoder.parameters() if p.requires_grad]
    task1_only = list(model.tag_head.parameters()) + list(model.tag_head2.parameters())
    if hasattr(model, "pos_embed"):
        task1_only += list(model.pos_embed.parameters())
    task2_only = list(model.stage2_head.parameters())

    g1 = torch.autograd.grad(loss1, shared, retain_graph=True, allow_unused=True)
    g2 = torch.autograd.grad(loss2, shared, retain_graph=True, allow_unused=True)
    g1 = [torch.zeros_like(p) if g is None else g for g, p in zip(g1, shared)]
    g2 = [torch.zeros_like(p) if g is None else g for g, p in zip(g2, shared)]
    flat1 = torch.cat([g.reshape(-1) for g in g1])
    flat2 = torch.cat([g.reshape(-1) for g in g2])
    dot = torch.dot(flat1, flat2)
    conflict = bool(dot.item() < 0)
    if conflict:
        flat1_orig, flat2_orig = flat1, flat2
        flat1 = flat1_orig - dot / (flat2_orig.norm() ** 2 + 1e-12) * flat2_orig
        flat2 = flat2_orig - dot / (flat1_orig.norm() ** 2 + 1e-12) * flat1_orig
    combined = flat1 + flat2
    offset = 0
    for p, g in zip(shared, g1):
        n = g.numel()
        p.grad = combined[offset:offset + n].view_as(g).clone()
        offset += n

    loss1.backward(inputs=task1_only, retain_graph=False)
    loss2.backward(inputs=task2_only, retain_graph=False)
    return conflict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--mu", type=float, default=1.0, help="stage-2 kaybının ağırlığı (loss1 + mu*loss2)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--pcgrad", action="store_true",
                     help="Deney T: paylaşılan encoder'da PCGrad gradyan cerrahisi (kapalıyken Deney S ile birebir aynı davranış)")
    ap.add_argument("--freeze", type=int, default=0,
                     help="Deney U: v3/IdiomaticityClf ile AYNI reçete — embeddings + alttan N "
                          "transformer katmanını dondur (paylaşılan gövdede, yalnız üst katmanlar "
                          "+ head'ler eğitilir). 0 = kapalı, Deney S/T ile birebir aynı davranış.")
    args = ap.parse_args()
    suffix_parts = ["vJoint"]
    if args.pcgrad: suffix_parts.append("PC")
    if args.freeze: suffix_parts.append(f"F{args.freeze}")
    suffix = "".join(suffix_parts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  mu={args.mu}  pcgrad={args.pcgrad}  freeze={args.freeze}")

    ls = IdiomLabelSpace.load()
    tokenizer = AutoTokenizer.from_pretrained(ls.encoder_model)
    pad_id = tokenizer.pad_token_id or 0
    collate1 = make_collate(pad_id)

    # ── stage-1 veri: vL'nin BİREBİR aynı reçetesi ──────────────────────────
    train_ds = IdiomDataset(DATA_DIR / "train.json", tokenizer, ls)
    dev_ds = IdiomDataset(DATA_DIR / "dev.json", tokenizer, ls)
    weight_sources = [DATA_DIR / "train.json"]
    tdk_ds = IdiomDataset(DATA_DIR / "tdk_examples.json", tokenizer, ls)
    train_ds = torch.utils.data.ConcatDataset([train_ds, tdk_ds])
    weight_sources += [DATA_DIR / "tdk_examples.json"]
    cg_ds = IdiomDataset(DATA_DIR / "corpus_examples_glu.json", tokenizer, ls)
    train_ds = torch.utils.data.ConcatDataset([train_ds, cg_ds])
    weight_sources += [DATA_DIR / "corpus_examples_glu.json"]
    print(f"stage-1 train {len(train_ds)}  dev {len(dev_ds)}")

    weights = build_class_weights(weight_sources, ls, device)
    weights2 = build_class_weights2(weight_sources, ls, device)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate1)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, collate_fn=collate1)

    # ── stage-2 veri: train_idiomaticity_clf.py'nin AYNI frozen havuzu ──────
    s2_train_rows, _s2_test_rows = load_pairs()
    s2_train_ds = ClfDS(s2_train_rows, tokenizer, "stage2-train")
    s2_cnt = Counter(x["y"] for x in s2_train_ds.items)
    s2_w = torch.tensor([1.0 / max(s2_cnt[0], 1), 1.0 / max(s2_cnt[1], 1)], device=device)
    s2_w = (s2_w / s2_w.sum() * 2).float()
    s2_dl = DataLoader(s2_train_ds, batch_size=args.batch_size, shuffle=True,
                       collate_fn=clf_collate(pad_id))
    print(f"stage-2 train {len(s2_train_ds)}  class-weights {s2_w.tolist()}")

    model = JointIdiomTagger(ls, ls.encoder_model).to(device)
    if args.freeze > 0:
        # v3/IdiomaticityClf ile BİREBİR aynı dondurma (training/train_idiomaticity_clf.py) —
        # PAYLAŞILAN gövdede uygulanıyor, bu yüzden stage-1'in tag_head/tag_head2'sini de korur.
        for p in model.encoder.embeddings.parameters():
            p.requires_grad_(False)
        layers = model.encoder.encoder.layer
        for lyr in layers[:min(args.freeze, len(layers))]:
            for p in lyr.parameters():
                p.requires_grad_(False)
        ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"freeze {args.freeze} → trainable {ntr/1e6:.1f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_RATIO), total_steps)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        model.train()
        s2_iter = itertools.cycle(s2_dl)
        tot1 = tot2 = 0.0
        conflicts = 0
        for batch1 in tqdm(train_dl, desc="train"):
            batch1 = {k: v.to(device) for k, v in batch1.items()}
            batch2 = {k: v.to(device) for k, v in next(s2_iter).items()}
            optimizer.zero_grad()
            logits1 = model(batch1["input_ids"], batch1["attention_mask"],
                            batch1["first_pos"], batch1["last_pos"], batch1["pos_ids"])
            loss1 = compute_loss(logits1, batch1, weights, weights2)
            logits2 = model.forward_stage2(batch2["input_ids"], batch2["attention_mask"],
                                           batch2["sf"], batch2["sl"])
            loss2 = F.cross_entropy(logits2, batch2["y"], weight=s2_w)
            if args.pcgrad:
                conflicts += int(pcgrad_step(model, loss1, loss2 * args.mu))
            else:
                (loss1 + args.mu * loss2).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tot1 += loss1.item(); tot2 += loss2.item()
        n = len(train_dl)
        conflict_note = f"  pcgrad_conflict_rate {conflicts/n:.2%}" if args.pcgrad else ""
        print(f"train loss1(stage-1) {tot1/n:.4f}  loss2(stage-2) {tot2/n:.4f}{conflict_note}")

        # not: empty_cache() BİLEREK yok — PyTorch'un caching allocator'ı eval/train geçişini
        # kendi yönetir; 4GB'lık kartta cache'i zorla boşaltmak allocator'ı sıfırdan (daha
        # parçalı) yeniden ısınmaya zorluyor, bu da bir sonraki epoch'ta ~3x kalıcı yavaşlama
        # yaratıyordu (Deney T, 2026-09-18) — checkpoint kaydını CPU'ya taşımak bunu ÇÖZMEDİ,
        # asıl sebep bu çağrının kendisiydi.
        res = evaluate(model, dev_dl, device, ls)
        print_eval(res)
        score = selection_score(res)
        print(f"\n  selection score (span F1, ALL): {score:.2f}")
        if score > best:
            best = score
            split_checkpoints(model, ls, {"epoch": epoch, "target_epochs": args.epochs,
                                          "best": best, "metrics": res}, suffix=suffix)
            print(f"  → best kaydedildi ({suffix} çifti)")

    print(f"\nBest selection score: {best:.2f}")


if __name__ == "__main__":
    main()
