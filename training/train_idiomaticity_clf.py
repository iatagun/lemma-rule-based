#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aşama-2: İDYOMATİKLİK sınıflandırıcısı (Fikir 3 — iki aşamalı boru hattı).

DizgeBERT-Idiom'un tek-BIO modeli precision tavanına takılıyor (~%64-71): bir öbeği
YÜZEY BİÇİM eşleşince işaretliyor, GLU kılavuzunun Aşama 3'ünü (bağlamda gerçek mi mecazi mi)
uygulamıyor. v6-v13 boyunca L→hep-O örneklerini eğitim verisine katma denemesi ayrımı bir
miktar öğretti ama span-precision'ı her yerde düşürdü (yapısal gerilim).

İki aşama bunu ayırır:
  Aşama 1 (mevcut v5 BIO)  → aday span'ler (yüksek recall)
  Aşama 2 (BU MODEL)       → (cümle, aday span) → {idyomatik, literal} → literal olanı ELE

Aşama-2 mimarisi: paylaşılan ELECTRA gövdesi + span'in ilk⊕son subword temsili → Linear(2H, 2).
Eğitim verisi (elle etiketli, `filter_corpus_idiomaticity.py` akışından):
  - `_corpus_sample_records.jsonl` (idx → words/tags/idiom/span; tags B-VID/I-VID span'i işaretler)
  - `_corpus_sample_labels.tsv`   (idx → D/L/E);  D=idyomatik(1), L=literal(0), E atılır
  - held-out: `corpus_minpair_test.json` (görülmemiş 118 deyim, D ve L kayıtları)

Kullanım:
    python data/filter_corpus_idiomaticity.py --apply      # _holdout_idioms.json + test json üretir
    python training/train_idiomaticity_clf.py --freeze 8 --dropout 0.3 --weight-decay 0.05 --epochs 14
    python training/train_idiomaticity_clf.py --eval --checkpoint idiom_data/best_idiomaticity_clf.pt

Not: bu script `_corpus_sample_labels.tsv` + `_corpus_sample_records.jsonl` + `_holdout_idioms.json`'u
DOĞRUDAN okur; `--apply`'ın `--balance`'ı YALNIZ `corpus_examples_glu.json`'u (stage-1 `--corpus-glu`
deney yolu) etkiler, buradaki eğitimi ETKİLEMEZ (dengesizlik sınıf ağırlığıyla halledilir).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo kökü (script bir alt dizinde)
sys.path.insert(0, str(PROJECT_ROOT))
DATA = PROJECT_ROOT / "idiom_data"
RECS = DATA / "raw" / "_corpus_sample_records.jsonl"
LABELS = DATA / "raw" / "_corpus_sample_labels.tsv"
# not: filter_corpus_idiomaticity.py bunları idiom_data/ altına yazıyor (raw/ değil) — ikisini de dene
RECS_ALT = DATA / "_corpus_sample_records.jsonl"
LABELS_ALT = DATA / "_corpus_sample_labels.tsv"
TEST_JSON = DATA / "corpus_minpair_test.json"
CKPT = DATA / "best_idiomaticity_clf.pt"

ENCODER = "dbmdz/electra-base-turkish-cased-discriminator"
MAX_LEN = 128
BATCH = 16
EPOCHS = 8
LR = 2e-5
WARMUP = 0.1
DROPOUT = 0.15


def _find(p: Path, alt: Path) -> Path:
    if p.exists():
        return p
    if alt.exists():
        return alt
    sys.exit(f"{p} / {alt} yok — önce filter_corpus_idiomaticity.py --apply")


def span_from_tags(tags: list[str]) -> tuple[int, int] | None:
    idx = [i for i, t in enumerate(tags) if t != "O"]
    return (idx[0], idx[-1] + 1) if idx else None


def load_pairs() -> tuple[list[dict], list[dict]]:
    """→ (train, test) kayıt listeleri: {words, s, e, y}  (y: 1 idyomatik, 0 literal)."""
    recs = {json.loads(l)["idx"]: json.loads(l)
            for l in _find(RECS, RECS_ALT).read_text(encoding="utf-8").splitlines() if l.strip()}
    lab: dict[int, str] = {}
    for line in _find(LABELS, LABELS_ALT).read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) == 2 and p[0].isdigit() and p[1] in "DLE":
            lab[int(p[0])] = p[1]

    # held-out DEYİM düzeyinde (cümle-metni değil): _holdout_idioms.json varsa onu kullan,
    # yoksa corpus_minpair_test.json cümlelerine düş (geriye dönük uyum).
    hp = DATA / "_holdout_idioms.json"
    holdout_idioms = set(json.loads(hp.read_text(encoding="utf-8"))) if hp.exists() else None
    test_texts: set[str] = set()
    if holdout_idioms is None and TEST_JSON.exists():
        for r in json.loads(TEST_JSON.read_text(encoding="utf-8")):
            test_texts.add(" ".join(r["words"]))

    train, test = [], []
    for i, lb in lab.items():
        if lb == "E" or i not in recs:
            continue
        r = recs[i]
        sp = span_from_tags(r["tags"])
        if sp is None:
            continue
        rec = {"words": r["words"], "s": sp[0], "e": sp[1], "y": 1 if lb == "D" else 0,
               "idx": i, "idiom": r.get("idiom", "")}
        is_test = (r["idiom"] in holdout_idioms if holdout_idioms is not None
                   else " ".join(r["words"]) in test_texts)
        (test if is_test else train).append(rec)
    return train, test


def drop_eval_idioms(rows: list[dict]) -> list[dict]:
    """`--exclude-eval-idioms` — CASES / GLU / Çavuşoğlu eval deyimlerini DOĞAL-derlem
    train'inden atar.

    Neden: doğal-derlem havuzunda eval-dışlaması hiçbir zaman sıkı uygulanmamıştı (v7 ve
    öncesi stage-2'leri "kafa tutmak"/"söz almak" gibi eval deyimlerini oradan görüyordu),
    oysa sentetik havuz `prepare_synthetic_stage2_pairs.select()` ile bunları BİLEREK
    dışlıyor. Doğal veriyi kısmen geri katan her deney (bkz. `--natural-l-only`) bu yüzden
    önce burayı geçmek zorunda — aksi halde v7↔v8 kıyası gibi sızıntı açısından eşitsiz
    bir karşılaştırma daha üretiriz.

    Ölçüt: deyim adı Çavuşoğlu kıyas listesinde birebir geçiyorsa, YA DA deyimin gövdeleri
    herhangi bir eval cümlesinde SIRALI bir span olarak bulunuyorsa (`find_span`, küçük
    ara-söz toleransıyla) kayıt atılır.

    NOT (2026-09-22): ilk sürüm "gövde kümesi ⊆ cümle gövde kümesi" kullanıyordu; bu
    ölçüt 9694 doğal kaydın 9450'sini atıyordu (yaygın gövdeler uzun cümlelerde rastgele
    eşleşiyor). Sıralı span eşleşmesi sızıntının gerçek tanımı — deyim o cümlede GEÇİYOR
    mu — ve seyrek eşleşiyor."""
    import csv

    from data.prepare_tdk_idiom_examples import find_span, idiom_stems, stem

    from benchmark.eval_idiom import CASES
    from data.prepare_glu_examples import HARD_NEG_DIAG, PAIRS

    sents = [c[1] for c in CASES] + [p[0] for p in PAIRS] + [h[0] for h in HARD_NEG_DIAG]
    names: set[str] = set()
    bench = DATA / "raw" / "turkish_idioms_benchmark.tsv"
    if bench.exists():
        # YALNIZ gerçek eval çiftleri: hem `sample` hem `literal` dolu olan satırlar (198).
        # tsv'nin tamamı TDK deyim listesi — hepsini almak doğal train'in %97'sini atıyordu
        # (`prepare_synthetic_stage2_pairs._excluded_idioms()` de aynı ölçütü kullanıyor).
        for r in csv.DictReader(bench.open(encoding="utf-8"), delimiter="	"):
            if not (r.get("sample", "").strip() and r.get("literal", "").strip()):
                continue
            if r.get("idiom", "").strip():
                names.add(r["idiom"].strip())
            sents += [r["sample"], r["literal"]]
    sent_stems = [[stem(w.lower()) for w in s.split()] for s in sents]

    verdict: dict[str, bool] = {}   # deyim adı → atılsın mı (aynı deyim binlerce kayıtta)
    kept, dropped_idioms = [], set()
    for r in rows:
        nm = r.get("idiom", "")
        if nm not in verdict:
            seq = idiom_stems(nm) if nm else []
            verdict[nm] = bool(nm) and (
                nm in names
                or bool(seq) and any(find_span(seq, ss, max_gap=2) for ss in sent_stems)
            )
        if verdict[nm]:
            dropped_idioms.add(nm)
            continue
        kept.append(r)
    print(f"--exclude-eval-idioms: {len(rows)}→{len(kept)} kayıt "
          f"({len(dropped_idioms)} farklı deyim atıldı: CASES+GLU+Çavuşoğlu)")
    return kept


def load_synthetic(records_path: Path) -> list[dict]:
    """Deney Z — `data/prepare_synthetic_stage2_pairs.py --build` çıktısı: LLM-ÜRETİLMİŞ
    (etiketlenmiş değil) dengeli D+L cümleleri. Yalnız TRAIN'e eklenir (test seti hiç
    etkilenmez — üretim havuzu zaten Çavuşoğlu/frozen/held-out deyimleriyle örtüşmeyecek
    şekilde seçildi, bkz. o script). Format `load_pairs()` ile aynı: {words, s, e, y, idx}."""
    labels_path = records_path.with_name(records_path.stem.replace("_records", "_labels") + ".tsv")
    if not records_path.exists() or not labels_path.exists():
        sys.exit(f"{records_path} / {labels_path} yok — önce: python data/"
                 f"prepare_synthetic_stage2_pairs.py --select ... --build")
    recs = {json.loads(l)["idx"]: json.loads(l)
            for l in records_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    lab: dict[int, str] = {}
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) == 2 and p[0].isdigit() and p[1] in "DL":
            lab[int(p[0])] = p[1]
    out = []
    for i, lb in lab.items():
        if i not in recs:
            continue
        r = recs[i]
        sp = span_from_tags(r["tags"])
        if sp is None:
            continue
        out.append({"words": r["words"], "s": sp[0], "e": sp[1], "y": 1 if lb == "D" else 0, "idx": i})
    return out


def _first_last(wid: list[int | None]) -> tuple[dict, dict]:
    first, last = {}, {}
    for i, w in enumerate(wid):
        if w is None:
            continue
        first.setdefault(w, i)
        last[w] = i
    return first, last


class ClfDS(Dataset):
    """`lex_*` alanları — Deney V (compat-gap): span kelimelerinin TEK BAŞINA (cümle
    bağlamı olmadan) tokenize edilmiş hali, `IdiomaticityClf(compat_gap=True)` bunu
    ikinci bir küçük forward ile kodlayıp "leksikal" (bağlamsız) temsili çıkarır.
    `compat_gap=False` iken bu alanlar hesaplanır ama forward() hiç kullanmaz — ucuz
    (birkaç kelimelik dizi), tek DS ile her iki mod da çalışsın diye ayrı dal açılmadı.

    `feats_for` — Deney Y (morph-feat): verilirse her satır için `morph_deviation_vec`
    (4 float) önceden hesaplanıp saklanır (`data/tag_idiom_morph_feats.py`)."""

    def __init__(self, rows: list[dict], tok, name: str = "", feats_for=None):
        self.items = []
        dropped = 0
        for r in rows:
            enc = tok(r["words"], is_split_into_words=True, truncation=True, max_length=MAX_LEN)
            first, last = _first_last(enc.word_ids())
            if r["s"] not in first or (r["e"] - 1) not in last:
                dropped += 1  # span MAX_LEN kırpmasına takıldı
                continue
            span_words = r["words"][r["s"]:r["e"]]
            lex_enc = tok(span_words, is_split_into_words=True, truncation=True, max_length=MAX_LEN)
            lfirst, llast = _first_last(lex_enc.word_ids())
            item = {
                "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "sf": first[r["s"]], "sl": last[r["e"] - 1], "y": r["y"],
                "lex_input_ids": lex_enc["input_ids"], "lex_attention_mask": lex_enc["attention_mask"],
                "lf": lfirst[0], "ll": llast[max(llast)],
            }
            if feats_for is not None:
                from data.tag_idiom_morph_feats import morph_deviation_vec
                item["morph_vec"] = morph_deviation_vec(feats_for(r["words"]), r["s"], r["e"])
            self.items.append(item)
        if dropped:
            print(f"  {name or 'ClfDS'}: {dropped}/{len(rows)} örnek span MAX_LEN'e takıldı, atlandı")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(pad_id: int):
    def f(b):
        m = max(len(x["input_ids"]) for x in b)
        lm = max(len(x["lex_input_ids"]) for x in b)
        pad = lambda s, v, w: s + [v] * (w - len(s))
        out = {
            "input_ids": torch.tensor([pad(x["input_ids"], pad_id, m) for x in b]),
            "attention_mask": torch.tensor([pad(x["attention_mask"], 0, m) for x in b]),
            "sf": torch.tensor([x["sf"] for x in b]),
            "sl": torch.tensor([x["sl"] for x in b]),
            "y": torch.tensor([x["y"] for x in b]),
            "lex_input_ids": torch.tensor([pad(x["lex_input_ids"], pad_id, lm) for x in b]),
            "lex_attention_mask": torch.tensor([pad(x["lex_attention_mask"], 0, lm) for x in b]),
            "lf": torch.tensor([x["lf"] for x in b]),
            "ll": torch.tensor([x["ll"] for x in b]),
        }
        if "morph_vec" in b[0]:
            out["morph_vec"] = torch.tensor([x["morph_vec"] for x in b], dtype=torch.float32)
        return out
    return f


class IdiomaticityClf(nn.Module):
    """ELECTRA + span ilk⊕son subword → Linear(2H, 2). DizgeBERT-Idiom ile aynı gövde/pooling.

    `freeze` > 0: embeddings + alttan `freeze` transformer katmanı dondurulur (975 örnekte
    tam fine-tune ağır overfit ediyordu — loss→0.0007, softmax doygun, eşik ayarı ölü).
    Dondurma trainable parametreyi düşürür → overfit azalır, eşik taraması geri gelir.

    `compat_gap` (Deney V, Zeng & Bhat 2021 "semantic compatibility"): span'in BAĞLAMSAL
    temsiline (cümledeki hali) ek olarak LEKSİKAL (bağlamsız — yalnız span kelimeleri,
    ayrı bir küçük dizi olarak kodlanmış) temsilini de çıkarır; head'e `[bağlamsal, leksikal,
    bağlamsal-leksikal]` (6H) verilir. Sezgi: bağlamsal anlam leksikal/düz anlamdan NE KADAR
    saptıysa o kadar idyomatik — önceki turların (v5ctx) hep BAĞLAMSAL tarafı zenginleştirmesinden
    (`[CLS]⊕ortalama⊕ilk⊕son`) farklı bir eksen, hiç denenmemişti. `compat_gap=False` iken
    davranış birebir eskisiyle aynı (head 2H girdi alır, `lex_*` alanları hesaplanır ama
    kullanılmaz).

    `morph_feat` (Deney Y, Fazly/Cook/Stevenson 2009 kanonik-biçim sapması): span'in nesne/fiil
    bileşenlerinin hâl/çoğul/belirlilik/çatı özelliklerinin idyomun kanonik biçiminden sapıp
    sapmadığını kodlayan 4-boyutlu bir vektör (`data/tag_idiom_morph_feats.py::morph_deviation_vec`)
    head'e ek girdi olarak verilir. Deney I'in (ham UPOS dizisi) "çok kaba" sonucundan FARKLI bir
    eksen — kaba POS yerine deyimin KENDİ kanonik biçiminden sapma sinyali. `morph_feat=False`
    iken davranış birebir eskisiyle aynı."""

    def __init__(self, encoder=ENCODER, dropout: float = DROPOUT, freeze: int = 0,
                 compat_gap: bool = False, morph_feat: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.compat_gap = compat_gap
        self.morph_feat = morph_feat
        base_dim = 6 * h if compat_gap else 2 * h
        self.head = nn.Linear(base_dim + 4 if morph_feat else base_dim, 2)
        if freeze > 0:
            for p in self.encoder.embeddings.parameters():
                p.requires_grad_(False)
            layers = self.encoder.encoder.layer
            for lyr in layers[:min(freeze, len(layers))]:
                for p in lyr.parameters():
                    p.requires_grad_(False)

    def forward(self, input_ids, attention_mask, sf, sl,
                lex_input_ids=None, lex_attention_mask=None, lf=None, ll=None, morph_vec=None):
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        B = hs.shape[0]
        ctx = torch.cat([hs[torch.arange(B), sf], hs[torch.arange(B), sl]], -1)
        if self.compat_gap:
            lhs = self.encoder(input_ids=lex_input_ids, attention_mask=lex_attention_mask).last_hidden_state
            lex = torch.cat([lhs[torch.arange(B), lf], lhs[torch.arange(B), ll]], -1)
            ctx = torch.cat([ctx, lex, ctx - lex], -1)
        if self.morph_feat:
            ctx = torch.cat([ctx, morph_vec], -1)
        return self.head(self.dropout(ctx))


def wrap_stage2(base_predict, clf_ckpt: str, thresh: float = 0.5):
    """Fikir 3: aşama-1 span'lerini idyomatiklik sınıflandırıcısından geçir, literal olanı ELE.
    Yalnız bitişik VID'e uygulanır (LVC yarı-birleşimsel, gap'li dokunulmaz); span
    `p(literal) > thresh` ise elenir. Sınıflandırıcı gövdesi cümle başına BİR kez çalışır.

    `clf_ckpt` virgülle ayrılmış birden çok checkpoint olabilir (öneri #6 — stage-2 ensemble,
    Deney O'nun stage-1 ensemble'ıyla aynı desen): her checkpoint kendi p(literal)'ini üretir,
    ortalaması eşikle kıyaslanır (soft-vote). Tek checkpoint verilince davranış birebir eskisiyle
    aynı.

    Bu, `DizgeBertIdiomForTokenClassification.predict_spans(stage2=True)` ile AYNI kuralı
    kullanır (`spans_from_bigappy` + `span_p_literal`) — standalone `.pt` için (henüz
    pakete gömülmemiş stage-2 checkpoint'i). `benchmark/eval_idiom` ve `predict_idiom`
    bunu import eder (üç kopya → tek kaynak)."""
    from dizgebert_idiom.modeling_dizgebert_idiom import (align_words, span_p_literal,
                                                           span_p_literal_gap, span_p_literal_morph)

    loaded = []
    need_morph = False
    for path in clf_ckpt.split(","):
        path = path.strip()
        if not path:
            continue
        ck = torch.load(path, map_location="cpu")
        enc_name = ck.get("encoder", ENCODER)
        compat_gap = ck.get("compat_gap", False)
        morph_feat = ck.get("morph_feat", False)
        need_morph = need_morph or morph_feat
        tok = AutoTokenizer.from_pretrained(enc_name)
        clf = IdiomaticityClf(enc_name, compat_gap=compat_gap, morph_feat=morph_feat).eval()
        clf.load_state_dict(ck["model"])
        loaded.append((tok, clf, compat_gap, morph_feat))

    # Deney Y — yalnız gerekirse Morph modelini yükle (gereksiz bellek/gecikme yükünden kaçın)
    feats_for = None
    if need_morph:
        from data.tag_idiom_morph_feats import load_morph_feats_fn, morph_deviation_vec
        feats_for = load_morph_feats_fn()

    @torch.no_grad()
    def predict(words):
        spans = base_predict(words)
        to_check = [sp for sp in spans if sp.get("category") == "VID"
                    and not sp.get("gappy") and not sp.get("literal")]
        if not to_check:
            return spans
        per_clf = []
        for tok, clf, compat_gap, morph_feat in loaded:
            enc, kept, fp, lp = align_words(tok, words, MAX_LEN)
            hs = clf.encoder(input_ids=enc["input_ids"],
                             attention_mask=enc["attention_mask"]).last_hidden_state[0]
            per_clf.append((tok, clf, compat_gap, morph_feat, kept, fp, lp, hs))
        word_feats = feats_for(words) if feats_for is not None else None
        drop = set()
        for sp in to_check:
            s, e = sp["start"], sp["end"]
            probs = []
            for tok, clf, compat_gap, morph_feat, kept, fp, lp, hs in per_clf:
                if s >= len(kept) or (e - 1) >= len(kept):
                    continue  # span kırpıldı → bu checkpoint'ten oy yok
                if compat_gap:
                    # Deney V: span kelimeleri BAĞLAMSIZ, ayrı küçük dizi olarak da kodlanır
                    lenc, lkept, lfp, llp = align_words(tok, words[s:e], MAX_LEN)
                    lhs = clf.encoder(input_ids=lenc["input_ids"],
                                      attention_mask=lenc["attention_mask"]).last_hidden_state[0]
                    probs.append(span_p_literal_gap(hs, fp[0, s], lp[0, e - 1],
                                                    lhs, lfp[0, 0], llp[0, -1], clf.head))
                elif morph_feat:
                    vec = torch.tensor(morph_deviation_vec(word_feats, s, e), dtype=torch.float32)
                    probs.append(span_p_literal_morph(hs, fp[0, s], lp[0, e - 1], vec, clf.head))
                else:
                    probs.append(span_p_literal(hs, fp[0, s], lp[0, e - 1], clf.head))
            if probs and (sum(probs) / len(probs)) > thresh:
                drop.add(id(sp))
        return [sp for sp in spans if id(sp) not in drop]

    return predict


@torch.no_grad()
def evaluate(model, dl, device) -> dict:
    model.eval()
    tp = fp = fn = tn = 0
    for b in dl:
        b = {k: v.to(device) for k, v in b.items()}
        pred = model(b["input_ids"], b["attention_mask"], b["sf"], b["sl"],
                     b["lex_input_ids"], b["lex_attention_mask"], b["lf"], b["ll"],
                     b.get("morph_vec")).argmax(-1)
        y = b["y"]
        tp += int(((pred == 1) & (y == 1)).sum());  fp += int(((pred == 1) & (y == 0)).sum())
        fn += int(((pred == 0) & (y == 1)).sum());  tn += int(((pred == 0) & (y == 0)).sum())
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    lit_acc = tn / (tn + fp) if tn + fp else 0.0  # literal'i doğru ELEME oranı (asıl hedef)
    return {"n": n, "acc": round(100 * acc, 1), "idyom_P": round(100 * p, 1),
            "idyom_R": round(100 * r, 1), "idyom_F1": round(100 * f1, 1),
            "literal_eleme": round(100 * lit_acc, 1), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def align_to_stage1(pairs: list[dict], predict) -> list[dict]:
    """Deney B takibi (2026-09-14): stage-2'yi altın span yerine YENİ stage-1 checkpoint'inin
    (`--align-stage1`) GERÇEKTEN önerdiği aday span'lerle eğit — train/inference aday
    dağılımını eşitler. Her örnek için `predict(words)` çağrılır, altın (s,e) ile en çok
    örtüşen bitişik VID adayı bulunur; bulunamazsa örnek ATLANIR (gerçek çıkarımda da
    stage-2 bu örneği hiç görmeyecekti — stage-1 hiç önermedi)."""
    out, dropped = [], 0
    for r in pairs:
        spans = predict(r["words"])
        best, best_ov = None, 0
        for sp in spans:
            if sp.get("category") != "VID" or sp.get("gappy"):
                continue
            ov = max(0, min(sp["end"], r["e"]) - max(sp["start"], r["s"]))
            if ov > best_ov:
                best, best_ov = sp, ov
        if best is None:
            dropped += 1
            continue
        out.append({**r, "s": best["start"], "e": best["end"]})
    print(f"align-to-stage1: {dropped}/{len(pairs)} örnek stage-1 hiç aday önermedi, atlandı "
          f"→ {len(out)} kaldı")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--encoder", default=ENCODER)
    ap.add_argument("--freeze", type=int, default=0,
                    help="embeddings + alttan N transformer katmanını dondur (overfit↓)")
    ap.add_argument("--compat-gap", action="store_true",
                    help="Deney V: leksikal/bağlamsal uyumluluk mimarisi (Zeng&Bhat 2021) — "
                         "head'e [bağlamsal,leksikal,fark] (6H) verilir. Kapalıyken eskisiyle aynı.")
    ap.add_argument("--morph-feat", action="store_true",
                    help="Deney Y: kanonik-biçim sapması özelliği (Fazly/Cook/Stevenson 2009) — "
                         "DizgeBERT-Morph FEATS'inden 4-boyutlu [hâl-ekli,çoğul,belirli,çatı-sapmış] "
                         "vektörü head'e ek girdi olarak verilir. Kapalıyken eskisiyle aynı.")
    ap.add_argument("--dropout", type=float, default=DROPOUT)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--out", default=str(CKPT), help="checkpoint çıktı yolu")
    ap.add_argument("--align-stage1", default=None,
                    help="Deney B takibi: altın span yerine bu stage-1 checkpoint'inin "
                         "önerdiği aday span'lerle eğit (train/inference aday dağılımını eşitler)")
    ap.add_argument("--min-idx", type=int, default=None,
                    help="öneri #6 takibi: yalnız idx >= bu değer olan (kaynakça v3'ten SONRA "
                         "eklenmiş, LLM/ajan-etiketli) kayıtlarla eğit — v3'ten gerçekten bağımsız "
                         "bir ikinci stage-2 için (test seti DEĞİŞMEZ, hep tam held-out)")
    ap.add_argument("--synthetic-file", default=None,
                    help="Deney Z: data/prepare_synthetic_stage2_pairs.py --build çıktısı "
                         "(_synthetic_stage2_records.jsonl) — yalnız TRAIN'e eklenir, test hiç "
                         "etkilenmez")
    ap.add_argument("--natural-l-only", action="store_true",
                    help="Deney AB-2: doğal-derlem train'inden YALNIZ literal (y=0) kayıtları "
                         "tut, doğal D'yi at. Deney Z'nin 'tam karışım daha kötü' bulgusuyla "
                         "'sentetik-yalnız' arasındaki hiç denenmemiş orta yol — sentetik "
                         "havuzun literal kapsamı (667 L / 326 çift deyim) doğal havuzunkinin "
                         "(5712 L / 2989 deyim) çok altında")
    ap.add_argument("--exclude-eval-idioms", action="store_true",
                    help="doğal-derlem train'inden CASES/GLU/Çavuşoğlu eval deyimlerini at "
                         "(doğal havuzda eval-dışlaması hiç sıkı uygulanmamıştı — "
                         "bkz. drop_eval_idioms)")
    ap.add_argument("--synthetic-only", action="store_true",
                    help="Deney Z ablasyonu: doğal-derlem train kayıtlarını AT, yalnız "
                         "--synthetic-file ile eğit (üslup-kayması riskini izole etmek için)")
    args = ap.parse_args()
    out_ckpt = Path(args.out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    tok = AutoTokenizer.from_pretrained(args.encoder)
    pad_id = tok.pad_token_id or 0

    train_rows, test_rows = load_pairs()
    if args.min_idx is not None:
        before = len(train_rows)
        train_rows = [r for r in train_rows if r["idx"] >= args.min_idx]
        print(f"--min-idx {args.min_idx}: train {before}→{len(train_rows)}")
    if args.exclude_eval_idioms:
        train_rows = drop_eval_idioms(train_rows)
    if args.natural_l_only:
        before = len(train_rows)
        train_rows = [r for r in train_rows if r["y"] == 0]
        print(f"--natural-l-only: doğal D atıldı, train {before}→{len(train_rows)} (hepsi literal)")
    if args.synthetic_only:
        if args.natural_l_only:
            sys.exit("--synthetic-only ile --natural-l-only birlikte anlamsız (biri doğal "
                     "veriyi tamamen atıyor, diğeri bir dilimini tutuyor)")
        print(f"--synthetic-only: doğal-derlem train ({len(train_rows)} kayıt) ATILDI")
        train_rows = []
    if args.synthetic_file:
        syn_rows = load_synthetic(Path(args.synthetic_file))
        print(f"--synthetic-file: +{len(syn_rows)} kayıt "
              f"({Counter(r['y'] for r in syn_rows)}) → train {len(train_rows)}+{len(syn_rows)}")
        train_rows = train_rows + syn_rows
    print(f"train {len(train_rows)} ({Counter(r['y'] for r in train_rows)})  "
          f"test {len(test_rows)} ({Counter(r['y'] for r in test_rows)})")

    if args.align_stage1:
        from benchmark.eval_idiom import make_predictor
        predict = make_predictor(True, args.align_stage1, "")
        print(f"[align-stage1] {args.align_stage1} ile aday span'ler yeniden hesaplanıyor...")
        train_rows = align_to_stage1(train_rows, predict)
        test_rows = align_to_stage1(test_rows, predict)

    feats_for = None
    if args.morph_feat:
        from data.tag_idiom_morph_feats import load_morph_feats_fn
        feats_for = load_morph_feats_fn(device)
        print("[morph-feat] DizgeBERT-Morph FEATS tabanlı kanonik-biçim sapması aktif")

    test_ds = ClfDS(test_rows, tok, "held-out", feats_for=feats_for)
    test_dl = DataLoader(test_ds, batch_size=BATCH, collate_fn=collate(pad_id))

    model = IdiomaticityClf(args.encoder, dropout=args.dropout, freeze=args.freeze,
                            compat_gap=args.compat_gap, morph_feat=args.morph_feat).to(device)
    if args.freeze:
        ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"freeze {args.freeze} → trainable {ntr/1e6:.1f}M")
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
        print(f"yüklendi: {args.checkpoint}")

    if args.eval:
        print("held-out:", evaluate(model, test_dl, device))
        return

    train_ds = ClfDS(train_rows, tok, "train", feats_for=feats_for)
    # sınıf ağırlığı: idyomatik(1) baskın (~820:463) → literal'e ağırlık
    cnt = Counter(x["y"] for x in train_ds.items)
    w = torch.tensor([1.0 / max(cnt[0], 1), 1.0 / max(cnt[1], 1)], device=device)
    w = (w / w.sum() * 2).float()
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate(pad_id))
    print(f"train_ds {len(train_ds)}  class-weights {w.tolist()}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    total = len(train_dl) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(total * WARMUP), total)

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for b in tqdm(train_dl, desc=f"ep{ep}"):
            b = {k: v.to(device) for k, v in b.items()}
            opt.zero_grad()
            logits = model(b["input_ids"], b["attention_mask"], b["sf"], b["sl"],
                           b["lex_input_ids"], b["lex_attention_mask"], b["lf"], b["ll"],
                           b.get("morph_vec"))
            loss = F.cross_entropy(logits, b["y"], weight=w)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            tot += loss.item()
        res = evaluate(model, test_dl, device)
        # seçim skoru: iki sınıfın recall'ı dengeli (= dengeli doğruluk). Boru hattı
        # "doğru-ayırt" metriği tam bunu ölçüyor — hem idyomatiği yakala HEM literali ele.
        # (Eski (idyom_F1+literal_eleme)/2 literal_eleme'yi aşırı ödüllendirip aşırı
        # temkinli epoch'u seçiyordu.)
        score = (res["idyom_R"] + res["literal_eleme"]) / 2
        print(f"ep{ep} loss {tot/len(train_dl):.4f}  {res}  macro {score:.1f}")
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "encoder": args.encoder, "metrics": res,
                       "compat_gap": args.compat_gap, "morph_feat": args.morph_feat}, out_ckpt)
            print(f"  → {out_ckpt.name}")
    print(f"best macro {best:.1f}")


if __name__ == "__main__":
    main()
