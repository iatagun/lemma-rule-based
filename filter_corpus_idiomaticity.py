#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derlemden madenlenen deyim örneklerini İDYOMATİKLİK için LLM ile filtreler (Fikir 2).

`prepare_tdk_corpus_examples.py` sıkı yüzey-biçim eşleşmesiyle örnek üretir ama GLU
kılavuzunun Aşama 3'ünü (bağlamda gerçek mi mecazi mi) uygulamaz → ~%20-25 literal/terim
yanlış-pozitif (v8'in precision'ını çökerten sebep). Bu script her örneğin CÜMLEDEKİ
kullanımını bir LLM'e GLU §5 rubric'iyle sınıflandırtır ve yalnız GERÇEKTEN İDYOMATİK
olanları tutar.

Sınıflar:  D = deyim (mecazi/aktarılmış) · L = literal (birebir/gerçek anlam) · E = deyim
değil (sıradan eşdizim / terim / belirsiz).  Yalnız D tutulur.

LLM: OpenAI-uyumlu endpoint (varsayılan LM Studio http://localhost:1234/v1). Resumable —
etiketler `_corpus_idiomaticity_labels.jsonl`'e satır satır yazılır, tekrar çalıştırınca
kaldığı yerden.

Kullanım:
    # (LM Studio'da bir model yükle, sunucuyu başlat)
    python filter_corpus_idiomaticity.py --max-per-idiom 3 --max-total 6000
    # farklı endpoint:
    python filter_corpus_idiomaticity.py --base-url https://api.openai.com/v1 --model gpt-4o-mini
    # sonra:
    python train_idiom_bert.py --class-weights --tdk-examples --corpus-glu --epochs 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
IN_JSON = PROJECT_ROOT / "idiom_data" / "corpus_examples.json"
LABELS = PROJECT_ROOT / "idiom_data" / "_corpus_idiomaticity_labels.jsonl"
OUT_JSON = PROJECT_ROOT / "idiom_data" / "corpus_examples_glu.json"
SAMPLE_TSV = PROJECT_ROOT / "idiom_data" / "_corpus_sample.tsv"
MANUAL_LABELS = PROJECT_ROOT / "idiom_data" / "_corpus_sample_labels.tsv"
SAMPLE_RECS = PROJECT_ROOT / "idiom_data" / "_corpus_sample_records.jsonl"  # idx→tam kayıt (apply bunu okur)

SYS_PROMPT = (
    "Sen Türkçe deyim/söz varlığı uzmanısın. GLU (Gülsün Leylâ Uzun) deyim etiketleme "
    "kılavuzunun ölçütlerini uyguluyorsun. Yalnızca istenen biçimde, kısa yanıt ver."
)

TASK = """Aşağıda numaralı Türkçe cümleler var; her birinde bir SÖZ ÖBEĞİ işaretli. Her öbeğin \
O CÜMLEDEKİ kullanımını sınıflandır:

D = öbek DEYİM olarak kullanılmış: bütüne ait, mecazi/aktarılmış bir anlam taşıyor; birebir \
okunuşu bu bağlamda olağan değil. Örn: "Projede yol aldık", "Soruyu topu taca atarak geçiştirdi".
L = öbek LİTERAL kullanılmış: kelimesi kelimesine, gerçek/temel anlamıyla; birebir okunuş bu \
bağlamda olağan ve mantıklı. Örn: "Otobüs kısa sürede çok yol aldı", "Hırsız eli kolu bağlanarak \
götürüldü", "Devrede kısa devre oldu".
E = öbek aslında deyim değil: sıradan eşdizim ("karar verdi", "yardım etti"), terim ya da \
bitki/canlı adı ("aslan ağzı" = çiçek, "kuşburnu", "deve dikeni"), ya da karar verilemiyor.

Test soruları: Öbeğin birebir okunuşu bu bağlamda olağan mı? Özne/nesne somut mu, soyut mu? \
Öbeği temel (sözlük) anlamıyla değiştirince cümle tutarlı kalıyor mu? (evet → L)

Her satır için SADECE: <numara> <D|L|E>  — başka açıklama yok.

"""


def load_items() -> list[dict]:
    if not IN_JSON.exists():
        sys.exit(f"{IN_JSON} yok — önce: python prepare_tdk_corpus_examples.py")
    data = json.loads(IN_JSON.read_text(encoding="utf-8"))
    if data and "idiom" not in data[0]:
        sys.exit(f"{IN_JSON} 'idiom'/'span' alanı taşımıyor — prepare_tdk_corpus_examples.py'yi "
                 "güncel sürümle yeniden çalıştır.")
    return data


def sample(items: list[dict], per_idiom: int, total: int, seed: int) -> list[dict]:
    import random
    rng = random.Random(seed)
    by: dict[str, list[dict]] = {}
    for it in items:
        by.setdefault(it["idiom"], []).append(it)
    picked = []
    for k, lst in by.items():
        rng.shuffle(lst)
        picked += lst[:per_idiom]
    rng.shuffle(picked)
    return picked[:total]


def call_llm(base_url: str, model: str, api_key: str, batch: list[dict], timeout: int) -> dict[int, str]:
    lines = [f"{i+1}. {' '.join(it['words'])}  ||  öbek: {it['span']}" for i, it in enumerate(batch)]
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": TASK + "\n".join(lines)},
        ],
        "temperature": 0,
        "max_tokens": 8 * len(batch) + 64,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = json.loads(r.read())["choices"][0]["message"]["content"]
    out: dict[int, str] = {}
    for m in re.finditer(r"(\d+)\s*[.):\-]?\s*([DLEdle])\b", txt):
        out[int(m.group(1))] = m.group(2).upper()
    return out


def _write_recs(recs: list[dict], mode: str = "w") -> None:
    with SAMPLE_RECS.open(mode, encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps({"idx": r["idx"], "words": r["words"], "tags": r["tags"],
                                "idiom": r["idiom"], "span": r["span"]}, ensure_ascii=False) + "\n")


def _write_tsv(recs: list[dict], mode: str = "w") -> None:
    with SAMPLE_TSV.open(mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("idx\tspan\tcümle\n")
        for r in recs:
            f.write(f"{r['idx']}\t{r['span']}\t{' '.join(r['words'])}\n")


def load_records(items: list[dict], per_idiom: int, total: int, seed: int) -> list[dict]:
    """apply/focus: kayıt listesini _corpus_sample_records.jsonl'den oku; yoksa örnekten kur."""
    if SAMPLE_RECS.exists():
        return [json.loads(l) for l in SAMPLE_RECS.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = [{**it, "idx": i} for i, it in enumerate(sample(items, per_idiom, total, seed))]
    _write_recs(recs)
    return recs


def dump_sample(picked: list[dict]) -> None:
    """Örneklenen kayıtları elle sınıflandırma için numaralı TSV + kayıt jsonl'e yaz."""
    recs = [{**it, "idx": i} for i, it in enumerate(picked)]
    _write_tsv(recs)
    _write_recs(recs)
    print(f"yazıldı: {SAMPLE_TSV.name} + {SAMPLE_RECS.name}  ({len(recs)} satır)")
    print(f"Her satırı D/L/E ile sınıflandırıp {MANUAL_LABELS.name}'e 'idx<TAB>D' biçiminde yaz, "
          f"sonra: python filter_corpus_idiomaticity.py --apply")


def focus_l(items: list[dict], per_idiom: int, total: int, seed: int, focus_n: int) -> None:
    """L etiketi almış deyimlerden (literal-eğilimli) DAHA ÇOK cümle örnekle → elle sınıflandır.
    Rastgele örnekte L oranı ~%12; bu deyimlerde ~%40-50 → çok daha verimli L madenciliği."""
    recs = load_records(items, per_idiom, total, seed)
    lab: dict[int, str] = {}
    if MANUAL_LABELS.exists():
        for line in MANUAL_LABELS.read_text(encoding="utf-8").splitlines():
            p = re.split(r"[\t ,]+", line.strip())
            if len(p) >= 2 and p[0].isdigit() and p[1].upper() in "DLE":
                lab[int(p[0])] = p[1].upper()
    l_idioms = {r["idiom"] for r in recs if lab.get(r["idx"]) == "L"}
    seen_txt = {" ".join(r["words"]) for r in recs}
    print(f"L etiketli deyim: {len(l_idioms)}  (bunlardan +{focus_n} cümle/deyim örneklenecek)")

    import random
    rng = random.Random(seed + 1)
    by: dict[str, list[dict]] = {}
    for it in items:
        if it["idiom"] in l_idioms and " ".join(it["words"]) not in seen_txt:
            by.setdefault(it["idiom"], []).append(it)
    new: list[dict] = []
    nxt = max((r["idx"] for r in recs), default=-1) + 1
    for k in sorted(by):
        rng.shuffle(by[k])
        for it in by[k][:focus_n]:
            new.append({**it, "idx": nxt}); nxt += 1
    _write_tsv(new, "a")
    _write_recs(new, "a")
    print(f"eklendi: {len(new)} yeni cümle (idx {new[0]['idx'] if new else '-'}..{nxt-1}) "
          f"→ {SAMPLE_TSV.name} / {SAMPLE_RECS.name}")
    print(f"Yeni satırları {MANUAL_LABELS.name}'e ekle, sonra: python filter_corpus_idiomaticity.py --apply --balance")


TEST_JSON = PROJECT_ROOT / "idiom_data" / "corpus_minpair_test.json"


def apply_manual(recs: list[dict], holdout: float = 0.15, seed: int = 7,
                 l_only: bool = False, balance: bool = False, lit_class: bool = False) -> None:
    """D → span'li örnek, L → aynı öbek hep-O (minimal-çift negatif sinyali). E atılır.
    `recs` = _corpus_sample_records.jsonl (idx alanlı). Deyim düzeyinde held-out.
    l_only=True: eğitime YALNIZ L→hep-O. balance=True: D'yi L sayısına indir (1:1).
    lit_class=True (Fikir 4): L → hep-O yerine `B/I-VID-LIT` span'i (etiket uzayında
    açık "deyim-biçimin literal kullanımı" sınıfı; label_space.json'a eklenmiş olmalı)."""
    if not MANUAL_LABELS.exists():
        sys.exit(f"{MANUAL_LABELS} yok — önce --dump ve elle etiketleme.")
    lab: dict[int, str] = {}
    for line in MANUAL_LABELS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("idx"):
            continue
        parts = re.split(r"[\t ,]+", line)
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].upper() in "DLE":
            lab[int(parts[0])] = parts[1].upper()
    dist = Counter(lab.values())
    picked = recs  # idx == list konumu (jsonl sıralı yazılıyor); güvenlik için idx ile eşle
    by_idx = {r["idx"]: r for r in recs}

    def to_rec(it: dict, label: str) -> dict:
        if label == "D":
            return {"words": it["words"], "tags": it["tags"]}
        if lit_class:  # L → B/I-VID-LIT span'i (Fikir 4)
            lt = [t + "-LIT" if t != "O" else "O" for t in it["tags"]]
            return {"words": it["words"], "tags": lt}
        return {"words": it["words"], "tags": ["O"] * len(it["words"])}  # L → hep-O

    import random
    rng = random.Random(seed)
    # held-out (deyim düzeyinde): önce HEM D HEM L taşıyan deyimler (gerçek minimal çift),
    # sonra rastgele D/L deyimleriyle doldur. Amaç: GÖRÜLMEMİŞ deyimlerde model idyomatik
    # kullanımda span işaretlemeli, literal kullanımda İŞARETLEMEMELİ.
    by_idiom_lab: dict[str, set[str]] = {}
    for it in picked:
        L = lab.get(it["idx"])
        if L in ("D", "L"):
            by_idiom_lab.setdefault(it["idiom"], set()).add(L)
    dl_idioms = sorted(by_idiom_lab)
    rng.shuffle(dl_idioms)
    paired = [k for k in dl_idioms if {"D", "L"} <= by_idiom_lab[k]]
    n_test = max(len(paired), round(len(dl_idioms) * holdout))
    test_idioms = set(paired + [k for k in dl_idioms if k not in paired])
    test_idioms = set(list(paired) + [k for k in dl_idioms if k not in paired][:n_test - len(paired)])

    train_d, train_l, test_out = [], [], []
    for it in picked:
        L = lab.get(it["idx"])
        if L not in ("D", "L"):
            continue
        if it["idiom"] in test_idioms:
            test_out.append(to_rec(it, L))                 # test'te D ve L ikisi de
        elif L == "L":
            train_l.append(to_rec(it, "L"))
        elif not l_only:
            train_d.append(to_rec(it, "D"))
    if balance and not l_only and len(train_d) > len(train_l):
        rng.shuffle(train_d)
        train_d = train_d[:len(train_l)]                    # 1:1 D:L
    train_out = train_d + train_l

    OUT_JSON.write_text(json.dumps(train_out, ensure_ascii=False), encoding="utf-8")
    TEST_JSON.write_text(json.dumps(test_out, ensure_ascii=False), encoding="utf-8")
    # held-out deyim listesi — train_idiomaticity_clf.py bunu DEYİM düzeyinde bölme için okur
    # (cümle-metni düzeyi yetmez: focus-l sonradan aynı deyimden yeni cümle ekleyince sızar)
    (PROJECT_ROOT / "idiom_data" / "_holdout_idioms.json").write_text(
        json.dumps(sorted(test_idioms), ensure_ascii=False), encoding="utf-8")
    nd = sum(1 for r in train_out if any(t.startswith(("B-VID", "I-VID")) and not t.endswith("-LIT")
                                          or t.startswith(("B-LVC", "I-LVC")) for t in r["tags"]))
    lspec = "L→VID-LIT" if lit_class else "L→hepO"
    print(f"etiket: {len(lab)}/{len(picked)}  dağılım {dict(dist)}")
    print(f"train: {len(train_out):,} ({nd} D-span / {len(train_out)-nd} {lspec})  → {OUT_JSON.name}")
    print(f"held-out minimal-çift test: {len(test_out)} kayıt, {len(test_idioms)} deyim  → {TEST_JSON.name}")
    print("Sonraki: python train_idiom_bert.py --class-weights --tdk-examples --corpus-glu --epochs 10")
    print("         python benchmark/eval_idiom.py --local --checkpoint <ckpt> --eval-file idiom_data/corpus_minpair_test.json  # (train_idiom_bert --eval yolu)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true", help="örneği elle sınıflandırma için TSV'ye yaz")
    ap.add_argument("--apply", action="store_true", help="elle etiketleri uygula → corpus_examples_glu.json")
    ap.add_argument("--l-only", action="store_true",
                    help="--apply: eğitime yalnız L→hep-O kayıtları (derlem D pozitifi ekleme)")
    ap.add_argument("--balance", action="store_true",
                    help="--apply: derlem D kayıtlarını L sayısına indir (1:1 D:L)")
    ap.add_argument("--lit-class", action="store_true",
                    help="--apply: L → hep-O yerine B/I-VID-LIT span'i (Fikir 4, etiket uzayı genişletme)")
    ap.add_argument("--focus-l", action="store_true",
                    help="L etiketli (literal-eğilimli) deyimlerden DAHA ÇOK cümle örnekle (v13)")
    ap.add_argument("--focus-n", type=int, default=6, help="--focus-l: deyim başına ek cümle")
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1"))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "local-model"))
    ap.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")))
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--max-per-idiom", type=int, default=3)
    ap.add_argument("--max-total", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()

    items = load_items()
    picked = sample(items, args.max_per_idiom, args.max_total, args.seed)
    key = lambda it: " ".join(it["words"])
    print(f"{len(items):,} örnek → örneklenen {len(picked):,} "
          f"(deyim başına ≤{args.max_per_idiom}, toplam ≤{args.max_total})")

    if args.dump:
        return dump_sample(picked)
    if args.focus_l:
        return focus_l(items, args.max_per_idiom, args.max_total, args.seed, args.focus_n)
    if args.apply:
        return apply_manual(load_records(items, args.max_per_idiom, args.max_total, args.seed),
                            l_only=args.l_only, balance=args.balance, lit_class=args.lit_class)

    if args.restart:
        LABELS.unlink(missing_ok=True)
    done: dict[str, str] = {}
    if LABELS.exists():
        for line in LABELS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done[d["k"]] = d["label"]
    todo = [it for it in picked if key(it) not in done]
    print(f"etiketli {len(done):,}, kalan {len(todo):,}")

    t = time.time()
    with LABELS.open("a", encoding="utf-8") as lf:
        for b0 in range(0, len(todo), args.batch):
            batch = todo[b0:b0 + args.batch]
            try:
                res = call_llm(args.base_url, args.model, args.api_key, batch, args.timeout)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                print(f"  LLM hatası ({e}) — 10s bekle, tekrar dene")
                time.sleep(10)
                try:
                    res = call_llm(args.base_url, args.model, args.api_key, batch, args.timeout)
                except Exception as e2:
                    sys.exit(f"LLM erişilemiyor: {e2}\nLM Studio açık mı? --base-url doğru mu?")
            for i, it in enumerate(batch):
                lab = res.get(i + 1, "E")  # ayrıştırılamayan → E (elenir)
                lf.write(json.dumps({"k": key(it), "label": lab,
                                     "idiom": it["idiom"], "span": it["span"]},
                                    ensure_ascii=False) + "\n")
                done[key(it)] = lab
            lf.flush()
            n = len(done)
            if (b0 // args.batch) % 10 == 0:
                dist = Counter(done.values())
                print(f"  {n:,}/{len(picked):,}  {dict(dist)}  {time.time() - t:.0f}s")

    dist = Counter(done.values())
    kept = [it for it in picked if done.get(key(it)) == "D"]
    out = [{"words": it["words"], "tags": it["tags"]} for it in kept]
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"\netiket dağılımı: {dict(dist)}")
    print(f"tutulan (D): {len(out):,}  → {OUT_JSON.relative_to(PROJECT_ROOT)}")
    print("Sonraki: python train_idiom_bert.py --class-weights --tdk-examples --corpus-glu --epochs 10")


if __name__ == "__main__":
    main()
