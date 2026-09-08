#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GLU deyim etiketleme kılavuzundan idyomatik/literal minimal çiftleri + hard-negative'ler.

Kaynak: `GLU_Deyim_Etiketleme_Ogretmen_Kilavuzu.pdf` (Gülsün Leylâ Uzun, Ankara Üniv., ÖSP).
Damıtılmış karar çerçevesi: `.claude/skills/idiom/glu_karar_cercevesi.md`.

DizgeBERT-Idiom'un precision tavanının kökü: weak-supervision (TDK stem-match, derlem sıkı
eşleşme) yalnız YÜZEY BİÇİME bakıyor, GLU'nun Aşama 3'ünü (bağlamda gerçek mi mecazi mi)
hiç uygulamıyor. Bu dosya tam o sinyali veriyor — küçük ama elle seçilmiş, gerekçelendirilmiş.

İki çıktı:
  1. `idiom_data/glu_hard_examples.json`  — {"words","tags"} eğitim örnekleri (yalnız kılavuzdaki
     GERÇEK cümleler). `train_idiom_bert.py --glu-examples` ile train'e eklenir.
  2. `glu_diagnostic_cases()` — `benchmark/eval_idiom.py --mode glu` için tanı seti
     (idyomatik cümlede span İŞARETLEMELİ, literal/terim cümlesinde İŞARETLEMEMELİ).

Kullanım:
    python prepare_glu_examples.py            # eğitim json'unu yaz
    python benchmark/eval_idiom.py --local --checkpoint <ckpt> --mode glu
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from prepare_tdk_idiom_examples import tokenize, tr_lower  # noqa: E402

OUT_JSON = PROJECT_ROOT / "idiom_data" / "glu_hard_examples.json"

# ─────────────────────────────────────────────────────────────────────────────
#  Minimal çiftler — kılavuzdaki GERÇEK cümleler (Tablo 18/19/30/31/34 + C.4).
#  (cümle, span_yüzey_biçimi | None, kategori)  — None = span beklenmez (literal/terim).
#  span_yüzey_biçimi cümlede ARDIŞIK bir alt-dizi olmalı (tokenize sonrası).
# ─────────────────────────────────────────────────────────────────────────────
PAIRS: list[tuple[str, str | None, str | None]] = [
    # eli kolu bağlanmak
    ("Yeni yönetmelik yüzünden öğretmenin eli kolu bağlandı .", "eli kolu bağlandı", "VID"),
    ("Hırsız , polis tarafından eli kolu bağlanarak götürüldü .", None, None),
    # gözden düşmek
    ("Yalan söylediği ortaya çıkınca arkadaşlarının gözünden düştü .", "gözünden düştü", "VID"),
    ("Kontak lensi gözünden düştü ve yere yuvarlandı .", None, None),
    # yol almak
    ("Projede son aylarda önemli ölçüde yol aldık .", "yol aldık", "VID"),
    ("Çalışmalarımız son aylarda önemli ölçüde yol aldı .", "yol aldı", "VID"),
    ("Yürüyüş kolu kısa sürede oldukça yol aldı .", None, None),
    ("Otobüs kısa sürede çok yol aldı .", None, None),
    # el vermek
    ("Bu çalışmaya birçok kişi el verdi .", "el verdi", "LVC"),
    ("Bu projeye birçok kurum el verdi .", "el verdi", "LVC"),
    ("Çocuğu kaldırmak için el verdi .", None, None),
    # topu taca atmak
    ("Soruyu yine topu taca atarak geçiştirdi .", "topu taca atarak", "VID"),
    ("Oyuncu topu taca attı .", None, None),
    # rölantide olmak (deyim) vs terim
    ("Proje aylardır rölantide .", "rölantide", "VID"),
    ("Motor rölantide çalışıyor .", None, None),
    # kısa devre yapmak
    ("Soruyu duyunca beynim kısa devre yaptı .", "kısa devre yaptı", "VID"),
    ("Devrede kısa devre meydana geldi .", None, None),
    # devre dışı kalmak
    ("Sakatlığı nedeniyle sezonu devre dışı geçirdi .", "devre dışı", "VID"),
    ("Arızalı ünite devre dışı bırakıldı .", None, None),
    # sigortası atmak
    ("Bir anda sigortaları attı .", "sigortaları attı", "VID"),
    ("Aşırı yüklenince sigorta attı .", None, None),
    # tek yönlü deyim örnekleri (kılavuz Tablo 30 — "eşdizim sanılan deyim")
    ("Toplantıda ben de söz aldım .", "söz aldım", "VID"),
    ("Toplantıda bana da söz verildi .", "söz verildi", "VID"),
    ("Danışmanım tez çalışmam boyunca bana yol gösterdi .", "yol gösterdi", "VID"),
]

# ─────────────────────────────────────────────────────────────────────────────
#  Hard-negative — YALNIZ tanı için (eğitime girmez; cümleler kılavuz örneklerinden
#  minimal biçimde kuruldu). Kılavuz B.5.1 / B.7.1 / B.6: bunlar deyim SANILAN ama
#  eşdizim / terim / tek-sözcük. Model bunlarda VID span İŞARETLEMEMELİ.
#  (LVC kabul edilebilir — eşdizimler için; ama VID = hata.)
# ─────────────────────────────────────────────────────────────────────────────
HARD_NEG_DIAG: list[tuple[str, str]] = [
    # deyim sanılan eşdizimlilik → VID YANLIŞ
    ("Komisyon bu konuda karar verdi .", "eşdizim"),
    ("Uzmandan görüş aldık .", "eşdizim"),
    ("Öğretmen sınıfa izin verdi .", "eşdizim"),
    ("Rapor sonuca önemli katkı sağladı .", "eşdizim"),
    ("Toplantıda herkes not aldı .", "eşdizim"),
    # deyim sanılan terim (bitki/kavram adı) → span YOK / VID YANLIŞ
    ("Bahçede kuşburnu topladık .", "terim"),
    ("Duvarın dibinde aslan ağzı açmıştı .", "terim"),
    ("Astronomlar yeni bir kara delik keşfetti .", "terim"),
    ("Tarlanın kenarını deve dikeni kaplamış .", "terim"),
    # tek sözcüklü mecaz → span YOK
    ("Adam tam bir tilki .", "tek-sözcük"),
    ("Bu haber gerçek bir bomba .", "tek-sözcük"),
]


def _find_span(span_words: list[str], toks: list[str]) -> tuple[int, int] | None:
    n = len(span_words)
    low = [tr_lower(t) for t in toks]
    want = [tr_lower(w) for w in span_words]
    for i in range(len(toks) - n + 1):
        if low[i:i + n] == want:
            return i, i + n
    return None


def build_training_records() -> list[dict]:
    """Yalnız minimal çiftlerin GERÇEK cümleleri → {words,tags}. Literal cümleler → hep O."""
    recs = []
    for sent, span, cat in PAIRS:
        toks = tokenize(sent)
        tags = ["O"] * len(toks)
        if span is not None:
            sw = tokenize(span)
            found = _find_span(sw, toks)
            if found is None:
                sys.exit(f"span '{span}' bulunamadı: {sent}")
            s, e = found
            tags[s] = f"B-{cat}"
            for j in range(s + 1, e):
                tags[j] = f"I-{cat}"
        recs.append({"words": toks, "tags": tags})
    return recs


def glu_diagnostic_cases():
    """(etiket, cümle, beklenen_öbek|None, beklenen_kategori|None) — eval_idiom.CASES biçimi."""
    out = []
    for sent, span, cat in PAIRS:
        label = "glu-deyim" if span else "glu-literal"
        out.append((label, sent, span, cat))
    for sent, kind in HARD_NEG_DIAG:
        out.append((f"glu-{kind}", sent, None, None))
    return out


def main() -> None:
    recs = build_training_records()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
    n_span = sum(1 for r in recs if any(t != "O" for t in r["tags"]))
    print(f"yazıldı: {OUT_JSON.relative_to(PROJECT_ROOT)}  ({len(recs)} kayıt, "
          f"{n_span} span'li / {len(recs) - n_span} hep-O literal)")
    print(f"tanı seti: {len(glu_diagnostic_cases())} vaka "
          f"({sum(1 for c in glu_diagnostic_cases() if c[2])} span-beklenen / "
          f"{sum(1 for c in glu_diagnostic_cases() if not c[2])} span-beklenmeyen)")
    print("Sonraki: python benchmark/eval_idiom.py --local --checkpoint <ckpt> --mode glu")
    print("         python train_idiom_bert.py --class-weights --tdk-examples --glu-examples --epochs 10")


if __name__ == "__main__":
    main()
