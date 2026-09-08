#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TDK deyim sözlüğündeki gömülü örnek cümlelerden BIO eğitim verisi üretir.

`idiom_data/raw/tdk_atasozu_deyim.csv`'nin (`fetch_tdk_deyim.mjs`) `meaning` alanı çoğu
zaman `tanım: örnek cümle` biçiminde (bazen `1) ... : örnek. 2) ... : örnek.` çok-anlamlı,
bazen yazar alıntılı: `'... .' - T. Uyar.`). Bu, PARSEME'nin kapsamadığı isim/sıfat
deyimleri için GERÇEK bağlam cümlesi kaynağı — sentetik üretime gerek yok.

Zorluk: TDK'nin deyim metni sözlük/sözlük-madde biçiminde (`gözden düşmek`), örnek cümledeki
yüzey biçimiyle birebir eşleşmez (`gözden düştü`). Repo'nun kendi kural-tabanlı morfoloji
çözümleyicisi (`morphology.create_default_analyzer`) ile hem deyim kelimelerini hem cümle
kelimelerini GÖVDEYE (stem) indirgeyip aynı gövde dizisini cümlede ardışık arıyoruz — bu,
çekim farkını aşan basit ama repo-içi bir eşleştirme (corpus_engine'in kaldırılmış ÖSP
katmanındaki `naiveLemmaSeq()`'in parantez/`...` temizleme mantığı buraya taşındı, gövdeleme
için ise TAM UYUMLU YERİNE bu projenin kendi analizörü kullanıldı — dictionary/lemma değil,
stem yeterli çünkü yalnız span sınırı arıyoruz, tam morfolojik ayrıştırma değil).

Etiket: TDK 'idiom' (Deyim) kaydı → **VID** (PARSEME'nin verbal-idiom etiketiyle aynı sınıf;
kavramsal olarak deyim = deyim, PARSEME yalnız fiil-merkezli olanları etiketlemişti, TDK
hepsini). Yeni bir etiket sınıfı AÇILMIYOR — mevcut label_space (O/B-VID/I-VID/B-LVC/I-LVC)
korunuyor, TDK verisi VID sınıfını isim/sıfat deyimleriyle zenginleştiriyor.
'proverb' (Atasözü) kayıtları bu script'te İŞLENMİYOR — atasözü tipik olarak kendi başına
tam bir cümle, "daha büyük bir cümle içinde span" modeline uymuyor; kapsam dışı (plana not).

Çıktı: idiom_data/tdk_examples.json  ({"words":[...],"tags":[...]} listesi, train.json ile
aynı biçim) — train_idiom_bert.py'ye --tdk-examples ile karışım olarak eklenir.

Kullanım:
    python prepare_tdk_idiom_examples.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from morphology import create_default_analyzer  # noqa: E402

IN_CSV = PROJECT_ROOT / "idiom_data" / "raw" / "tdk_atasozu_deyim.csv"
OUT_JSON = PROJECT_ROOT / "idiom_data" / "tdk_examples.json"


def tr_lower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()


def strip_balanced_parens(text: str) -> str:
    """Parantezleri İÇERİKLERİYLE birlikte atar (corpus_engine ÖSP'nin naiveLemmaSeq'i)."""
    if text.count("(") != text.count(")"):
        return text.replace("(", "").replace(")", "")
    out, depth = [], 0
    for ch in text:
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            out.append(ch)
    return "".join(out)


def extract_examples(meaning: str) -> list[str]:
    """`meaning` alanından örnek cümleleri çıkarır (çok-anlamlı `N) ...` parçalarına böler)."""
    parts = re.split(r"\d+\)\s*", meaning)
    chunks = [p for p in parts if p.strip()] if len(parts) > 1 else [meaning]
    examples = []
    for chunk in chunks:
        if ":" not in chunk:
            continue
        # SON iki-noktadan böl, ilkinden değil — bazı tanımlar kendi içinde iki-nokta
        # taşıyor (kısaltma/alıntı), örnek cümle her zaman EN SONDAKİ ayırıcının sağında
        # (kod incelemesinde bulundu — ~%0.2 kayıtta birden fazla iki-nokta var).
        ex = chunk.rsplit(":", 1)[1].strip()
        if ex:
            examples.append(ex)
    return examples


_TOK_RE = re.compile(r"^(.*?)([.,!?;:'\"]+)$")


def tokenize(sent: str) -> list[str]:
    out = []
    for tok in sent.split():
        m = _TOK_RE.match(tok)
        out += [m.group(1), m.group(2)] if m and m.group(1) else [tok]
    return out


def idiom_stems(text: str, analyzer) -> list[str]:
    cleaned = strip_balanced_parens(text)
    words = [w for w in re.sub(r"[.,!?;:\"']", "", tr_lower(cleaned)).split() if w]
    return [analyzer.analyze(w).stem or w for w in words]


def find_span(idiom_seq: list[str], sent_stems: list[str]) -> tuple[int, int] | None:
    n = len(idiom_seq)
    if n == 0:
        return None
    for i in range(len(sent_stems) - n + 1):
        if sent_stems[i:i + n] == idiom_seq:
            return i, i + n
    return None


def main() -> None:
    if not IN_CSV.exists():
        sys.exit(f"{IN_CSV} yok — önce: node fetch_tdk_deyim.mjs")

    analyzer = create_default_analyzer(dictionary_path=str(PROJECT_ROOT / "turkish_words.txt"))

    import csv
    rows = list(csv.DictReader(IN_CSV.open(encoding="utf-8")))
    idiom_rows = [r for r in rows if r["kind"] == "idiom"]
    print(f"TDK deyim kaydı: {len(idiom_rows)}")

    # kayıtları DEYİM METNİNE göre grupla — aynı deyimin farklı örnekleri aynı split'e
    # düşsün (yoksa "eli açık"ın bir örneği train'de, bir başkası test'te aynı deyimi
    # ezberden geri çağırma riskini gizler; genelleme ölçmek için deyim-düzeyinde ayır).
    by_idiom: dict[str, list[dict]] = {}
    stats = Counter()
    for row in idiom_rows:
        idiom_seq = idiom_stems(row["text"], analyzer)
        if not idiom_seq:
            continue
        for example in extract_examples(row["meaning"]):
            stats["örnek_bulundu"] += 1
            words = tokenize(example)
            if len(words) < len(idiom_seq):
                stats["cümle_kısa_atlandı"] += 1
                continue
            sent_stems = [analyzer.analyze(tr_lower(w)).stem or tr_lower(w) for w in words]
            span = find_span(idiom_seq, sent_stems)
            if span is None:
                stats["eşleşme_bulunamadı"] += 1
                continue
            start, end = span
            tags = ["O"] * len(words)
            tags[start] = "B-VID"
            for i in range(start + 1, end):
                tags[i] = "I-VID"
            by_idiom.setdefault(row["text"], []).append({"words": words, "tags": tags})
            stats["eşleşti"] += 1

    n = len(by_idiom)
    dev_path = PROJECT_ROOT / "idiom_data" / "tdk_examples_dev.json"
    test_path = PROJECT_ROOT / "idiom_data" / "tdk_examples_test.json"

    if dev_path.exists() and test_path.exists():
        # FROZEN SPLIT: mevcut dev/test (ör. v5'in held-out'u) birebir korunur — yeni
        # taranan deyimler yalnız train'e eklenir. Bölme shuffle-slice olduğundan (deyim
        # sayısı değişince tüm permütasyon kayardı) versiyonlar-arası kıyas ancak böyle
        # temiz kalır. Eşleme cümle-metni düzeyinde: bir deyimin herhangi bir örnek cümlesi
        # eski dev/test dosyasında geçiyorsa o deyim o split'e sabitlenir.
        def _texts(p):
            return {" ".join(r["words"]) for r in json.loads(p.read_text(encoding="utf-8"))}

        frozen = {"dev": _texts(dev_path), "test": _texts(test_path)}
        split_keys = {"dev": set(), "test": set(), "train": set()}
        for key, recs in by_idiom.items():
            rec_texts = {" ".join(r["words"]) for r in recs}
            if rec_texts & frozen["test"]:
                split_keys["test"].add(key)
            elif rec_texts & frozen["dev"]:
                split_keys["dev"].add(key)
            else:
                split_keys["train"].add(key)
        print(f"FROZEN SPLIT: dev/test mevcut dosyalardan sabitlendi "
              f"(dev {len(split_keys['dev'])} deyim / test {len(split_keys['test'])} deyim), "
              f"kalan {len(split_keys['train'])} deyim → train")
    else:
        import random
        idiom_keys = sorted(by_idiom)  # deterministik sıra, sonra sabit seed'le karıştır
        random.Random(42).shuffle(idiom_keys)
        # min(..., n//3): çok küçük deyim havuzunda (ör. kısmi/smoke veri) dev+test train'i
        # tüketmesin — kod incelemesinde bulundu (n<=2 iken train boş kalabiliyordu).
        n_dev = min(max(1, round(n * 0.10)), n // 3) if n >= 3 else 0
        n_test = min(max(1, round(n * 0.10)), n // 3) if n >= 3 else 0
        split_keys = {
            "dev": set(idiom_keys[:n_dev]),
            "test": set(idiom_keys[n_dev:n_dev + n_test]),
            "train": set(idiom_keys[n_dev + n_test:]),
        }
    splits = {name: [rec for k in keys for rec in by_idiom[k]] for name, keys in split_keys.items()}

    # Cümle-düzeyi sızıntı koruması: TDK'de AYNI alıntı cümle birden fazla farklı deyime
    # örnek olarak geçebiliyor (örn. "... el ayak çekilmişti ..." hem "el ayak çekilmek"
    # hem başka bir deyimin örneği). Deyim-anahtarına göre bölmek bunu yakalamaz — cümle
    # METNİ train'de görülmüşse dev/test'ten çıkar (train > dev > test önceliği).
    seen_text: set[str] = set()
    dropped = Counter()
    for name in ("train", "dev", "test"):
        kept = []
        for rec in splits[name]:
            text = " ".join(rec["words"])
            if text in seen_text:
                dropped[name] += 1
                continue
            kept.append(rec)
            seen_text.add(text)
        splits[name] = kept
    if dropped:
        print(f"cümle-düzeyi çapraz-sızıntı temizliği: {dict(dropped)} kayıt düşürüldü")

    print(f"benzersiz deyim (eşleşen): {n}  → train {len(split_keys['train'])} deyim / "
          f"dev {len(split_keys['dev'])} / test {len(split_keys['test'])}")
    for name, recs in splits.items():
        path = PROJECT_ROOT / "idiom_data" / f"tdk_examples_{name}.json"
        path.write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
        print(f"  yazıldı: {path.relative_to(PROJECT_ROOT)}  ({len(recs)} kayıt, "
              f"{len(split_keys[name])} deyim)")

    records = splits["train"]  # geriye dönük uyumluluk: tdk_examples.json = yalnız train parçası
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    print(f"\nyazıldı: {OUT_JSON.relative_to(PROJECT_ROOT)}  ({len(records)} kayıt)")
    print("istatistik:")
    for k in ("örnek_bulundu", "eşleşti", "eşleşme_bulunamadı", "cümle_kısa_atlandı"):
        print(f"  {k}: {stats[k]}")


if __name__ == "__main__":
    main()
