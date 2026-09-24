"""Sözcük sınırı bilgisini TOKEN olarak değil, fonem başına ÖZNİTELİK olarak taşımak (Aşama 5 ablasyonu).

Neden: sözcük ayracı token'ı ~100 ms'lik ayrı bir birim olarak modellenip her sınırda "sözcükler arası boşluk" üretiyordu; ayracı tamamen
atmak ise (nosep/breaks koşuları) anlaşılırlığı çöktürdü (reports/stage5_boundaries.md). Öznitelik yolu sınırı süre harcamadan bildirir.

Öznitelik = final_sınıf + 4*sözcük_başı  (0..7):
  sözcük_başı : sözcüğün ilk fonemi 1
  final_sınıf : sözcüğün son fonemi için sonraki sınırın sınıfı  1 = B1 (bağ, duraklama yok), 2 = B2 (60-250 ms), 3 = B3 (>=250 ms / noktalama / cümle sonu)
                sözcük ortası fonemler ve noktalama token'ları 0.
Metinden türetilen sürüm (cls_by_k=None) dağıtılabilir: B3 = noktalama izleyen ya da son sözcük, aksi halde B1.
"""
from __future__ import annotations

from .symbols import PAUSES, WORD_SEP

N_FEATS = 8


def add_features(tokens: list[str], cls_by_k: dict[int, int] | None = None) -> tuple[list[str], list[int]]:
    """WORD_SEP'li token listesi -> (ayraçsız token listesi, aynı uzunlukta öznitelik listesi).
    cls_by_k: k-inci sözcük sınırı için ölçülmüş sınıf (1/2/3); yoksa/eksikse metin kuralı."""
    out: list[str] = []
    feats: list[int] = []
    word: list[int] = []  # mevcut sözcüğün fonem indeksleri (out içinde)
    punct = False
    k = 0

    def close(last_word: bool):
        nonlocal word, punct
        if word:
            rule = 3 if (punct or last_word) else 1
            cls = (cls_by_k or {}).get(k, rule) if not last_word else 3
            feats[word[0]] += 4
            feats[word[-1]] += cls
        word, punct = [], False

    for t in tokens:
        if t == WORD_SEP:
            close(False)
            k += 1
        elif t in PAUSES:
            out.append(t); feats.append(0)
            punct = True
        else:
            word.append(len(out)); out.append(t); feats.append(0)
    close(True)
    return out, feats
