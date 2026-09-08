# -*- coding: utf-8 -*-
"""DizgeBERT-Idiom — HF PreTrainedModel sarmalayıcı.

Katman yapısı `train_idiom_bert.IdiomTagger` ile BİREBİR aynı (aynı state_dict
anahtarları) → eğitim checkpoint'i doğrudan yüklenir.

Türkçe deyim (VID) ve eşdizim/yardımcı-fiil (LVC.full) span'lerini token-düzeyi BIO
etiketiyle işaretler (serbest birleşim = O). Kelime temsili = ilk subword ⊕ son subword,
DizgeBERT-Morph ile aynı yöntem. Tek head — çok-treebank yok (kaynak tek: PARSEME-TR).

Kullanım:
    from transformers import AutoModel, AutoTokenizer
    m = AutoModel.from_pretrained("iatagun/DizgeBERT-Idiom", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained("iatagun/DizgeBERT-Idiom")
    print(m.predict(["Sonunda", "gözden", "düştü", "."], tokenizer=tok))
    # [('Sonunda', 'O'), ('gözden', 'B-VID'), ('düştü', 'I-VID'), ('.', 'O')]
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_dizgebert_idiom import DizgeBertIdiomConfig


def _is_inside_tag(t: str) -> bool:
    """`I-X`/`i-x` (BIO 'inside') mi? Büyük/küçük harf duyarsız — katman 1 (O/B/I,
    büyük) ve katman 2 (o/b/i, küçük — bkz. bigappy-unicrossy) AYNI fonksiyonlarla
    işlenebilsin diye."""
    return len(t) > 1 and t[1] == "-" and t[0] in "Ii"


def _valid_transition(prev_tag: str, cur_tag: str) -> bool:
    """`I-X`/`i-x` yalnız hemen önceki etiket `B-X`/`I-X` (kendi katmanında) ise
    geçerli (yetim başlangıç veya kategori değişimi geçersiz). `O`/`o`/`B-*`/`b-*`'a
    her zaman geçilebilir."""
    if _is_inside_tag(cur_tag):
        cat = cur_tag.split("-", 1)[1]
        begin_tag = ("B" if cur_tag[0] == "I" else "b") + "-" + cat
        return prev_tag in (begin_tag, cur_tag)
    return True


def viterbi_decode(logits: "torch.Tensor", tags: list[str]) -> list[str]:
    """[W, T] emisyon logit'i → en-iyi-yol BIO dizisi (Viterbi, geçiş kısıtlı).

    Argmax'ın aksine yapısal geçerliliği garanti eder: yetim `I-X` (önce eşleşen `B-X`
    yok) veya kategori-karışık `B-VID→I-LVC` gibi geçişler -inf skorla engellenir —
    `decode_bio_spans`'ın toleranslı-ama-tahminî onarımına ihtiyaç kalmaz. Yeniden eğitim
    GEREKTİRMEZ — yalnız çıkarım-zamanı çözümleme (Joint modelin `canonicalize()` post-hoc
    katmanıyla aynı felsefe: model ağırlıkları sabit, yalnız çıktı katmanı).
    T=5 sabit, W (kelime sayısı) küçük → O(W·T²) önemsiz maliyetli.
    """
    W, T = logits.shape
    if W == 0:
        return []
    NEG = float("-1e9")
    emit = logits.tolist()
    trans = [[0.0 if _valid_transition(tags[p], tags[c]) else NEG for c in range(T)] for p in range(T)]

    dp = [[NEG] * T for _ in range(W)]
    bp = [[0] * T for _ in range(W)]
    for j in range(T):
        dp[0][j] = emit[0][j] if not _is_inside_tag(tags[j]) else NEG  # cümle I-X ile başlayamaz
    for t in range(1, W):
        for j in range(T):
            best_score, best_prev = NEG, 0
            for p in range(T):
                s = dp[t - 1][p] + trans[p][j]
                if s > best_score:
                    best_score, best_prev = s, p
            dp[t][j] = best_score + emit[t][j]
            bp[t][j] = best_prev

    last = max(range(T), key=lambda j: dp[W - 1][j])
    path = [last]
    for t in range(W - 1, 0, -1):
        path.append(bp[t][path[-1]])
    path.reverse()
    return [tags[i] for i in path]


def align_words(tokenizer, words: list[str], max_len: int, device=None):
    """Tokenize + ilk⊕son subword hizalaması — `first_pos`/`last_pos` tensörlerini üretir.

    `predict()`, `benchmark/eval_idiom.py` ve `predict_idiom.py`'nin yerel çıkarım
    yollarındaki AYNI hizalama mantığının TEK kopyası (üç ayrı kopya birbirinden
    sessizce sapabiliyordu — bkz. kod incelemesi). `kept` her zaman `[0, len(kept))`
    aralığıdır (kırpma yalnız sondan keser) — `words[w] for w in kept` bu yüzden
    güvenlidir.
    """
    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt",
                    truncation=True, max_length=max_len)
    if device is not None:
        enc = enc.to(device)
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for i, wid in enumerate(enc.word_ids()):
        if wid is None:
            continue
        first.setdefault(wid, i)
        last[wid] = i
    kept = sorted(first)
    dev = enc["input_ids"].device
    fp = torch.tensor([[first[w] for w in kept]], device=dev)
    lp = torch.tensor([[last[w] for w in kept]], device=dev)
    return enc, kept, fp, lp


def decode_bio_spans(tags: list[str]) -> list[tuple[int, int, str]]:
    """BIO etiket dizisi → [(başlangıç, bitiş_hariç, kategori), ...] span listesi.

    Büyük/küçük harf duyarsız (`O/B-X/I-X` katman 1 VE `o/b-x/i-x` katman 2 için aynı
    fonksiyon). Şema-dışı bir `I-X` (önce eşleşen `B-X`/`I-X` yok, ya da kategori
    değişti) yeni span başlangıcı sayılır — hatalı model çıktısına karşı toleranslı
    çözümleme.
    """
    spans: list[tuple[int, int, str]] = []
    start, cat = None, None
    for i, t in enumerate(tags):
        is_o = t in ("O", "o")
        c = None if is_o else t.split("-", 1)[1]
        continues = _is_inside_tag(t) and start is not None and c == cat
        if not continues:
            if start is not None:
                spans.append((start, i, cat))
            start, cat = (i, c) if c is not None else (None, None)
    if start is not None:
        spans.append((start, len(tags), cat))
    return spans


def decode_bigappy_spans(tags1: list[str], tags2: list[str]) -> list[tuple]:
    """İki katmanı (bkz. bigappy-unicrossy) birleştirip span listesi üretir.

    Sıradan (bitişik) span → `(start, end, kategori)` 3'lü — eskisiyle bire bir uyumlu.
    Gap'li span → `(start1, end1, start2, end2, kategori)` 5'li (küçük pozisyon önce).
    Katman 2'deki her parça, AYNI kategoriden en yakın EŞLEŞMEMİŞ katman-1 parçasıyla
    eşleştirilir (bir cümlede aynı kategoriden birden fazla gap'li span nadiren çakışır;
    yakınlık sezgisel ama pratikte yeterli — ampirik olarak PARSEME'de asla >1 gap'li
    span/kategori/cümle görülmedi). Katman 2'de eşleşmemiş (yetim) parça yok sayılır.
    """
    l1 = decode_bio_spans(tags1)
    l2 = decode_bio_spans(tags2)
    used = [False] * len(l1)
    spans: list[tuple] = []
    for s2, e2, cat2 in l2:
        best_i, best_dist = None, None
        for i, (s1, e1, cat1) in enumerate(l1):
            if used[i] or cat1 != cat2:
                continue
            dist = min(abs(s2 - e1), abs(s1 - e2))
            if best_dist is None or dist < best_dist:
                best_i, best_dist = i, dist
        if best_i is not None:
            used[best_i] = True
            s1, e1, cat1 = l1[best_i]
            spans.append((s1, e1, s2, e2, cat1) if s1 <= s2 else (s2, e2, s1, e1, cat1))
    for i, (s, e, cat) in enumerate(l1):
        if not used[i]:
            spans.append((s, e, cat))
    return spans


class DizgeBertIdiomForTokenClassification(PreTrainedModel):
    config_class = DizgeBertIdiomConfig

    def __init__(self, config: DizgeBertIdiomConfig):
        super().__init__(config)
        # Kaydedilmiş modelde encoder ağırlıkları state_dict'te; iskeleti config'ten kur.
        self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(config.encoder_name))
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.tag_head = nn.Linear(2 * h, len(config.tags))
        # bigappy-unicrossy 2. katman — yalnız gap'li span'lerin 2. parçası (bkz.
        # prepare_idiom_data.py). Morph'un çoklu-head deseniyle aynı: bağımsız softmax.
        self.tag_head2 = nn.Linear(2 * h, len(config.tags2))
        self.post_init()

    def forward(self, input_ids, attention_mask=None, first_pos=None, last_pos=None):
        B, L = input_ids.shape
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if first_pos is None:  # her subword'ü kendi "kelimesi" say
            first_pos = last_pos = torch.arange(L, device=input_ids.device)[None].expand(B, -1)
        H = hs.size(-1)
        f = hs.gather(1, first_pos.unsqueeze(-1).expand(-1, -1, H))
        g = hs.gather(1, last_pos.unsqueeze(-1).expand(-1, -1, H))
        z = self.dropout(torch.cat([f, g], dim=-1))
        return ModelOutput(logits=self.tag_head(z), logits2=self.tag_head2(z))

    # ── kolaylık: ön-token'lanmış kelime listesi → [(kelime, katman1_etiket, katman2_etiket)] ──
    @torch.no_grad()
    def predict(self, words: list[str], tokenizer=None) -> list[tuple[str, str, str]]:
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(self.config._name_or_path)
        enc, kept, fp, lp = align_words(tokenizer, words, self.config.max_len, self.device)
        out = self.forward(enc["input_ids"], enc["attention_mask"], fp, lp)
        tags1 = viterbi_decode(out.logits[0], self.config.tags)
        tags2 = viterbi_decode(out.logits2[0], self.config.tags2)
        return list(zip([words[w] for w in kept], tags1, tags2))

    @torch.no_grad()
    def predict_spans(self, words: list[str], tokenizer=None) -> list[dict]:
        """`predict()` + iki-katman çözümleme → span listesi.

        Sıradan (bitişik) span: `{"text","start","end","category","gappy": False}`.
        Gap'li (süreksiz) span, ör. "sahip ... olarak": `{"text": "sahip ... olarak",
        "start","end" (1. parça), "start2","end2" (2. parça), "category", "gappy": True}`.
        """
        triples = self.predict(words, tokenizer)
        tags1 = [t1 for _w, t1, _t2 in triples]
        tags2 = [t2 for _w, _t1, t2 in triples]
        out = []
        for span in decode_bigappy_spans(tags1, tags2):
            if len(span) == 3:
                s, e, cat = span
                out.append({"text": " ".join(words[s:e]), "start": s, "end": e,
                            "category": cat, "gappy": False})
            else:
                s1, e1, s2, e2, cat = span
                text = " ".join(words[s1:e1]) + " ... " + " ".join(words[s2:e2])
                out.append({"text": text, "start": s1, "end": e1, "start2": s2, "end2": e2,
                            "category": cat, "gappy": True})
        return out
