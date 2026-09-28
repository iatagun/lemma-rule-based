"""Okuma kayıtlarından Praat TextGrid: `sözcük` katmanı (MMS zorlamalı hizalama; torchaudio MMS_FA) + boş `duraklama` katmanı (aynı aralıklar).
Kullanıcı yalnız `duraklama` katmanında duraklama hissettiği sözcüğün aralığına 1 / 2 / 3 yazar (bkz. D:/dizgetts/user_prosody/BENIOKU.md).

  python -X utf8 -m dizgetts.tools.make_textgrid --wav-dir D:/dizgetts/user_prosody/wav --sentences D:/dizgetts/user_prosody/cumleler.tsv
  python -X utf8 -m dizgetts.tools.make_textgrid --wav X.wav --text "cümle" --out X.TextGrid     # tek dosya (doğrulama)

Dosya adı: NN.wav (cumleler.tsv `no` sütunu). MMS_FA yalnız a-z tanır: Türkçe harfler eşlenir (ç->c, ğ->g, ı->i ...); hizalama sesle yapılır,
TextGrid'e sözcüğün ASIL yazımı yazılır. Sessizlikler (>= 30 ms) boş aralık olur. Mevcut bir TextGrid'in ÜZERİNE YAZMAZ (kullanıcının işaretleri korunur).
"""
import argparse, csv, os, re

import torch
import torchaudio

TR = str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u", "â": "a", "î": "i", "û": "u", "İ": "i"})
MIN_GAP = 0.03


def words_of(text):
    """-> [(asıl yazım, hizalama biçimi)]; noktalama sözcüğe yapışık kalır (asıl yazımda), hizalamada atılır."""
    out = []
    for w in text.split():
        key = re.sub(r"[^a-z]", "", w.replace("İ", "i").lower().translate(TR))
        if key:
            out.append((w, key))
    return out


class Aligner:
    def __init__(self, device="cpu"):
        b = torchaudio.pipelines.MMS_FA
        self.dev, self.sr = torch.device(device), b.sample_rate
        self.model = b.get_model().to(self.dev).eval()
        self.tok, self.align = b.get_tokenizer(), b.get_aligner()

    @torch.inference_mode()
    def __call__(self, wav_path, text):
        w, sr = torchaudio.load(wav_path)
        w = torchaudio.functional.resample(w.mean(0, keepdim=True), sr, self.sr)
        ws = words_of(text)
        em, _ = self.model(w.to(self.dev))
        spans = self.align(em[0], self.tok([k for _, k in ws]))
        ratio = w.shape[1] / em.shape[1] / self.sr
        return [(orig, s[0].start * ratio, s[-1].end * ratio) for (orig, _), s in zip(ws, spans)], w.shape[1] / self.sr


def intervals(words, dur):
    iv, t = [], 0.0
    for w, s, e in words:
        if s - t >= MIN_GAP:
            iv.append((t, s, ""))
        iv.append((t if s - t < MIN_GAP else s, max(e, t + 1e-3), w))  # küçük boşluk/örtüşme sözcüğe katılır; aralıklar ardışık kalır
        t = iv[-1][1]
    if dur - t > 1e-3:
        iv.append((t, dur, ""))
    return iv


def write_textgrid(path, iv, dur):
    def tier(name, labels):
        rows = "".join(f'        intervals [{i}]:\n            xmin = {a:.4f}\n            xmax = {b:.4f}\n            text = "{l}"\n'
                       for i, ((a, b, _), l) in enumerate(zip(iv, labels), 1))
        return (f'    item [{{}}]:\n        class = "IntervalTier"\n        name = "{name}"\n        xmin = 0\n        xmax = {dur:.4f}\n'
                f"        intervals: size = {len(iv)}\n{rows}")
    items = [tier("sözcük", [l for *_, l in iv]), tier("duraklama", [""] * len(iv))]
    body = "".join(t.replace("item [{}]", f"item [{k}]", 1) for k, t in enumerate(items, 1))
    with open(path, "w", encoding="utf8", newline="\n") as f:
        f.write(f'File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0\nxmax = {dur:.4f}\ntiers? <exists>\nsize = 2\nitem []:\n{body}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir"); ap.add_argument("--sentences")
    ap.add_argument("--wav"); ap.add_argument("--text"); ap.add_argument("--out")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    al = Aligner(a.device)
    jobs = [(a.wav, a.text, a.out)] if a.wav else []
    if a.wav_dir:
        for r in csv.DictReader(open(a.sentences, encoding="utf8"), delimiter="\t"):
            p = os.path.join(a.wav_dir, f"{r['no']}.wav")
            if os.path.exists(p):
                jobs.append((p, r["cümle"], p[:-4] + ".TextGrid"))
    for wav, text, out in jobs:
        if os.path.exists(out):
            print(f"atlandı (var, üzerine yazılmaz): {out}")
            continue
        words, dur = al(wav, text)
        write_textgrid(out, intervals(words, dur), dur)
        print(f"{os.path.basename(out)}: {len(words)} sözcük, {dur:.1f} sn")


if __name__ == "__main__":
    main()
