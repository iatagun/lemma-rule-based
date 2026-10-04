"""Ön eğitim verisi: risaleinur/risalei-nur-text-audio (CC BY 4.0) parquet parçalarından tek okuyuculu alt küme -> Antalia ile AYNI ses işleme (preprocess.process).

  python -X utf8 -m dizgetts.scripts.prep_risale --shards "D:/dizgetts/data/raw/hf_gate/risalei-nur-text-audio/audio-text/train-000*.parquet" [--hours 20]
Çıktı: <out>/wavs/*.wav (22,05 kHz), <out>/{train,val,test}_phon.jsonl (id, wav, dur, text), <out>/stats.json. Sonra: build_manifest --data-config configs/data_risale.yaml ...

Süzgeç (experiments.yaml v10-pre): okuyucu A.Köseoğlu, Türkçe, kaynakla birebir eşleşen metin, 3-15 sn, rakamsız, sözcük içi tiresiz (izafet `rağbet-i âmme`:
normalize tireyi boşluğa çevirir, `i` ayrı sözcük olur). Mel istatistiği ANTALIA'nınki: ön eğitim ve ince ayar aynı normalize mel uzayında kalsın.
ponytail: bölme kaynak kaydın özetine göre (~%2 val, ~%2 test); metin sızıntısı denetimi yok (ön eğitim, kıyas kümesi değil).
"""
import argparse, glob, hashlib, io, json, os, re

import pyarrow.parquet as pq
import soundfile as sf
import yaml

from dizgetts import paths
from dizgetts.scripts.preprocess import process

HERE = os.path.join(os.path.dirname(__file__), "..")
HYPHEN = re.compile(r"[^\W\d_]-[^\W\d_]")
COLS = ["audio", "text", "reader", "language", "duration", "record_id", "source_audio_id", "match_method"]


def keep(r: dict, reader: str) -> bool:
    return (r["language"] == "tr" and r["reader"] == reader and r["match_method"] == "source_exact" and 3.0 <= r["duration"] <= 15.0
            and not HYPHEN.search(r["text"]) and not any(c.isdigit() for c in r["text"]))


def split_of(source: str) -> str:
    h = int(hashlib.sha1(source.encode("utf8")).hexdigest(), 16) % 50
    return "val" if h == 0 else "test" if h == 1 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="parquet glob")
    ap.add_argument("--out", default=f"{paths.HOME}/data/processed/risale")
    ap.add_argument("--reader", default="A.Köseoğlu")
    ap.add_argument("--hours", type=float, default=20.0, help="işlenmiş (kırpılmış) toplam saat hedefi; dolunca durur")
    a = ap.parse_args()
    c = yaml.safe_load(open(os.path.join(HERE, "configs", "data.yaml"), encoding="utf8"))  # ses ayarı Antalia ile aynı
    os.makedirs(os.path.join(a.out, "wavs"), exist_ok=True)
    rows, sec, seen, dropped = {"train": [], "val": [], "test": []}, 0.0, 0, 0
    for path in sorted(glob.glob(a.shards)):
        pf = pq.ParquetFile(path)
        for g in range(pf.num_row_groups):
            for r in pf.read_row_group(g, columns=COLS).to_pylist():
                seen += 1
                if not keep(r, a.reader):
                    dropped += 1
                    continue
                x, sr, _ = process(io.BytesIO(r["audio"]["bytes"]), c)
                cid = r["record_id"].replace("/", "_")
                sf.write(os.path.join(a.out, "wavs", cid + ".wav"), x, sr, subtype="PCM_16")
                rows[split_of(r["source_audio_id"])].append(dict(id=cid, wav=f"wavs/{cid}.wav", dur=round(len(x) / sr, 3), text=r["text"], source=r["source_audio_id"]))
                sec += len(x) / sr
            if sec >= a.hours * 3600:
                break
        print(f"{os.path.basename(path)}: toplam {sec / 3600:.1f} sa, {sum(map(len, rows.values()))} klip", flush=True)
        if sec >= a.hours * 3600:
            break
    for sp, rs in rows.items():
        with open(os.path.join(a.out, f"{sp}_phon.jsonl"), "w", encoding="utf8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ant = json.load(open(os.path.join(paths.ANTALIA, "stats.json"), encoding="utf8"))
    hours = {sp: round(sum(r["dur"] for r in rs) / 3600, 3) for sp, rs in rows.items()}
    json.dump(dict(n={sp: len(rs) for sp, rs in rows.items()}, hours=hours, seen=seen, dropped_by_filter=dropped, reader=a.reader,
                   mel_mean=ant["mel_mean"], mel_std=ant["mel_std"], mel_stats_from="antalia", source="risaleinur/risalei-nur-text-audio (CC BY 4.0)"),
              open(os.path.join(a.out, "stats.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print("->", a.out, hours)


if __name__ == "__main__":
    main()
