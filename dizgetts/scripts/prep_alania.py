"""Ön eğitim verisi: cloud0day3/alania-synthetic-speech-tr (CC BY 4.0, tamamı SENTETİK; `cc-by` bölümü) -> Antalia ile AYNI ses işleme (preprocess.process).

  python -X utf8 -m dizgetts.scripts.prep_alania [--hours 20]
Parçalar tek tek indirilir, süzülür, işlenir ve SİLİNİR (arşiv 540 GB; süzgeçten parça başına ~0,4 sa geçer).
Çıktı: <out>/wavs/*.wav (22,05 kHz), <out>/{train,val,test}_phon.jsonl (id, wav, dur, text, voice), <out>/stats.json. Sonra: build_manifest --data-config data_alania.yaml ...

Süzgeç (experiments.yaml v10-pre): konuşmacı koşullaması YOK (n_spks=1, mimari değişmesin) -> sesler havuzlanır, bu yüzden havuz daraltılır:
kadın sesler (Antalia okuyucusu), düz okuma (`reference`), 2-15 sn, qc_utmos >= 3,2, qc_cer <= 0,02, konuşma hızı Antalia'nın %5-%95 bandında (12,4-16,9 karakter/sn).
Metin = `text_spoken` (sesin gerçekte okuduğu açılmış biçim: TBMM -> tebememe). Mel istatistiği ANTALIA'nınki (ön eğitim ve ince ayar aynı mel uzayında).
ponytail: bölme SES kimliğinin özetine göre (~%2 val, ~%2 test); kıyas kümesi değil.
"""
import argparse, hashlib, io, json, os, random

import pyarrow.parquet as pq
import soundfile as sf
import yaml
from huggingface_hub import hf_hub_download

from dizgetts import paths
from dizgetts.scripts.preprocess import process

HERE = os.path.join(os.path.dirname(__file__), "..")
REPO, N_SHARDS = "cloud0day3/alania-synthetic-speech-tr", 654  # data/cc-by/train-00000 .. 00653 (son parça 00654 küçük, kapı ölçümünde kullanıldı)
COLS = ["id", "audio", "text_spoken", "duration", "voice_id", "gender", "render_mode", "qc_utmos", "qc_cer"]
RATE = (12.4, 16.9)  # Antalia train: karakter/sn %5-%95 (text_norm / dur)


def keep(r: dict) -> bool:
    return (r["gender"] == "female" and r["render_mode"] == "reference" and r["voice_id"] is not None and 2.0 <= r["duration"] <= 15.0
            and r["qc_utmos"] >= 3.2 and r["qc_cer"] <= 0.02 and RATE[0] <= len(r["text_spoken"]) / r["duration"] <= RATE[1])


def split_of(voice: str) -> str:
    h = int(hashlib.sha1(voice.encode("utf8")).hexdigest(), 16) % 50
    return "val" if h == 0 else "test" if h == 1 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{paths.HOME}/data/processed/alania")
    ap.add_argument("--hours", type=float, default=20.0, help="işlenmiş toplam saat hedefi; dolunca durur")
    ap.add_argument("--seed", type=int, default=1234, help="parça sırası (ses çeşitliliği için karışık)")
    a = ap.parse_args()
    c = yaml.safe_load(open(os.path.join(HERE, "configs", "data.yaml"), encoding="utf8"))  # ses ayarı Antalia ile aynı
    os.makedirs(os.path.join(a.out, "wavs"), exist_ok=True)
    tmp = os.path.join(a.out, "_shard")
    order = list(range(N_SHARDS)); random.Random(a.seed).shuffle(order)
    rows, sec, seen, used = {"train": [], "val": [], "test": []}, 0.0, 0, []
    for n in order:
        p = hf_hub_download(REPO, f"data/cc-by/train-{n:05d}.parquet", repo_type="dataset", local_dir=tmp)
        pf = pq.ParquetFile(p)
        for g in range(pf.num_row_groups):
            for r in pf.read_row_group(g, columns=COLS).to_pylist():
                seen += 1
                if not keep(r):
                    continue
                x, sr, _ = process(io.BytesIO(r["audio"]["bytes"]), c)
                sf.write(os.path.join(a.out, "wavs", r["id"] + ".wav"), x, sr, subtype="PCM_16")
                rows[split_of(r["voice_id"])].append(dict(id=r["id"], wav=f"wavs/{r['id']}.wav", dur=round(len(x) / sr, 3), text=r["text_spoken"], voice=r["voice_id"]))
                sec += len(x) / sr
        del pf
        os.remove(p)
        used.append(n)
        print(f"parça {n:05d} ({len(used)}): toplam {sec / 3600:.2f} sa, {sum(map(len, rows.values()))} klip, {len({r['voice'] for rs in rows.values() for r in rs})} ses", flush=True)
        for sp, rs in rows.items():  # her parçadan sonra yaz: yarıda kesilirse eldeki kullanılabilir
            with open(os.path.join(a.out, f"{sp}_phon.jsonl"), "w", encoding="utf8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rs)
        if sec >= a.hours * 3600:
            break
    ant = json.load(open(os.path.join(paths.ANTALIA, "stats.json"), encoding="utf8"))
    hours = {sp: round(sum(r["dur"] for r in rs) / 3600, 3) for sp, rs in rows.items()}
    json.dump(dict(n={sp: len(rs) for sp, rs in rows.items()}, hours=hours, seen=seen, shards=used, voices=len({r["voice"] for rs in rows.values() for r in rs}),
                   mel_mean=ant["mel_mean"], mel_std=ant["mel_std"], mel_stats_from="antalia", source=f"{REPO} cc-by (CC BY 4.0, PatientDesk AI; sentetik)"),
              open(os.path.join(a.out, "stats.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print("->", a.out, hours)


if __name__ == "__main__":
    main()
