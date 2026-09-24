#!/bin/bash
# Yeni ep*.pt düştükçe (her run için) CPU'da sentezle + Whisper CER hesapla. Eğitimi yavaşlatmamak için CPU kullanır.
# Kullanım: nohup bash dizgetts/eval/auto_samples.sh > D:/dizgetts/cache/auto_samples.log 2>&1 &
PY=/d/dizgetts/venv/Scripts/python.exe
cd "$(dirname "$0")/../.."
while true; do
  for ck in /d/dizgetts/runs/*/ep*.pt; do
    [ -f "$ck" ] || continue
    run=$(basename "$(dirname "$ck")"); ep=$(basename "$ck" .pt); out="D:/dizgetts/samples/$run/$ep"
    if [ ! -f "$out/asr.json" ]; then
      sleep 5  # yazma bitsin
      $PY -X utf8 dizgetts/eval/synth.py --ckpt "D:/dizgetts/runs/$run/$ep.pt" --device cpu >/dev/null 2>&1 \
        && $PY -X utf8 dizgetts/eval/asr_check.py "$out" --device cpu 2>&1 | grep "TOPLAM" | sed "s/^/$run $ep /"
    fi
  done
  sleep 60
done
