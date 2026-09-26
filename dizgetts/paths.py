"""Ağır dosyaların (veri, koşular, değerlendirme çıktıları) kökü. Varsayılan D:/dizgetts; başka makinede DIZGETTS_HOME ile değiştirilir.
configs/*.yaml kendi yollarını (raw_root, out_root, run_root, init.ckpt) taşır; başka makinede onlar da güncellenmeli."""
import os

HOME = os.environ.get("DIZGETTS_HOME", "D:/dizgetts")
ANTALIA = f"{HOME}/data/processed/antalia"   # preprocess.py çıktısı (configs/data.yaml out_root ile aynı olmalı)
G2PTTS_DATA = f"{HOME}/data/g2ptts"
RUNS = f"{HOME}/runs"
EVAL_OUT = f"{HOME}/eval_out"
SAMPLES = f"{HOME}/samples"
AB, AB_KEYS = f"{HOME}/ab", f"{HOME}/ab_keys"
VOCODER = f"{HOME}/pretrained/hifigan_univ_v1"
