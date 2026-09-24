"""4 GB VRAM smoke test: rastgele veriyle Matcha-TTS fp16 ileri/geri; EĞİTİM DEĞİL.
En kötü durum klibi (antalia ~20 sn) + out_size segment kırpma ile batch boyutlarını dener.

  D:/dizgetts/venv/Scripts/python.exe dizgetts/scripts/smoke_vram.py
"""
import os, sys, time

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from frontend.symbols import SYMBOLS  # noqa: E402
from matcha.models.matcha_tts import MatchaTTS  # noqa: E402

# Upstream configs/model/* varsayılanları (LJSpeech), n_vocab bizim tablodan
enc = OmegaConf.create(dict(
    encoder_type="RoPE Encoder",
    encoder_params=dict(n_feats=80, n_channels=192, filter_channels=768, filter_channels_dp=256, n_heads=2,
                        n_layers=6, kernel_size=3, p_dropout=0.1, spk_emb_dim=64, n_spks=1, prenet=True),
    duration_predictor_params=dict(filter_channels_dp=256, kernel_size=3, p_dropout=0.1),
))
dec = OmegaConf.create(dict(channels=[256, 256], dropout=0.05, attention_head_dim=64, n_blocks=1,
                            num_mid_blocks=2, num_heads=2, act_fn="snakebeta"))
cfm = OmegaConf.create(dict(name="CFM", solver="euler", sigma_min=1e-4))
stats = dict(mel_mean=-5.536622, mel_std=2.116101)


def run(bs, mel_frames, n_tok, out_size, amp=True):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = MatchaTTS(n_vocab=len(SYMBOLS), n_spks=1, spk_emb_dim=64, n_feats=80, encoder=enc, decoder=dec, cfm=cfm,
                  data_statistics=stats, out_size=out_size).cuda().train()
    opt = torch.optim.Adam(m.parameters(), 1e-4)
    scaler = torch.amp.GradScaler()
    x = torch.randint(2, len(SYMBOLS), (bs, n_tok), device="cuda")
    xl = torch.full((bs,), n_tok, device="cuda")
    y = torch.randn(bs, 80, mel_frames, device="cuda")
    yl = torch.full((bs,), mel_frames, device="cuda")
    t = time.time()
    y = y * 2.1 - 5.5  # ham mel istatistiğine yakın (model içeride normalize etmez; MAS ham y ister)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        dur, prior, diff, _ = m(x, xl, y, yl, out_size=out_size)
        loss = dur + prior + diff
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**30, time.time() - t, float(loss.detach())


if __name__ == "__main__":
    # BULGU: GTX 1650 (TU117, sm_75) + cuDNN 9.10 (torch 2.8+cu128): fp16 Conv1d(768->192) NaN üretiyor;
    # cuDNN kapalıyken doğru (bkz. reports/stage1_audit.md). Bu yüzden fp16 modunda cuDNN kapatılır.
    frames, ntok = 1740, 600  # 20.2 sn @22.05kHz/hop256; ~300 fonem*2+1 (add_blank)
    for label, amp, cudnn in (("fp16+cudnn KAPALI", True, False), ("fp16+cudnn AÇIK", True, True), ("fp32+cudnn AÇIK", False, True)):
        torch.backends.cudnn.enabled = cudnn
        print("==", label, flush=True)
        for out_size in (None, 172):
            for bs in (2, 4, 8, 16):
                try:
                    run(bs, frames, ntok, out_size, amp)  # ısınma
                    gb, sec, loss = run(bs, frames, ntok, out_size, amp)
                    print(f"  out_size={out_size} bs={bs}: peak {gb:.2f} GiB, {sec:.2f}s/adım, loss {loss:.3f}", flush=True)
                except torch.cuda.OutOfMemoryError:
                    print(f"  out_size={out_size} bs={bs}: OOM", flush=True)
                    break
