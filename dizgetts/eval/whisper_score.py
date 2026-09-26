"""Whisper ile HİPOTEZ PUANLAMA: aynı ses için birkaç aday metnin (öğretmen-zorlamalı) NLL'ini karşılaştırır.
Normalizer kararlarını (para okunuşu, SMS, www, uzun numara...) sesten bağımsız kanıtla seçmek için.
Kaba: Whisper'ın dil modeli önyargısı her iki adayda da vardır; yalnız AYNI biçimli adaylar kıyaslanır, küçük fark kanıt sayılmaz.

  from dizgetts.eval.whisper_score import Scorer; s = Scorer(); s.nll(wav_path, ["metin a", "metin b"])
"""
import os, sys
import warnings

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL = "openai/whisper-small"


class Scorer:
    def __init__(self, model=MODEL, device=None):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.proc = WhisperProcessor.from_pretrained(model)
        self.model = WhisperForConditionalGeneration.from_pretrained(model, torch_dtype=torch.float32).to(self.dev).eval()  # fp32: GTX 1650 fp16 cuDNN NaN
        tok = self.proc.tokenizer
        self.prefix = tok.convert_tokens_to_ids(["<|startoftranscript|>", "<|tr|>", "<|transcribe|>", "<|notimestamps|>"])
        self.eot = tok.convert_tokens_to_ids("<|endoftext|>")
        self.tok = tok

    def _feats(self, wav):
        x, sr = sf.read(wav, dtype="float32")
        if len(x) / sr > 30:  # Whisper öznitelik çıkarıcısı 30 sn'de SESSİZCE keser -> sentezin sonu transkripte girmez, CER şişer
            warnings.warn(f"{wav}: {len(x) / sr:.1f} sn > 30 sn; Whisper yalnız ilk 30 saniyeyi işler (CER/WER güvenilmez)", RuntimeWarning, stacklevel=2)
        if sr != 16000:
            g = np.gcd(16000, sr)
            x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
        return self.proc.feature_extractor(x, sampling_rate=16000, return_tensors="pt").input_features.to(self.dev)

    @torch.inference_mode()
    def nll(self, wav, hyps):
        """Her aday için toplam NLL (nat) ve token sayısı. Ses 30 sn altı olmalı."""
        f = self._feats(wav)
        enc = self.model.model.encoder(f).last_hidden_state
        out = []
        for h in hyps:
            ids = self.prefix + self.tok.encode(" " + h, add_special_tokens=False) + [self.eot]
            t = torch.tensor([ids], device=self.dev)
            logits = self.model(encoder_outputs=(enc,), decoder_input_ids=t[:, :-1]).logits
            lp = torch.log_softmax(logits.float(), -1)[0, len(self.prefix) - 1:, :]
            tgt = t[0, len(self.prefix):]
            out.append((-lp[torch.arange(len(tgt)), tgt].sum().item(), len(tgt)))
        return out

    @torch.inference_mode()
    def transcribe(self, wav):
        f = self._feats(wav)
        ids = self.model.generate(f, language="tr", task="transcribe", max_new_tokens=400)
        return self.tok.batch_decode(ids, skip_special_tokens=True)[0].strip()
