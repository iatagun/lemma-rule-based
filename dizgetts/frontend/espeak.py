"""espeak-ng (baseline, Aşama 4). Windows'ta kurulum gerektirmez: `espeakng-loader` DLL + veriyi getirir.
Aynı normalize() çıktısı üzerinde çalışır ki baseline ile ana model yalnız fonem kaynağında ayrışsın."""
from __future__ import annotations

_backend = None


def backend():
    global _backend
    if _backend is None:
        import espeakng_loader
        from phonemizer.backend import EspeakBackend
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        _backend = EspeakBackend("tr", preserve_punctuation=True, with_stress=True, language_switch="remove-flags")
    return _backend


def espeak_version() -> str:
    return str(backend().version())


def phonemize(norm_text: str) -> str:
    return backend().phonemize([norm_text], strip=True)[0]
