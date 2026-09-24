"""Baseline (espeak-ng) sembol tablosu. Kaynak: reports/frontend_report.json `espeak_symbols` (1.053 klip, espeak-ng 1.52.0,
`tr`, with_stress=True, preserve_punctuation=True). Karakter düzeyinde: uzunluk `ː` ve vurgu `ˈ ˌ` ayrı semboller.
Not: espeak çıktısı vurgu işareti taşır, dizge fonemleri (Aşama 5) henüz taşımaz -> baseline'ın avantajı olabilir (bkz. config notu)."""
PAD = "_"
_CHARS = " !,.;?abcdefhijklmnoprstuvwyzæøœɔɛɟɡɪɫɯɾʃʊʒˈˌː"
SYMBOLS = [PAD, *_CHARS]
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
STRESS = ("ˈ", "ˌ")


def tokenize(espeak_str: str, strip_stress: bool = False, unknown: dict | None = None) -> list[str]:
    out = []
    for ch in espeak_str:
        if strip_stress and ch in STRESS:
            continue
        if ch in SYMBOL_TO_ID:
            out.append(ch)
        elif unknown is not None:
            unknown[ch] = unknown.get(ch, 0) + 1
    return out


assert len(SYMBOLS) == len(SYMBOL_TO_ID) == 47
