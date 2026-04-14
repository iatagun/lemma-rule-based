"""
Türkçe sözlük modülü – sözcük listesi yükleme ve kök çözümleme.

SOLID:
  SRP – Yalnızca sözlük verisi ve morfofonemik kök çözümlemeden sorumlu.
  OCP – Yeni morfofonemik kurallar _TRANSFORMS listesine eklenebilir.
  DIP – Çözümleyici bu sınıfa doğrudan değil, is_valid arayüzü
        üzerinden bağımlıdır.
"""

from __future__ import annotations

from pathlib import Path

from .phonology import CONSONANTS, VOWELS


# ── Morfofonemik Sabitler ─────────────────────────────────────

# Ünsüz yumuşaması tersine çevirme tablosu (yüzey → sözlük)
_SOFTENING_REVERSE: dict[str, str] = {
    "b": "p",  # kitab → kitap
    "c": "ç",  # ağac  → ağaç
    "d": "t",  # kanad → kanat
    "g": "k",  # reng  → renk
    "ğ": "k",  # çocuğ → çocuk
}

# Ünlü uyumuna göre düşen ünlüyü geri ekleme tablosu
_VOWEL_HARMONY_INSERT: dict[str, str] = {
    "a": "ı",
    "ı": "ı",
    "e": "i",
    "i": "i",
    "o": "u",
    "u": "u",
    "ö": "ü",
    "ü": "ü",
}

# Ünlü daralması tersine çevirme (diyor → de+yor)
_NARROWING_REVERSE: dict[str, str] = {
    "ı": "a",
    "i": "e",
    "u": "o",
    "ü": "ö",
}

# Kaynaştırma (yardımcı) ünsüzleri
# y: ünlüyle biten köklere ünlü eki geldiğinde (su+y+u, ne+y+i)
# n: işaret zamirleri ve bazı sözcüklerde (o+n+un, bu+n+un)
# s: 3. tekil iyelik ünlüyle biten köklerde (araba+s+ı, hava+s+ı)
_BUFFER_CONSONANTS: frozenset[str] = frozenset("yns")


class TurkishDictionary:
    """
    Türkçe sözcük listesi tabanlı sözlük.

    Sözcük listesi doğrudan kök sözlüğü değildir — TDK lemma/madde başı
    formatındadır. Bu nedenle fiiller mastar biçiminde (-mak/-mek),
    türemiş sözcükler (-lık, -cı, -sız vb.) sözlükte bulunur.
    """

    def __init__(self, words: frozenset[str]) -> None:
        self._words = words

    @classmethod
    def from_file(cls, path: str | Path) -> TurkishDictionary:
        """Dosyadan sözcük listesini yükler."""
        with open(path, encoding="utf-8") as f:
            words = frozenset(
                line.strip().lower() for line in f if line.strip()
            )
        return cls(words)

    def __len__(self) -> int:
        return len(self._words)

    def contains(self, word: str) -> bool:
        """Sözcük sözlükte var mı?"""
        return word.lower() in self._words

    def find_root(self, stem: str) -> str | None:
        """
        Kök adayının sözlük biçimini bulmaya çalışır.

        Sırasıyla dener:
          1. Doğrudan eşleşme (isim/sıfat)
          2. Fiil kökü (kök + mak/mek)
          3. Ünsüz yumuşaması tersine çevirme (kitab → kitap)
          4. Ünlü düşmesi tersine çevirme (burn → burun)
          5. Kaynaştırma harfi kaldırma (suy → su, bun → bu)

        Returns:
            Sözlük biçimi veya None.
        """
        stem = stem.lower()

        # 1. Doğrudan eşleşme
        if stem in self._words:
            return stem

        # 2. Fiil kökü kontrolü (gelmek, yazmak vb.)
        for infinitive in ("mak", "mek"):
            if stem + infinitive in self._words:
                return stem

        # 3. Ünsüz yumuşaması tersine çevirme
        hardened = self._try_harden(stem)
        if hardened is not None:
            return hardened

        # 4. Ünlü düşmesi tersine çevirme
        restored = self._try_vowel_restore(stem)
        if restored is not None:
            return restored

        # 5. Kaynaştırma harfi kaldırma (suy→su, bun→bu)
        buffer_removed = self._try_buffer_remove(stem)
        if buffer_removed is not None:
            return buffer_removed

        return None

    def find_root_with_narrowing(self, stem: str) -> str | None:
        """
        Ünlü daralması (diyor → de+yor) dahil kök çözümleme.
        Yalnızca -yor eki öncesi kısa kökler için kullanılır.
        """
        result = self.find_root(stem)
        if result is not None:
            return result

        # Ünlü daralması: di → de, yi → ye
        if len(stem) >= 1:
            last_ch = stem[-1]
            if last_ch in _NARROWING_REVERSE:
                widened = stem[:-1] + _NARROWING_REVERSE[last_ch]
                root = self.find_root(widened)
                if root is not None:
                    return root

        return None

    # ── Dahili Morfofonemik Dönüşümler ────────────────────────

    def _try_harden(self, stem: str) -> str | None:
        """Ünsüz yumuşaması tersine: kitab→kitap, reng→renk."""
        if not stem or stem[-1] not in _SOFTENING_REVERSE:
            return None

        hardened = stem[:-1] + _SOFTENING_REVERSE[stem[-1]]

        # İsim/sıfat kontrolü
        if hardened in self._words:
            return hardened

        # Fiil kökü kontrolü
        for infinitive in ("mak", "mek"):
            if hardened + infinitive in self._words:
                return hardened

        return None

    def _try_vowel_restore(self, stem: str) -> str | None:
        """
        Ünlü düşmesi tersine çevirme: burn→burun, oğl→oğul, gönl→gönül.

        Son iki karakterin ikisi de ünsüzse, araya uyumlu ünlü ekler
        ve sözlükte arar.
        """
        if len(stem) < 2:
            return None
        if stem[-1] not in CONSONANTS or stem[-2] not in CONSONANTS:
            return None

        # Önceki son ünlüyü bul
        last_v = None
        for ch in stem[:-2]:
            if ch in VOWELS:
                last_v = ch
        if last_v is None:
            return None

        insert_v = _VOWEL_HARMONY_INSERT.get(last_v)
        if insert_v is None:
            return None

        restored = stem[:-1] + insert_v + stem[-1]
        if restored in self._words:
            return restored

        return None

    def _try_buffer_remove(self, stem: str) -> str | None:
        """
        Kaynaştırma harfi kaldırma: suy→su, bun→bu.

        Ünlüyle biten köklere ünlüyle başlayan ek geldiğinde araya
        giren y/n ünsüzü tanınır ve kaldırılır.
          y: su+y+u, ne+y+i, arı+y+ı
          n: o+n+un, bu+n+un, şu+n+un
        """
        if len(stem) < 2:
            return None
        if stem[-1] not in _BUFFER_CONSONANTS:
            return None
        if stem[-2] not in VOWELS:
            return None

        base = stem[:-1]

        if base in self._words:
            return base

        for infinitive in ("mak", "mek"):
            if base + infinitive in self._words:
                return base

        return None
