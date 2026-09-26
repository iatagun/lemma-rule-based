"""Metin normalizasyonu: ham Türkçe metin -> yalnız harf, boşluk ve duraklama işareti (, . ? ! ;).

Sayılar/tarih/saat/yüzde/para/birim/kısaltma/harfle okunan kısaltma/kod numarası çözülür. Kesme işaretli ekler ("3'te" -> "üçte")
sayının OKUNUŞUNA eklenir ve okunuşa göre yeniden uyumlandırılır (kaynak metinde "79'i" gibi yanlış ekler var; Whisper puanlaması
58/58 klipte uyumlandırılmış biçimi seçti). Kararların kanıtı: scripts/normalizer_decisions.py -> reports/normalizer_decisions.json.
Bağımlılık yok (torch yok). Referans: antalia `normalized_transcript` (scripts/build_phonemes.py karşılaştırır).

ponytail: kural tabanlı ve kasten dar. Bilinmeyen kısaltma/yabancı sözcük LEXICON'a eklenir; bağlama duyarlı (eş yazımlı,
ordinal-nokta belirsizliği) çözümler yok. Büyük harfli 2-4 harflik sözcük harf harf okunur (WORD_ACRONYMS istisna).
"""
import re
import unicodedata

ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
SCALES = [(10**12, "trilyon"), (10**9, "milyar"), (10**6, "milyon"), (1000, "bin")]
MONTHS = ["ocak", "şubat", "mart", "nisan", "mayıs", "haziran", "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık"]
LETTERS = {  # Türkçe alfabe harf adları (antalia referansı: USB -> "u se be", PDF -> "pe de fe")
    "a": "a", "b": "be", "c": "ce", "ç": "çe", "d": "de", "e": "e", "f": "fe", "g": "ge", "ğ": "yumuşak ge", "h": "he",
    "ı": "ı", "i": "i", "j": "je", "k": "ke", "l": "le", "m": "me", "n": "ne", "o": "o", "ö": "ö", "p": "pe", "r": "re",
    "s": "se", "ş": "şe", "t": "te", "u": "u", "ü": "ü", "v": "ve", "y": "ye", "z": "ze", "w": "çift ve", "x": "iks", "q": "kü",
}
SQUARE = {"km": "kilometre", "m": "metre", "cm": "santimetre", "mm": "milimetre"}  # m² / m³ (Unicode üst simge)
UNITS = {"GB": "gigabayt", "MB": "megabayt", "KB": "kilobayt", "TB": "terabayt", "km": "kilometre", "kg": "kilogram",
         "cm": "santimetre", "mm": "milimetre", "dk": "dakika", "sn": "saniye", "°C": "derece", "TL": "lira", "₺": "lira"}
# Sözlük: harfle okunmayacak / özel okunacak (tümü küçük harf anahtar). Genişletmek serbest.
LEXICON = {"wi-fi": "vayfay", "wifi": "vayfay", "www": "vi vi vi",  # Whisper hipotez puanlaması: 13/13 klipte "vi vi vi" > "ve ve ve" (reports/normalizer_decisions.json); kulakla doğrulanmadı
           "sms": "es em es", "pin": "pin", "nato": "nato", "nasa": "nasa", "gb": "gigabayt", "mb": "megabayt", "tl": "lira"}
ABBR = {"dr": "doktor", "prof": "profesör", "doç": "doçent", "sn": "sayın", "vb": "ve benzeri", "vs": "ve saire", "örn": "örneğin",
        "yy": "yüzyıl", "mah": "mahallesi", "cad": "caddesi", "no": "numara", "tel": "telefon", "av": "avukat", "bkz": "bakınız"}
SYMBOLS = {"&": " ve ", "+": " artı ", "=": " eşittir ", "@": " et ", "#": " diyez ", "/": " ", "\\": " "}
ORD = {"bir": "birinci", "iki": "ikinci", "üç": "üçüncü", "dört": "dördüncü", "beş": "beşinci", "altı": "altıncı", "yedi": "yedinci",
       "sekiz": "sekizinci", "dokuz": "dokuzuncu", "on": "onuncu", "yirmi": "yirminci", "otuz": "otuzuncu", "kırk": "kırkıncı",
       "elli": "ellinci", "altmış": "altmışıncı", "yetmiş": "yetmişinci", "seksen": "sekseninci", "doksan": "doksanıncı",
       "yüz": "yüzüncü", "bin": "bininci", "milyon": "milyonuncu", "milyar": "milyarıncı", "sıfır": "sıfırıncı"}
UP = "A-ZÇĞİÖŞÜ"
LOW = "a-zçğıöşü"
PUNCT_RANK = {",": 1, ";": 2, ".": 3, "!": 4, "?": 5}  # ardışık işaretlerde en güçlüsü kalır


def tr_lower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()


def _below1000(n: int) -> list[str]:
    h, t, o = n // 100, n % 100 // 10, n % 10
    w = []
    if h:
        w += ["yüz"] if h == 1 else [ONES[h], "yüz"]
    if t:
        w.append(TENS[t])
    if o:
        w.append(ONES[o])
    return w


def cardinal(n: int) -> str:
    if n == 0:
        return "sıfır"
    if n >= 10**15:
        return digits(str(n))
    w = []
    for v, name in SCALES:
        q, n = divmod(n, v)
        if q:
            w += ["bin"] if (v == 1000 and q == 1) else _below1000(q) + [name]
    return " ".join(w + _below1000(n))


def digits(s: str) -> str:
    return " ".join("sıfır" if ch == "0" else ONES[int(ch)] for ch in s)


def number_words(s: str) -> str:
    """Tam sayı dizgesi -> okunuş. Baştaki sıfırlı ya da 15+ haneli diziler rakam rakam okunur."""
    return digits(s) if (len(s) > 1 and s[0] == "0") or len(s) > 15 else cardinal(int(s))


VOWELS, BACK, ROUND = "aeıioöuü", "aıou", "oöuü"


def _harmonize(word: str, suffix: str, buffer: str) -> str:
    """Ek, sayının OKUNUŞUNA göre yeniden uyumlandırılır: kaynak metinde ek yanlış yazılabiliyor (79'i -> dokuzu, 47'i -> yedisi).
    Ünlü uyumu (2'li a/e, 4'lü ı/i/u/ü), d/t ve c/ç ünsüz benzeşmesi, ünlü+ünlü arası kaynaştırma ünsüzü (buffer)."""
    prev = next((c for c in reversed(word) if c in VOWELS), "a")
    out = []
    for i, ch in enumerate(suffix):
        if ch in "aeıiuü":
            if i == 0 and word[-1] in VOWELS and buffer:
                out.append(buffer)
            back, rnd = prev in BACK, prev in ROUND
            ch = ("a" if back else "e") if ch in "ae" else (("u" if rnd else "ı") if back else ("ü" if rnd else "i"))
        if ch in VOWELS:
            prev = ch
        out.append(ch)
    s = "".join(out)
    if s[0] in "dtcç":  # sert ünsüzden sonra t/ç, aksi halde d/c
        hard = word[-1] in "çfhkpsşt"
        s = {"d": "t" if hard else "d", "t": "t" if hard else "d", "c": "ç" if hard else "c", "ç": "ç" if hard else "c"}[s[0]] + s[1:]
    return s


def attach(words: str, suffix: str | None, buffer: str = "y") -> str:
    """Kesme işaretli eki okunuşun sonuna yapıştırır (harmonize eder). dört + ünlü -> dörd."""
    if not suffix:
        return words
    suffix = tr_lower(suffix)
    if suffix[0] in "yns" and len(suffix) > 1 and suffix[1] in VOWELS:  # yazar kaynaştırmayı zaten koymuş (yi, nin, si)
        head, rest = suffix[0], suffix[1:]
        return words + head + _harmonize(words + head, rest, "")
    if words.endswith("dört") and suffix[0] in VOWELS:
        words = words[:-1] + "d"  # dört + ü -> dördü
    return words + _harmonize(words, suffix, buffer)


def decimal_words(int_s: str, frac_s: str) -> str:
    return f"{number_words(int_s)} virgül {number_words(frac_s)}"


SUF = r"(?:['’]([^\W\d_]+))?"
NUM = r"\d+(?:,\d+)?"


def _num(s: str) -> str:
    return decimal_words(*s.split(",")) if "," in s else number_words(s)


def _pairs(s: str) -> str:
    """Rakam dizisi ikişer okunur ("3254" -> "otuz iki elli dört"; "07" -> "sıfır yedi"; tek kalan rakam sona)."""
    return " ".join(digits(g) if g[0] == "0" else cardinal(int(g)) for g in (s[i:i + 2] for i in range(0, len(s), 2)))


def _sub_codes(t):
    """Kod/numara okunuşu (Whisper hipotez puanlaması, reports/normalizer_decisions.json): PPV-3254-K klipleri 64/65 ikişer
    okunuyor; 7+ haneli numara 1/1 ikişer (az kanıt). Yıl aralıkları (1990-2000) hariç."""
    up = "[A-ZÇĞİÖŞÜ]"
    t = re.sub(r"(?<![\d.,])\d{7,}(?![\d.,])", lambda m: _pairs(m.group()), t)
    t = re.sub(rf"(?<={up}-)\d{{4,6}}(?![\d.,])|(?<![\d.,])\d{{4,6}}(?=-{up}\b)",
               lambda m: _pairs(m.group()), t)
    return re.sub(r"(?<![\d.,-])(\d{4})-(\d{4})(?![\d.,-])",
                  lambda m: m.group() if m.group(1)[:2] in ("19", "20") and m.group(2)[:2] in ("19", "20") else _pairs(m.group(1)) + " " + _pairs(m.group(2)), t)


def _sub_thousands(t):
    return re.sub(r"(?<![\d.,])\d{1,3}(?:\.\d{3})+(?![\d])", lambda m: m.group().replace(".", ""), t)


def _sub_dates(t):
    def f(m):
        d, mo, y, suf = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        if not 1 <= mo <= 12 or not 1 <= int(d) <= 31:
            return m.group()
        return attach(f"{cardinal(int(d))} {MONTHS[mo - 1]} {cardinal(int(y))}", suf)
    return re.sub(rf"\b(\d{{1,2}})[./](\d{{1,2}})[./](\d{{4}})\b{SUF}", f, t)


def _sub_times(t):
    def f(m):
        h, mi, suf = int(m.group(1)), m.group(2), m.group(3)
        w = cardinal(h) + ("" if mi == "00" else " " + (digits(mi[0]) + " " + cardinal(int(mi[1])) if mi[0] == "0" else cardinal(int(mi))))
        return attach(w, suf)
    return re.sub(rf"\b([01]?\d|2[0-3])[.:]([0-5]\d)\b{SUF}", f, t)


def _sub_percent(t):
    t = re.sub(rf"%\s?({NUM}){SUF}", lambda m: attach("yüzde " + _num(m.group(1)), m.group(2), "s"), t)
    return re.sub(rf"({NUM})\s?%{SUF}", lambda m: attach("yüzde " + _num(m.group(1)), m.group(2), "s"), t)


def _sub_currency(t):
    names = {"TL": "lira", "₺": "lira", "USD": "dolar", "$": "dolar", "EUR": "euro", "€": "euro"}
    cur = r"(TL|₺|USD|\$|EUR|€)"

    def f(whole, frac, c, suf):
        name = names[c]
        w = f"{number_words(whole)} {name}"
        if frac and int(frac.ljust(2, "0")) > 0:
            w += f" {cardinal(int(frac.ljust(2, '0')))} " + ("kuruş" if name == "lira" else "sent")
        return attach(w, suf)
    t = re.sub(rf"\b(\d+)(?:,(\d{{1,2}}))?\s?{cur}(?![A-Za-z]){SUF}", lambda m: f(m.group(1), m.group(2), m.group(3), m.group(4)), t)
    return re.sub(rf"{cur}\s?(\d+)(?:,(\d{{1,2}}))?\b{SUF}", lambda m: f(m.group(2), m.group(3), m.group(1), m.group(4)), t)


def _sub_units(t):
    unit_re = "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))
    return re.sub(rf"({NUM})\s?({unit_re})(?![A-Za-zÇĞİÖŞÜçğıöşü]){SUF}", lambda m: f"{_num(m.group(1))} " + attach(UNITS[m.group(2)], m.group(3)), t)


def _sub_ordinals(t):
    def f(m):
        w = number_words(m.group(1)).split()
        w[-1] = ORD.get(w[-1], w[-1])
        return " ".join(w)
    return re.sub(rf"\b(\d{{1,3}})\.(?=\s+[{LOW}])", f, t)


def _sub_numbers(t):
    poss = lambda m: "s" if re.search(r"yüzde\s*$", m.string[: m.start()], re.I) else "y"  # "yüzde 47'si" iyelik; diğer: belirtme (-yi)
    t = re.sub(rf"\b(\d+),(\d+)\b{SUF}", lambda m: attach(decimal_words(m.group(1), m.group(2)), m.group(3), poss(m)), t)
    return re.sub(rf"\b(\d+)\b{SUF}", lambda m: attach(number_words(m.group(1)), m.group(2), poss(m)), t)


def _sub_squares(t):
    return re.sub(r"(?<![^\W\d_])(km|cm|mm|m)([²³])", lambda m: SQUARE[m.group(1)] + ("kare" if m.group(2) == "²" else "küp"), t)


def _sub_x(t):
    return re.sub(r"(?<=\d)\s?[x×]\s?(?=\d)", " çarpı ", t)  # 55x40x23 -> 55 çarpı 40 çarpı 23


def _split_alnum(t):
    return re.sub(r"(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)", " ", t)  # VKT7M2 -> VKT 7 M 2 (rakamlar kaybolmasın)


def _sub_abbr(t):
    t = re.sub(rf"\bA\.\s?Ş\.{SUF}", lambda m: "anonim şirketi" + (tr_lower(m.group(1)) if m.group(1) else ""), t)
    t = re.sub(rf"\bLtd\.\s?Şti\.{SUF}", lambda m: "limited şirketi" + (tr_lower(m.group(1)) if m.group(1) else ""), t)
    return re.sub(rf"\b({'|'.join(ABBR)})\.(?=\s|$)", lambda m: ABBR[tr_lower(m.group(1))], t, flags=re.I)


def _spell(s: str) -> str:
    return " ".join(LETTERS.get(tr_lower(ch), ch) for ch in s)


def _sub_letters(t):
    # sözlük (wi-fi, www, sms...) önce; sonra 2-4 harfli BÜYÜK sözcük -> harf harf; koddaki tek büyük harf (PPV-3254-K)
    lex = "|".join(sorted(map(re.escape, LEXICON), key=len, reverse=True))
    t = re.sub(rf"(?<![\w-])({lex})(?![\w-])", lambda m: LEXICON[m.group(1).lower()], t, flags=re.I)
    t = re.sub(rf"\b[{UP}]{{2,4}}\b", lambda m: LEXICON.get(m.group().lower()) or _spell(m.group()), t)
    t = re.sub(rf"(?<=[-\d])[{UP}]\b|\b[{UP}](?=-)", lambda m: _spell(m.group()), t)
    # tek başına büyük harf (A girişi, E yazın, B'yi): harf adı + ek; "O" (zamir) hariç
    return re.sub(rf"(?<!\w)(?<!\w['’])(?!O\b)[{UP}](?!\w)(?:['’]([^\W\d_]+))?",
                  lambda m: _spell(m.group()[0]) + (tr_lower(m.group(1)) if m.group(1) else ""), t)


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFC", text).replace(" ", " ").replace("\t", " ")  # NFD yazımda (c + ◌̧) sözcük ikiye bölünüyordu
    t = re.sub(r"[ \t]*\n[ \t]*\n\s*", " . ", t)  # boş satır = cümle sonu
    t = t.replace("\n", " ")
    for f in (_sub_squares, _sub_codes, _sub_x, _sub_thousands, _sub_dates, _sub_times, _sub_percent, _sub_currency, _sub_units, _split_alnum, _sub_ordinals, _sub_abbr, _sub_numbers):
        t = f(t)
    t = _sub_letters(t)
    for k, v in SYMBOLS.items():
        t = t.replace(k, v)
    t = re.sub(r"\s[-–—]+\s|[–—]", " , ", t)  # boşluklu tire = duraklama
    t = re.sub(r"(?<=[\wçğıöşü])-(?=[\wçğıöşü])", " ", t)  # e-posta, örnek-market
    t = re.sub(r"(?<=[^\W\d_])['’](?=[^\W\d_])", "", t)  # Ağustos'ta -> Ağustosta
    t = re.sub(r"[\"“”„‘’'«»]", "", t)
    t = re.sub(r"[()\[\]{}:]", " , ", t).replace("…", " . ")
    t = re.sub(r"([,.?!;])", r" \1 ", t)
    # kalan simgeler: yalnız harf, ondalık rakam ve duraklama kalır. \w burada YETMEZ: ², ½, ① gibi No/Nl karakterleri \w'dir ama str.isalpha() değildir;
    # engine (_TOK) ile g2ptts tagger (isalpha) sözcük listeleri ayrışır ve g2ptts sessizce vurgu/sınır üretmezdi (tests/test_normalize.py)
    t = re.sub(r"[^\s,.?!;]", lambda m: m.group() if m.group().isalpha() or m.group().isdecimal() else " ", t)
    t = re.sub(r"\d+", lambda m: " " + digits(m.group()) + " ", t)  # güvenlik ağı: kimsenin yakalamadığı rakamlar sessizce silinmesin
    toks, out = t.split(), []
    for tok in toks:
        if tok in PUNCT_RANK:
            if not out:
                continue  # baştaki işaret düşer
            if out[-1] in PUNCT_RANK:
                if PUNCT_RANK[tok] > PUNCT_RANK[out[-1]]:
                    out[-1] = tok
                continue
        out.append(tok)
    if out and out[-1] not in ".?!":
        if out[-1] in PUNCT_RANK:
            out[-1] = "."
        else:
            out.append(".")
    return " ".join(out)
