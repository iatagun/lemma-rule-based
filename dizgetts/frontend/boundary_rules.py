"""Sınır kuralları: g2ptts sınır tahminini (Word.boundary: 0 | ip | IP | cümle) dilsel kalıplarla düzeltir. Engine(boundary_rules=True).

Kaynak: kullanıcının 40 cümlelik kulak etiketleri (D:/dizgetts/user_prosody, docs/prosody_research.md §7) — bu set GELİŞTİRME setidir;
kuralların nihai ölçümü yeni, bakılmamış cümlelerle yapılmalı. Kural yalnız sınırın YERİNİ/GÜCÜNÜ değiştirir (token dizisi aynı; yalnız dp_feat).
  K1 cümle başı söylem belirteci sonrası IP (Yani ‖, Ayrıca ‖, Bu arada ‖ ...); zayıf olanlar ip (Öncelikle ‖).
  K2 tümce düzeyinde bağlaç (önceki sözcük çekimli eylem / soru eki): bağlaçtan ÖNCE IP, bağlaçtan hemen SONRA sınır yok (gelmişti ‖ ama sofrada).
  K3 zarf-fiil / koşul sonrası en az ip; öbek >= 3 sözcükse IP (aradığında ‖, bitirmeden ‖, gerekirse ‖).
  K4 konu "ise" sonrası: öncesindeki öbek >= 2 sözcükse IP.
  K5 ad öbeği içi yanlış sınırları sil: ardışık sayı sözcükleri (kırk beş), -mAk için, tamlayan -(n)In + sonraki sözcük.
Bilerek YOK: uzun özne sonu (sözdizimi gerekir; Dep ile ileride), "ve" (etiketler karışık).
"""
from __future__ import annotations

import re

from dizgetts.frontend.normalize import tr_lower

RANK = {"0": 0, "ip": 1, "IP": 2, "cümle": 3}
DISCOURSE_IP = {("yani",), ("ayrıca",), ("sonunda",), ("bu", "arada"), ("istersen",), ("isterseniz",), ("aslında",), ("üstelik",), ("kısacası",),
                ("sonuç", "olarak"), ("bu", "yüzden"), ("bu", "nedenle"), ("dolayısıyla",), ("özetle",), ("neyse",)}
DISCOURSE_ip = {("öncelikle",), ("peki",), ("öyleyse",), ("yine", "de")}
CONJ = {"ama", "fakat", "ancak", "çünkü", "yoksa", "veya", "veyahut", "oysa", "halbuki", "ya"}  # "ya" = "ya da" başı
QPART = re.compile(r"^m[ıiuü](y[ıiuü]m|s[ıiuü]n|y[ıiuü]z|s[ıiuü]n[ıiuü]z|d[ıiuü]r|yd[ıiuü]|yd[ıiuü]k|yd[ıiuü]n[ıiuü]z|yken|ymış|ymiş)?$")
FINITE = re.compile(r"(d[ıiuü]|t[ıiuü]|m[ıiuü]ş|yor|[ae]c[ae]k|[ae]c[ae]ğ|[ıiuüae]r|m[ae]z|m[ae]l[ıi]|s[ıiuü]n|l[ıi]m|d[ıiuü]r|t[ıiuü]r)"
                    r"(m|n|k|z|s[ıiuü]n|s[ıiuü]n[ıiuü]z|n[ıiuü]z|[ıiuü]m|[ıiuü]z|[ıiuü]n|l[ae]r|d[ıiuü]|t[ıiuü]|m[ıiuü]ş|s[ae])*$")
CONVERB = re.compile(r"([dt][ıiuü][ğk][ıiuü]nd[ae]|[ıiuü]nc[ae]|ken|m[ae]d[ae]n"  # -DIğIndA, -(y)IncA, -ken, -mAdAn
                     r"|([ıiuüae]rs[ae]|[dt][ıiuü]ys[ae]|m[ıiuü]şs[ae]|yors[ae]|[ae]c[ae]ks[ae]|m[ae]zs[ae])(m|n|k|n[ıiuü]z)?)$")  # koşul: gelirse, geldiyse ...
CONVERB_NOT = {"önce", "bence", "sence", "ince", "düşünce", "eğlence", "bilmece", "erken", "boyunca", "iken", "ise", "ki", "hepsince", "sonunca"}
NUMS = {"sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş",
        "seksen", "doksan", "yüz", "bin", "milyon", "milyar", "buçuk", "virgül"}
GEN = re.compile(r"n?[ıiuü]n$")
GEN_NOT = {"yarın", "bütün", "için", "dün", "gün", "bugün", "düzgün", "olgun", "uzun", "kısın", "yakın", "sonun", "şunun", "bunun", "onun", "benim"}


LOG = None  # tanı için: liste verilirse (sözcük, kural, eski, yeni) eklenir


def _set(w, b, rule):
    if LOG is not None and w.boundary != b:
        LOG.append((w.text, rule, w.boundary, b))
    w.boundary = b


def _up(w, b, rule=""):
    if RANK[b] > RANK[w.boundary] and w.boundary != "cümle":
        _set(w, b, rule)


def apply(words) -> None:
    """words: Engine Word listesi (text normalize edilmiş, boundary g2ptts). Yerinde değiştirir."""
    t = [tr_lower(w.text) for w in words]
    n = len(words)
    starts = [i == 0 or words[i - 1].boundary == "cümle" for i in range(n)]  # cümle başı
    # K1
    for i in range(n):
        if not starts[i]:
            continue
        for tab, b in ((DISCOURSE_IP, "IP"), (DISCOURSE_ip, "ip")):
            for seq in tab:
                if tuple(t[i:i + len(seq)]) == seq and i + len(seq) < n:
                    _up(words[i + len(seq) - 1], b, "K1")
    # K2
    for i in range(1, n):
        if t[i] in CONJ and not starts[i] and (t[i] != "ya" or (i + 1 < n and t[i + 1] == "da")):
            p = t[i - 1]
            if QPART.match(p) or FINITE.search(p) or words[i - 1].punct:
                _up(words[i - 1], "IP", "K2")
                last = i + 1 if t[i] == "ya" else i  # "ya da" -> "da"dan sonra
                if last < n - 1 and not words[last].punct and words[last].boundary in ("ip", "IP"):
                    _set(words[last], "0", "K2-sil")
    # K3 / K4
    seg = 0  # son sınırdan beri sözcük sayısı
    for i in range(n - 1):
        seg += 1
        if CONVERB.search(t[i]) and t[i] not in CONVERB_NOT and len(t[i]) > 4:
            _up(words[i], "IP" if seg >= 3 else "ip", "K3")
        if t[i] == "ise" and seg >= 3:  # "ise" + önünde en az 2 sözcük
            _up(words[i], "IP", "K4")
        if words[i].boundary != "0":
            seg = 0
    # K5
    for i in range(n - 1):
        if words[i].punct or words[i].boundary in ("0", "cümle"):
            continue
        if (t[i] in NUMS and t[i + 1] in NUMS) or (t[i + 1] == "için" and re.search(r"m[ae]k$", t[i])) or \
           (GEN.search(t[i]) and t[i] not in GEN_NOT and len(t[i]) > 4 and words[i].boundary == "ip"):
            _set(words[i], "0", "K5")


if __name__ == "__main__":  # öz-denetim: kural davranışı (Word benzeri basit nesnelerle)
    from types import SimpleNamespace as W

    def run(s, bs):
        ws = [W(text=x, boundary=b, punct=[]) for x, b in zip(s.split(), bs)]
        apply(ws)
        return [w.boundary for w in ws]

    assert run("yani ilk başta hiçbir masraf yok", "0 ip 0 0 0 cümle".split())[0] == "IP"
    assert run("bu arada senin kitabın bende", "0 0 0 0 cümle".split())[:2] == ["0", "IP"]
    r = run("o an dünyanın sonu gibi gelmişti ama sofrada kimse umursamadı", "0 ip 0 0 0 0 ip 0 0 cümle".split())
    assert r[5] == "IP" and r[6] == "0", r
    assert run("küçük bir şey ama ikimizi de etti", "0 0 0 ip 0 0 cümle".split())[3] == "ip"  # ad öbeği + ama: dokunulmaz
    assert run("tanıdık bir müşteri aradığında karşılama samimi olur", "0 0 0 0 0 0 cümle".split())[3] == "IP"
    assert run("eve gelince önce açtım", "0 0 0 cümle".split())[1] == "ip"
    assert run("yaklaşık kırk beş dakika sürüyor", "0 ip 0 0 cümle".split())[1] == "0"
    assert run("erken gelirse bize haber ver", "0 0 0 0 cümle".split())[:2] == ["0", "ip"]
    assert run("herkese önce bilgi verdik", "0 0 0 cümle".split())[:2] == ["0", "0"]
    assert run("dükkânın hemen yanındaki deftere", "ip 0 0 cümle".split())[0] == "0"
    assert run("iç salonumuz ise canlı müzik var", "0 0 0 0 0 cümle".split())[2] == "IP"
    assert run("ben ise emin değilim", "0 ip 0 cümle".split())[1] == "ip"
    print("boundary_rules öz-denetim OK")
