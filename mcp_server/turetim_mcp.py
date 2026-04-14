"""
Türkçe Morfoloji & Türetim Ekleri MCP Sunucusu
================================================
İki katmanlı MCP:
  1. BİLGİ TABANI — 191 türetim eki referansı (Uzun et al. 1992)
  2. AKTİF MOTOR  — Gerçek morfolojik çözümleme, cümle analizi,
                    dependency parsing ve benchmark değerlendirmesi.

VS Code Copilot Chat'e Türkçe dilbilim asistanı olarak hizmet verir.
"""

from mcp.server.fastmcp import FastMCP
import json
import os
import sys
from pathlib import Path

# Proje kök dizinini sys.path'e ekle (morphology paketine erişim için)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

mcp = FastMCP(
    "Türkçe Morfoloji Asistanı",
    instructions=(
        "Türkçe morfoloji ve sözdizim analiz sunucusu. "
        "İki katmanlı mimari: (1) 191 türetim ekinin yapılandırılmış bilgi tabanı, "
        "(2) çalışan morfolojik çözümleyici, cümle analizörü ve dependency parser. "
        "Sözcük çözümle, cümle analiz et, dependency ağacı çiz, benchmark çalıştır. "
        "Kaynak: Uzun et al. (1992) + lemma-rule-based kural tabanlı motor."
    ),
)


# ============================================================
# AKTİF MOTOR — Lazy Singleton
# ============================================================
_analyzer = None
_sentence_analyzer = None
_dep_parser = None


def _get_analyzer():
    """Morfolojik çözümleyiciyi lazily başlatır."""
    global _analyzer
    if _analyzer is None:
        from morphology import create_default_analyzer
        dict_path = _PROJECT_ROOT / "turkish_words.txt"
        if dict_path.exists():
            _analyzer = create_default_analyzer(dictionary_path=dict_path)
        else:
            _analyzer = create_default_analyzer()
    return _analyzer


def _get_sentence_analyzer():
    """Cümle analizörünü lazily başlatır."""
    global _sentence_analyzer
    if _sentence_analyzer is None:
        from morphology.sentence import SentenceAnalyzer
        _sentence_analyzer = SentenceAnalyzer(_get_analyzer())
    return _sentence_analyzer


def _get_dep_parser():
    """Dependency parser'ı lazily başlatır."""
    global _dep_parser
    if _dep_parser is None:
        from morphology.dependency import DependencyParser
        _dep_parser = DependencyParser()
    return _dep_parser

# ============================================================
# VERİ: 191 Türetim Eki
# ============================================================
# Her ek: kod, ek_formu, taban_turu, turev_turu, ornek_sayisi,
#          ornekler, turetim_turu (A→A, A→E, E→A, E→E), aciklama

EKLER = [
    # === A Serisi (A1-A30) ===
    {"kod": "A1", "ek": "-A", "taban": "EYLEM", "turev": "AD", "sayi": 22, "ornekler": "yara, gele", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "A2", "ek": "-(y)A-", "taban": "EYLEM", "turev": "EYLEM", "sayi": 16, "ornekler": "çık-a-, tık-a-", "tur": "E→E", "aciklama": "Fiilden fiil"},
    {"kod": "A3", "ek": "-(y)AcAk", "taban": "EYLEM", "turev": "AD", "sayi": 15, "ornekler": "içecek, çıkacak", "tur": "E→A", "aciklama": "Gelecek sıfat-fiil/isim"},
    {"kod": "A4", "ek": "-AcAn", "taban": "EYLEM", "turev": "SIFAT", "sayi": 3, "ornekler": "sevecen", "tur": "E→A", "aciklama": "Fiilden sıfat"},
    {"kod": "A5", "ek": "-AcIk", "taban": "SIFAT", "turev": "SIFAT", "sayi": 3, "ornekler": "daracık", "tur": "A→A", "aciklama": "Sıfattan sıfat"},
    {"kod": "A6", "ek": "-AdAk", "taban": "AD", "turev": "BELİRTEÇ", "sayi": 20, "ornekler": "cartadak", "tur": "A→A", "aciklama": "İsimden belirteç"},
    {"kod": "A7", "ek": "-AdAn", "taban": "AD", "turev": "BELİRTEÇ", "sayi": 3, "ornekler": "türpeden", "tur": "A→A", "aciklama": "İsimden belirteç"},
    {"kod": "A8", "ek": "-AgAn", "taban": "EYLEM", "turev": "AD", "sayi": 1, "ornekler": "gezegen", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "A9", "ek": "-AğAn", "taban": "EYLEM", "turev": "SIFAT", "sayi": 9, "ornekler": "kayağan", "tur": "E→A", "aciklama": "Fiilden sıfat"},
    {"kod": "A10", "ek": "-AgI", "taban": "EYLEM", "turev": "AD", "sayi": 1, "ornekler": "karağı", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "A11", "ek": "-AksA-", "taban": "EYLEM", "turev": "EYLEM", "sayi": 1, "ornekler": "duraksa-", "tur": "E→E", "aciklama": "Fiilden fiil"},
    {"kod": "A12", "ek": "-AlA-", "taban": "EYLEM", "turev": "EYLEM", "sayi": 21, "ornekler": "kovala-", "tur": "E→E", "aciklama": "Fiilden fiil"},
    {"kod": "A13", "ek": "-AlAk", "taban": "EYLEM", "turev": "AD.SIFAT", "sayi": 1, "ornekler": "yatalak", "tur": "E→A", "aciklama": "Fiilden ad/sıfat"},
    {"kod": "A14", "ek": "-AlgA", "taban": "EYLEM", "turev": "AD", "sayi": 8, "ornekler": "gezalga", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "A15", "ek": "-Am", "taban": "EYLEM", "turev": "AD", "sayi": 4, "ornekler": "dönüm", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "A16", "ek": "-AmAç", "taban": "EYLEM", "turev": "AD", "sayi": 2, "ornekler": "dönemeç", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "A17", "ek": "-AmAk", "taban": "EYLEM", "turev": "AD", "sayi": 3, "ornekler": "basamak", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "A18", "ek": "-AmAmAzlIk", "taban": "EYLEM", "turev": "AD", "sayi": 1, "ornekler": "çekememezlik", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "A19", "ek": "-(y)An", "taban": "EYLEM", "turev": "AD", "sayi": 34, "ornekler": "çağlayan, bileşen", "tur": "E→A", "aciklama": "Şimdiki zaman sıfat-fiili"},
    {"kod": "A20", "ek": "-An", "taban": "AD", "turev": "AD", "sayi": 7, "ornekler": "kızan", "tur": "A→A", "aciklama": "İsimden isim"},
    {"kod": "A21", "ek": "-AncA", "taban": "EYLEM", "turev": "AD", "sayi": 1, "ornekler": "dönence", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "A22", "ek": "-(ş)Ar", "taban": "AD.SIFAT", "turev": "SIFAT", "sayi": 1, "ornekler": "altışar", "tur": "A→A", "aciklama": "Üleştirme sayısı"},
    {"kod": "A23", "ek": "-ArAk", "taban": "EYLEM", "turev": "AD", "sayi": 3, "ornekler": "hılarak", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "A24", "ek": "-ArgA", "taban": "EYLEM", "turev": "AD", "sayi": 2, "ornekler": "düyarga", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "A25", "ek": "-At", "taban": "AD.SIFAT", "turev": "SIFAT", "sayi": 3, "ornekler": "dışat", "tur": "A→A", "aciklama": "İsimden sıfat"},
    {"kod": "A26", "ek": "-AsI", "taban": "EYLEM", "turev": "SIFAT", "sayi": 2, "ornekler": "olası, şaşılası", "tur": "E→A", "aciklama": "Fiilden sıfat"},
    {"kod": "A27", "ek": "-AsIyA", "taban": "EYLEM", "turev": "BELİRTEÇ", "sayi": 5, "ornekler": "kıyasıya", "tur": "E→A", "aciklama": "Fiilden belirteç"},
    {"kod": "A28", "ek": "-(y)Aş-", "taban": "AD.SIFAT", "turev": "EYLEM", "sayi": 1, "ornekler": "yanaş-", "tur": "A→E", "aciklama": "İsimden fiil"},
    {"kod": "A29", "ek": "-At", "taban": "AD", "turev": "AD/SIFAT", "sayi": 4, "ornekler": "gölat", "tur": "A→A", "aciklama": "İsimden isim"},
    {"kod": "A30", "ek": "-At-", "taban": "AD", "turev": "EYLEM", "sayi": 2, "ornekler": "yönet-", "tur": "A→E", "aciklama": "İsimden fiil"},
    # === C Serisi (C31-C34) ===
    {"kod": "C31", "ek": "-cAcIk", "taban": "BELİRTEÇ", "turev": "BELİRTEÇ", "sayi": 1, "ornekler": "hemencecik", "tur": "A→A", "aciklama": "Belirteçten belirteç"},
    {"kod": "C32", "ek": "-cAğIz", "taban": "AD", "turev": "AD", "sayi": 5, "ornekler": "kızcağız", "tur": "A→A", "aciklama": "Küçültme/acıma"},
    {"kod": "C33", "ek": "-cAsInA", "taban": "AD", "turev": "BELİRTEÇ", "sayi": 1, "ornekler": "taparcasına", "tur": "A→A", "aciklama": "Benzetme belirteci"},
    {"kod": "C34", "ek": "-(A)ç", "taban": "EYLEM/AD/SIFAT", "turev": "AD/SIFAT", "sayi": 54, "ornekler": "sayaç, döveç", "tur": "E→A", "aciklama": "Alet/nitelik"},
    # === Ç Serisi (Ç35-Ç44) ===
    {"kod": "Ç35", "ek": "-(I)ç", "taban": "EYLEM", "turev": "AD/SIFAT", "sayi": 23, "ornekler": "direnç, ilenç", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "Ç36", "ek": "-çA", "taban": "SIFAT/AD/EYLEM", "turev": "AD/SIFAT", "sayi": 344, "ornekler": "Türkçe, boyunca, İngilizce", "tur": "A→A", "aciklama": "Dil/ölçü/benzerlik (en verimli 11.)"},
    {"kod": "Ç37", "ek": "-çAk", "taban": "AD/BELİRTEÇ/EYLEM", "turev": "AD/BELİRTEÇ", "sayi": 22, "ornekler": "demincek, oyuncak", "tur": "A→A", "aciklama": "Küçültme/alet"},
    {"kod": "Ç38", "ek": "-çAl", "taban": "AD", "turev": "SIFAT", "sayi": 2, "ornekler": "kilçal", "tur": "A→A", "aciklama": "Nitelik"},
    {"kod": "Ç39", "ek": "-çI", "taban": "AD/SIFAT", "turev": "AD/SIFAT", "sayi": 1259, "ornekler": "balıkçı, işçi, gözcü, yalancı", "tur": "A→A", "aciklama": "Meslek/ilgi/eğilim (3. en sık)"},
    {"kod": "Ç40", "ek": "-çIk", "taban": "AD/SIFAT/EYLEM", "turev": "AD/SIFAT", "sayi": 80, "ornekler": "kapıcık", "tur": "A→A", "aciklama": "Küçültme"},
    {"kod": "Ç41", "ek": "-çIktAn", "taban": "AD", "turev": "BELİRTEÇ", "sayi": 3, "ornekler": "şakarcıktan", "tur": "A→A", "aciklama": "Belirteç"},
    {"kod": "Ç42", "ek": "-çI(l)", "taban": "AD/SIFAT", "turev": "SIFAT", "sayi": 53, "ornekler": "sesçil, iyimsil", "tur": "A→A", "aciklama": "Nitelik/eğilim"},
    {"kod": "Ç43", "ek": "-çIlAyIn", "taban": "ADIL", "turev": "BELİRTEÇ", "sayi": 2, "ornekler": "bencileyin", "tur": "A→A", "aciklama": "Zamir tabanlı belirteç"},
    {"kod": "Ç44", "ek": "-çIn", "taban": "AD", "turev": "AD", "sayi": 5, "ornekler": "balıkçın, gümüşçün", "tur": "A→A", "aciklama": "İsimden isim"},
    # === D Serisi (D45-D50) ===
    {"kod": "D45", "ek": "-dA-", "taban": "AD", "turev": "EYLEM", "sayi": 76, "ornekler": "fıkırda-, gürülde-", "tur": "A→E", "aciklama": "Yansıma tabanlı fiil"},
    {"kod": "D46", "ek": "-dAk", "taban": "AD", "turev": "AD", "sayi": 1, "ornekler": "özdek", "tur": "A→A", "aciklama": "İsimden isim"},
    {"kod": "D47", "ek": "-dAm", "taban": "AD", "turev": "AD", "sayi": 2, "ornekler": "erdem, gündem", "tur": "A→A", "aciklama": "İsimden isim"},
    {"kod": "D48", "ek": "-dIk", "taban": "EYLEM", "turev": "AD", "sayi": 2, "ornekler": "tanıdık", "tur": "E→A", "aciklama": "Sıfat-fiil kalıplaşması"},
    {"kod": "D49", "ek": "-dIrIk", "taban": "AD", "turev": "AD", "sayi": 4, "ornekler": "burundirik", "tur": "A→A", "aciklama": "İsimden isim"},
    {"kod": "D50", "ek": "-dIz(I)", "taban": "AD", "turev": "AD.BELİRTEÇ", "sayi": 1, "ornekler": "gündüz", "tur": "A→A", "aciklama": "İsimden isim/belirteç"},
    # === G Serisi (G51-G53) ===
    {"kod": "G51", "ek": "-gIç", "taban": "EYLEM", "turev": "AD", "sayi": 10, "ornekler": "başlangıç, yorgunluk", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "G52", "ek": "-gIl", "taban": "AD", "turev": "AD", "sayi": 9, "ornekler": "babamgil", "tur": "A→A", "aciklama": "Aile/ev halkı"},
    {"kod": "G53", "ek": "-gIller", "taban": "AD", "turev": "AD", "sayi": 240, "ornekler": "gülgiller, kedigiller", "tur": "A→A", "aciklama": "Biyolojik cins/familya (15. en sık)"},
    # === I Serisi (I54-I77) ===
    {"kod": "I54", "ek": "-(y)I", "taban": "EYLEM", "turev": "AD", "sayi": 89, "ornekler": "yazı, tartı, örtü", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "I55", "ek": "-(s)I", "taban": "ADIL/SIFAT", "turev": "ADIL/SIFAT", "sayi": 15, "ornekler": "şurası, hangisi", "tur": "A→A", "aciklama": "Zamir türetimi"},
    {"kod": "I56", "ek": "-(l)I-", "taban": "AD/SIFAT", "turev": "EYLEM", "sayi": 5, "ornekler": "tozla-? dozla-?", "tur": "A→E", "aciklama": "İsimden fiil"},
    {"kod": "I57", "ek": "-IcI", "taban": "EYLEM", "turev": "SIFAT/AD", "sayi": 255, "ornekler": "yazıcı, yapıştırıcı, alıcı", "tur": "E→A", "aciklama": "Fail/meslek/alet (14. en sık)"},
    {"kod": "I58", "ek": "-IcIk", "taban": "EYLEM", "turev": "AD", "sayi": 4, "ornekler": "gülücük", "tur": "E→A", "aciklama": "Küçültmeli fiilden isim"},
    {"kod": "I59", "ek": "-(I)lk-", "taban": "AD.SIFAT", "turev": "EYLEM", "sayi": 3, "ornekler": "birk-, gözk-?", "tur": "A→E", "aciklama": "İsimden fiil"},
    {"kod": "I60", "ek": "-IktIr-", "taban": "EYLEM", "turev": "EYLEM", "sayi": 2, "ornekler": "çizdirtir-", "tur": "E→E", "aciklama": "Çifte ettirgen"},
    {"kod": "I61", "ek": "-(I)l-", "taban": "EYLEM", "turev": "EYLEM", "sayi": 404, "ornekler": "yazıl-, sayıl-, yalıtıl-", "tur": "E→E", "aciklama": "Edilgen çatı (9. en sık)"},
    {"kod": "I62", "ek": "-(I)l- [dönüşlü]", "taban": "EYLEM", "turev": "EYLEM", "sayi": 29, "ornekler": "açıl-", "tur": "E→E", "aciklama": "Dönüşlü çatı (-l- allomorf)"},
    {"kod": "I63", "ek": "-(I)l", "taban": "EYLEM", "turev": "SIFAT", "sayi": 20, "ornekler": "yatık?", "tur": "E→A", "aciklama": "Fiilden sıfat"},
    {"kod": "I64", "ek": "-(y)Im", "taban": "EYLEM", "turev": "AD", "sayi": 202, "ornekler": "yazım, çizim, dönüşüm", "tur": "E→A", "aciklama": "Fiilden isim (16. en sık)"},
    {"kod": "I65", "ek": "-ImlIk", "taban": "EYLEM", "turev": "AD", "sayi": 2, "ornekler": "ısırımlık", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "I66", "ek": "-In", "taban": "EYLEM", "turev": "AD", "sayi": 24, "ornekler": "dizin, söken", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "I67", "ek": "-(s)InA", "taban": "AD/SIFAT/BELİRTEÇ", "turev": "BELİRTEÇ", "sayi": 10, "ornekler": "boşuna, zararına", "tur": "A→A", "aciklama": "Zarf yapan"},
    {"kod": "I68", "ek": "-(I)ncA", "taban": "AD", "turev": "BELİRTEÇ", "sayi": 8, "ornekler": "gereğince", "tur": "A→A", "aciklama": "Zarf yapan"},
    {"kod": "I69", "ek": "-IncAmA", "taban": "EYLEM", "turev": "AD", "sayi": 1, "ornekler": "sürünceme", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "I70", "ek": "-(I)ndA", "taban": "AD", "turev": "BELİRTEÇ", "sayi": 4, "ornekler": "zamanında", "tur": "A→A", "aciklama": "Zarf yapan"},
    {"kod": "I71", "ek": "-(I)r-", "taban": "AD", "turev": "EYLEM", "sayi": 4, "ornekler": "üfür-", "tur": "A→E", "aciklama": "İsimden fiil"},
    {"kod": "I72", "ek": "-(I)rgA-", "taban": "AD.SIFAT", "turev": "EYLEM", "sayi": 1, "ornekler": "yadırga-", "tur": "A→E", "aciklama": "İsimden fiil"},
    {"kod": "I73", "ek": "-(y)Iş", "taban": "EYLEM", "turev": "AD", "sayi": 4, "ornekler": "oturuş", "tur": "E→A", "aciklama": "Fiilden isim (biçim/tarz)"},
    {"kod": "I74", "ek": "-(y)Iş [geniş]", "taban": "EYLEM", "turev": "AD", "sayi": 36, "ornekler": "varoluş, açılış", "tur": "E→A", "aciklama": "Fiilden isim (süreç/olay)"},
    {"kod": "I75", "ek": "-(I)t-", "taban": "SIFAT", "turev": "EYLEM", "sayi": 2, "ornekler": "pekit-?", "tur": "A→E", "aciklama": "Sıfattan fiil"},
    {"kod": "I76", "ek": "-(s)IylA", "taban": "AD", "turev": "BELİRTEÇ", "sayi": 7, "ornekler": "vaktiyle", "tur": "A→A", "aciklama": "Zarf yapan"},
    {"kod": "I77", "ek": "-(I)z", "taban": "EYLEM/AD.SIFAT", "turev": "SIFAT/AD", "sayi": 2, "ornekler": "topuz", "tur": "E→A", "aciklama": "Fiilden isim"},
    # === K Serisi (K78-K92) ===
    {"kod": "K78", "ek": "-(A/I)k", "taban": "EYLEM/AD/SIFAT", "turev": "SIFAT/AD", "sayi": 277, "ornekler": "kıyak, soluk, topak", "tur": "E→A", "aciklama": "Çok verimli (13. en sık)"},
    {"kod": "K79", "ek": "-(I)k-", "taban": "EYLEM", "turev": "EYLEM", "sayi": 3, "ornekler": "burk-", "tur": "E→E", "aciklama": "Fiilden fiil"},
    {"kod": "K80", "ek": "-KA", "taban": "EYLEM/AD/SIFAT", "turev": "AD", "sayi": 23, "ornekler": "süpürge, sömürge", "tur": "E→A", "aciklama": "Fiilden alet ismi"},
    {"kod": "K81", "ek": "-KAç", "taban": "EYLEM", "turev": "AD/SIFAT", "sayi": 18, "ornekler": "kıskaç, değiştirgaç", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "K82", "ek": "-kAk", "taban": "AD.SIFAT", "turev": "AD.SIFAT", "sayi": 1, "ornekler": "erkek", "tur": "A→A", "aciklama": "İsimden isim"},
    {"kod": "K83", "ek": "-KAn", "taban": "EYLEM", "turev": "SIFAT/AD", "sayi": 74, "ornekler": "konuşkan, çalışkan", "tur": "E→A", "aciklama": "Fiilden sıfat (eğilim)"},
    {"kod": "K84", "ek": "-(y)ken", "taban": "BELİRTEÇ", "turev": "BELİRTEÇ", "sayi": 3, "ornekler": "böyleken", "tur": "A→A", "aciklama": "Belirteçten belirteç"},
    {"kod": "K85", "ek": "-KI", "taban": "EYLEM/AD.SIFAT", "turev": "AD/SIFAT", "sayi": 90, "ornekler": "biçki, alçı", "tur": "E→A", "aciklama": "Fiilden isim/sıfat"},
    {"kod": "K86", "ek": "-kI", "taban": "BELİRTEÇ", "turev": "AD/SIFAT/BELİRTEÇ", "sayi": 18, "ornekler": "dünkü, yukarıki", "tur": "A→A", "aciklama": "Zaman/yer niteleme"},
    {"kod": "K87", "ek": "-KIn", "taban": "EYLEM", "turev": "SIFAT/AD", "sayi": 84, "ornekler": "seçkin, yetişkin", "tur": "E→A", "aciklama": "Fiilden sıfat"},
    {"kod": "K88", "ek": "-kIn-", "taban": "EYLEM", "turev": "EYLEM", "sayi": 1, "ornekler": "yütk-", "tur": "E→E", "aciklama": "Fiilden fiil"},
    {"kod": "K89", "ek": "-kIr-", "taban": "AD", "turev": "EYLEM", "sayi": 12, "ornekler": "püskür-", "tur": "A→E", "aciklama": "İsimden fiil"},
    {"kod": "K90", "ek": "-(A/I)klA-", "taban": "EYLEM", "turev": "EYLEM", "sayi": 9, "ornekler": "dürtükle-", "tur": "E→E", "aciklama": "Fiilden fiil"},
    {"kod": "K91", "ek": "-(A/I)klI", "taban": "EYLEM", "turev": "SIFAT", "sayi": 5, "ornekler": "okumaklı?", "tur": "E→A", "aciklama": "Fiilden sıfat"},
    {"kod": "K92", "ek": "-(A/I)ksIz", "taban": "EYLEM", "turev": "SIFAT", "sayi": 2, "ornekler": "okumaksız?", "tur": "E→A", "aciklama": "Fiilden sıfat (olumsuz)"},
    # === L Serisi (L93-L113) ===
    {"kod": "L93", "ek": "-(A/I)l-", "taban": "SIFAT/EYLEM", "turev": "EYLEM", "sayi": 30, "ornekler": "daral-, çömel-", "tur": "A→E", "aciklama": "Sıfattan fiil"},
    {"kod": "L94", "ek": "-(A/I)l", "taban": "AD/SIFAT/EYLEM", "turev": "SIFAT/AD", "sayi": 29, "ornekler": "ulusal, yanal", "tur": "A→A", "aciklama": "İsimden sıfat"},
    {"kod": "L95", "ek": "-(l)A-", "taban": "AD/SIFAT", "turev": "EYLEM", "sayi": 840, "ornekler": "başla-, taşla-, gözle-", "tur": "A→E", "aciklama": "İsimden fiil (4. en sık)"},
    {"kod": "L96", "ek": "-(y)lA", "taban": "AD", "turev": "BELİRTEÇ", "sayi": 28, "ornekler": "öfkeyle", "tur": "A→A", "aciklama": "İsimden belirteç"},
    {"kod": "L97", "ek": "-(l)Ak", "taban": "AD", "turev": "AD", "sayi": 10, "ornekler": "otlak", "tur": "A→A", "aciklama": "İsimden isim (yer)"},
    {"kod": "L98", "ek": "-(l)Am", "taban": "AD/AD.SIFAT", "turev": "AD", "sayi": 4, "ornekler": "enlem, içlem", "tur": "A→A", "aciklama": "İsimden isim"},
    {"kod": "L99", "ek": "-(l)AmA", "taban": "AD/SIFAT/AD.SIFAT", "turev": "AD", "sayi": 27, "ornekler": "güzelleme", "tur": "A→A", "aciklama": "İsimden isim"},
    {"kod": "L100", "ek": "-(l)AmAsInA", "taban": "SIFAT/AD.SIFAT", "turev": "BELİRTEÇ", "sayi": 2, "ornekler": "uzunlamasına", "tur": "A→A", "aciklama": "Sıfattan belirteç"},
    {"kod": "L101", "ek": "-(l)An-", "taban": "AD/SIFAT", "turev": "EYLEM", "sayi": 277, "ornekler": "evlen-, utanlan-", "tur": "A→E", "aciklama": "İsimden fiil (12. en sık)"},
    {"kod": "L102", "ek": "-(l)AndIr-", "taban": "AD/SIFAT", "turev": "EYLEM", "sayi": 13, "ornekler": "ödüllendir-", "tur": "A→E", "aciklama": "Ettirgen isimden fiil"},
    {"kod": "L103", "ek": "-(l)Ar", "taban": "AD/SIFAT", "turev": "AD", "sayi": 157, "ornekler": "gülgiller, yırtıcılar", "tur": "A→A", "aciklama": "Çoğul/topluluk (18. en sık)"},
    {"kod": "L104", "ek": "-(l)ArcA", "taban": "SIFAT/BELİRTEÇ", "turev": "SIFAT", "sayi": 7, "ornekler": "binlerce", "tur": "A→A", "aciklama": "Sayı/miktar sıfatı"},
    {"kod": "L105", "ek": "-(l)ArdA", "taban": "SIFAT/BELİRTEÇ", "turev": "BELİRTEÇ", "sayi": 6, "ornekler": "geçenlerde", "tur": "A→A", "aciklama": "Zaman belirteci"},
    {"kod": "L106", "ek": "-(l)ArI", "taban": "AD/ADIL", "turev": "AD", "sayi": 35, "ornekler": "denizaltıları", "tur": "A→A", "aciklama": "Çoğul iyelik"},
    {"kod": "L107", "ek": "-(l)Aş-", "taban": "SIFAT/AD/AD.SIFAT", "turev": "EYLEM", "sayi": 518, "ornekler": "güzelleş-, farklılaş-", "tur": "A→E", "aciklama": "İsimden fiil (8. en sık)"},
    {"kod": "L108", "ek": "-(l)AştIr-", "taban": "AD/SIFAT/AD.SIFAT", "turev": "EYLEM", "sayi": 13, "ornekler": "karşılaştır-", "tur": "A→E", "aciklama": "Ettirgen -lAş-"},
    {"kod": "L109", "ek": "-(l)At-", "taban": "AD", "turev": "EYLEM", "sayi": 3, "ornekler": "körelti?", "tur": "A→E", "aciklama": "İsimden fiil"},
    {"kod": "L110", "ek": "-leyIn", "taban": "AD.BELİRTEÇ", "turev": "BELİRTEÇ", "sayi": 3, "ornekler": "sabahleyin, geceleyin", "tur": "A→A", "aciklama": "Zaman belirteci"},
    {"kod": "L111", "ek": "-(l)I", "taban": "AD/SIFAT/AD.SIFAT", "turev": "SIFAT", "sayi": 1644, "ornekler": "tuzlu, güçlü, kireçli", "tur": "A→A", "aciklama": "Sahiplik sıfatı (2. en sık)"},
    {"kod": "L112", "ek": "-(l)Ik", "taban": "AD/SIFAT/AD.SIFAT", "turev": "AD/SIFAT", "sayi": 3259, "ornekler": "güzellik, yazarlık, çocukluk", "tur": "A→A", "aciklama": "Soyutluk/durum (1. en sık, %22.27)"},
    {"kod": "L113", "ek": "-(l)U", "taban": "AD.SIFAT", "turev": "AD", "sayi": 1, "ornekler": "karayolu?", "tur": "A→A", "aciklama": "Nadir"},
    # === M Serisi (M114-M135) ===
    {"kod": "M114", "ek": "-mA [alet]", "taban": "EYLEM", "turev": "AD", "sayi": 1, "ornekler": "süpürme (alet)", "tur": "E→A", "aciklama": "Fiilden alet ismi"},
    {"kod": "M115", "ek": "-mA", "taban": "EYLEM", "turev": "AD", "sayi": 123, "ornekler": "dolma, yazma, boyama", "tur": "E→A", "aciklama": "Fiilden isim mastarı (19. en sık)"},
    {"kod": "M116", "ek": "-mAcA", "taban": "EYLEM", "turev": "AD", "sayi": 21, "ornekler": "kovlamaca, aldatmaca", "tur": "E→A", "aciklama": "Oyun/hareket ismi"},
    {"kod": "M117", "ek": "-mAç", "taban": "EYLEM", "turev": "AD", "sayi": 20, "ornekler": "bulmaca, dolmaç", "tur": "E→A", "aciklama": "Araç/nesne ismi"},
    {"kod": "M118", "ek": "-mAdIk", "taban": "EYLEM", "turev": "SIFAT", "sayi": 7, "ornekler": "olmadık", "tur": "E→A", "aciklama": "Olumsuz sıfat-fiil"},
    {"kod": "M119", "ek": "-mAk [alet]", "taban": "EYLEM", "turev": "AD", "sayi": 1, "ornekler": "süpürmek?", "tur": "E→A", "aciklama": "Alet ismi"},
    {"kod": "M120", "ek": "-mAk", "taban": "EYLEM", "turev": "AD", "sayi": 5, "ornekler": "çakmak", "tur": "E→A", "aciklama": "Kalıplaşmış mastar"},
    {"kod": "M121", "ek": "-mAmAzlIk", "taban": "EYLEM", "turev": "AD", "sayi": 1, "ornekler": "anlamamazlık", "tur": "E→A", "aciklama": "Olumsuz isim"},
    {"kod": "M122", "ek": "-mAn", "taban": "EYLEM/AD.SIFAT/SIFAT", "turev": "AD/SIFAT", "sayi": 14, "ornekler": "öğretmen, seçmen, kölemen", "tur": "E→A", "aciklama": "Fail/kişi ismi"},
    {"kod": "M123", "ek": "-mAsInA", "taban": "BELİRTEÇ", "turev": "BELİRTEÇ", "sayi": 3, "ornekler": "şöylemesine", "tur": "A→A", "aciklama": "Belirteçten belirteç"},
    {"kod": "M124", "ek": "-mAş-", "taban": "EYLEM", "turev": "EYLEM", "sayi": 2, "ornekler": "sarmaş-", "tur": "E→E", "aciklama": "İşteşlik"},
    {"kod": "M125", "ek": "-mAz", "taban": "EYLEM", "turev": "AD", "sayi": 3, "ornekler": "tükenmez", "tur": "E→A", "aciklama": "Olumsuz niteleme"},
    {"kod": "M126", "ek": "-mAzlIk", "taban": "EYLEM", "turev": "AD", "sayi": 11, "ornekler": "duymazlık", "tur": "E→A", "aciklama": "Olumsuz durum ismi"},
    {"kod": "M127", "ek": "-mbAç", "taban": "EYLEM", "turev": "AD", "sayi": 3, "ornekler": "saklambaç", "tur": "E→A", "aciklama": "Oyun ismi"},
    {"kod": "M128", "ek": "-mIk", "taban": "EYLEM", "turev": "AD/SIFAT", "sayi": 8, "ornekler": "kıymık", "tur": "E→A", "aciklama": "Parça/küçültme"},
    {"kod": "M129", "ek": "-mIr", "taban": "EYLEM", "turev": "AD", "sayi": 1, "ornekler": "yağmur", "tur": "E→A", "aciklama": "Doğa olayı ismi"},
    {"kod": "M130", "ek": "-mIş", "taban": "EYLEM", "turev": "AD", "sayi": 10, "ornekler": "ermiş, dolmuş", "tur": "E→A", "aciklama": "Kalıplaşmış sıfat-fiil"},
    {"kod": "M131", "ek": "-(I)mIz", "taban": "BELİRTEÇ", "turev": "ADIL", "sayi": 1, "ornekler": "hepimiz", "tur": "A→A", "aciklama": "Zamir"},
    {"kod": "M132", "ek": "-(I)msA-", "taban": "SIFAT/EYLEM/AD/ADIL", "turev": "EYLEM", "sayi": 11, "ornekler": "kötümsemek, yadırgamsa-", "tur": "A→E", "aciklama": "Psikolojik fiil"},
    {"kod": "M133", "ek": "-msAr", "taban": "SIFAT", "turev": "SIFAT", "sayi": 2, "ornekler": "iyimser, kötümser", "tur": "A→A", "aciklama": "Eğilim sıfatı"},
    {"kod": "M134", "ek": "-(I)msI", "taban": "AD/SIFAT/AD.SIFAT", "turev": "SIFAT", "sayi": 45, "ornekler": "mavimsi, yeşilimsi", "tur": "A→A", "aciklama": "Benzerlik sıfatı"},
    {"kod": "M135", "ek": "-(I)mtrAk", "taban": "SIFAT/AD.SIFAT", "turev": "SIFAT", "sayi": 8, "ornekler": "yeşilimtırak, ekşimtırak", "tur": "A→A", "aciklama": "Yaklaşıklık sıfatı"},
    # === N Serisi (N136-N144) ===
    {"kod": "N136", "ek": "-(I)n- [dönüşlü]", "taban": "EYLEM", "turev": "EYLEM", "sayi": 121, "ornekler": "giy-in-, yıka-n-", "tur": "E→E", "aciklama": "Dönüşlü çatı (20. en sık)"},
    {"kod": "N137", "ek": "-(I)n- [edilgen]", "taban": "EYLEM", "turev": "EYLEM", "sayi": 542, "ornekler": "oyna-n-, alkışla-n-", "tur": "E→E", "aciklama": "Edilgen çatı -n allomorf (7. en sık)"},
    {"kod": "N138", "ek": "-(I)n- [dönüşlü-2]", "taban": "EYLEM", "turev": "EYLEM", "sayi": 33, "ornekler": "bakın-", "tur": "E→E", "aciklama": "Dönüşlü çatı varyantı"},
    {"kod": "N139", "ek": "-(I)n", "taban": "AD/AD.SIFAT/İLGEÇ", "turev": "BELİRTEÇ/İLGEÇ", "sayi": 13, "ornekler": "kışın, karşın", "tur": "A→A", "aciklama": "İsimden belirteç/edat"},
    {"kod": "N140", "ek": "-(I)nAk", "taban": "EYLEM/AD", "turev": "AD", "sayi": 25, "ornekler": "seçenek, gözenek", "tur": "E→A", "aciklama": "Fiilden isim (olasılık)"},
    {"kod": "N141", "ek": "-(I)ncI", "taban": "AD.SIFAT", "turev": "SIFAT", "sayi": 8, "ornekler": "birinci, altıncı", "tur": "A→A", "aciklama": "Sıra sayısı"},
    {"kod": "N142", "ek": "-(I)nç", "taban": "EYLEM", "turev": "AD/SIFAT", "sayi": 12, "ornekler": "korkunç, ezinç", "tur": "E→A", "aciklama": "Duygu/durum ismi"},
    {"kod": "N143", "ek": "-(I)nIz", "taban": "BELİRTEÇ", "turev": "ADIL", "sayi": 1, "ornekler": "hepiniz", "tur": "A→A", "aciklama": "Zamir"},
    {"kod": "N144", "ek": "-(I)ntI", "taban": "EYLEM/AD", "turev": "AD", "sayi": 71, "ornekler": "girinti, süprüntü, kazıntı", "tur": "E→A", "aciklama": "Sonuç/kalıntı ismi"},
    # === P-R Serisi (P145-R156) ===
    {"kod": "P145", "ek": "-p-", "taban": "EYLEM", "turev": "EYLEM", "sayi": 2, "ornekler": "serp-", "tur": "E→E", "aciklama": "Fiilden fiil"},
    {"kod": "P146", "ek": "-pAk", "taban": "EYLEM", "turev": "SIFAT", "sayi": 1, "ornekler": "kaypak", "tur": "E→A", "aciklama": "Fiilden sıfat"},
    {"kod": "R147", "ek": "-(A/I)r- [ettirgen]", "taban": "EYLEM", "turev": "EYLEM", "sayi": 26, "ornekler": "kopar-, doyur-", "tur": "E→E", "aciklama": "Ettirgen çatı (tek heceli)"},
    {"kod": "R148", "ek": "-(A/I)r- [ettirgen-2]", "taban": "EYLEM", "turev": "EYLEM", "sayi": 5, "ornekler": "aşır-", "tur": "E→E", "aciklama": "Ettirgen çatı varyantı"},
    {"kod": "R149", "ek": "-(A/I)(l)", "taban": "EYLEM", "turev": "AD", "sayi": 23, "ornekler": "kaynar, iter", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "R150", "ek": "-(A/I)r", "taban": "AD.SIFAT/AD/SIFAT", "turev": "EYLEM", "sayi": 20, "ornekler": "önder, yaşar", "tur": "A→A", "aciklama": "İsimden isim/sıfat"},
    {"kod": "R151", "ek": "-(I)rA", "taban": "ADIL/AD.SIFAT", "turev": "AD", "sayi": 4, "ornekler": "şiire?", "tur": "A→A", "aciklama": "Nadir"},
    {"kod": "R152", "ek": "-rA-", "taban": "AD", "turev": "EYLEM", "sayi": 1, "ornekler": "şakırda-", "tur": "A→E", "aciklama": "Yansıma fiili"},
    {"kod": "R153", "ek": "-rAk", "taban": "AD.SIFAT/SIFAT", "turev": "SIFAT", "sayi": 14, "ornekler": "acırak, yeğrek", "tur": "A→A", "aciklama": "Karşılaştırma"},
    {"kod": "R154", "ek": "-(I)rI", "taban": "EYLEM", "turev": "AD", "sayi": 3, "ornekler": "yumru", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "R155", "ek": "-(I)rIk", "taban": "EYLEM", "turev": "AD", "sayi": 2, "ornekler": "yumruk", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "R156", "ek": "-(A/I)rlIk", "taban": "EYLEM", "turev": "AD", "sayi": 14, "ornekler": "yürürlük, geçerlik", "tur": "E→A", "aciklama": "Durum/yer ismi"},
    # === S Serisi (S157-S167) ===
    {"kod": "S157", "ek": "-sA-", "taban": "AD/SIFAT", "turev": "EYLEM", "sayi": 38, "ornekler": "önemse-, ağırsa-", "tur": "A→E", "aciklama": "Psikolojik fiil"},
    {"kod": "S158", "ek": "-(y)sA", "taban": "AD.BELİRTEÇ/ADIL", "turev": "BAĞLAÇ", "sayi": 5, "ornekler": "nedense, yoksa", "tur": "A→A", "aciklama": "Bağlaç türetme"},
    {"kod": "S159", "ek": "-sAk", "taban": "SIFAT/AD/EYLEM", "turev": "SIFAT", "sayi": 8, "ornekler": "ıraksak", "tur": "A→A", "aciklama": "Nitelik sıfatı"},
    {"kod": "S160", "ek": "-sAl", "taban": "AD/EYLEM/SIFAT", "turev": "SIFAT", "sayi": 179, "ornekler": "bilimsel, evrensel, ulusal", "tur": "A→A", "aciklama": "İlişki sıfatı (17. en sık)"},
    {"kod": "S161", "ek": "-sI(l)", "taban": "AD/SIFAT/EYLEM", "turev": "SIFAT", "sayi": 61, "ornekler": "destansı, genizsi", "tur": "A→A", "aciklama": "Benzerlik/aidiyet"},
    {"kod": "S162", "ek": "-sI(l)-", "taban": "SIFAT/AD.SIFAT", "turev": "EYLEM", "sayi": 2, "ornekler": "yadsı-?", "tur": "A→E", "aciklama": "Nadir"},
    {"kod": "S163", "ek": "-sIl", "taban": "AD", "turev": "SIFAT/AD", "sayi": 8, "ornekler": "dudaksıl", "tur": "A→A", "aciklama": "Nitelik"},
    {"kod": "S164", "ek": "-sIn", "taban": "SIFAT/AD", "turev": "SIFAT", "sayi": 2, "ornekler": "dürüstsün?", "tur": "A→A", "aciklama": "Nadir"},
    {"kod": "S165", "ek": "-sIn-", "taban": "AD", "turev": "EYLEM", "sayi": 3, "ornekler": "yoksun-", "tur": "A→E", "aciklama": "İsimden fiil"},
    {"kod": "S166", "ek": "-sIz", "taban": "AD/SIFAT", "turev": "SIFAT", "sayi": 748, "ornekler": "güçsüz, tuzsuz, sessiz", "tur": "A→A", "aciklama": "Yokluk/yoksunluk (5. en sık)"},
    {"kod": "S167", "ek": "-sIzlAr", "taban": "AD", "turev": "AD", "sayi": 6, "ornekler": "evsizler", "tur": "A→A", "aciklama": "Çoğul yokluk"},
    # === Ş Serisi (Ş168-Ş171) ===
    {"kod": "Ş168", "ek": "-(I)ş-", "taban": "EYLEM", "turev": "EYLEM", "sayi": 90, "ornekler": "gülüş-, bakış-, dövüş-", "tur": "E→E", "aciklama": "İşteş çatı"},
    {"kod": "Ş169", "ek": "-(I)ş- [2]", "taban": "EYLEM", "turev": "EYLEM", "sayi": 24, "ornekler": "buruş-", "tur": "E→E", "aciklama": "Dönüşlü varyant"},
    {"kod": "Ş170", "ek": "-(I)şIn", "taban": "AD.SIFAT", "turev": "SIFAT", "sayi": 3, "ornekler": "sarışın", "tur": "A→A", "aciklama": "Nitelik sıfatı"},
    {"kod": "Ş171", "ek": "-(I)ştIr-", "taban": "EYLEM", "turev": "EYLEM", "sayi": 10, "ornekler": "kıpıştır-", "tur": "E→E", "aciklama": "Bileşik ettirgen"},
    # === T Serisi (T172-T185) ===
    {"kod": "T172", "ek": "-(I)t-", "taban": "EYLEM", "turev": "EYLEM", "sayi": 374, "ornekler": "ürküt-, ısıt-, yenilet-", "tur": "E→E", "aciklama": "Ettirgen çatı (10. en sık)"},
    {"kod": "T173", "ek": "-(I)t- [2]", "taban": "EYLEM", "turev": "EYLEM", "sayi": 5, "ornekler": "aşıt-?", "tur": "E→E", "aciklama": "Ettirgen varyant"},
    {"kod": "T174", "ek": "-(I)t", "taban": "EYLEM/AD/SIFAT", "turev": "AD", "sayi": 29, "ornekler": "geçit, boyut", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "T175", "ek": "-TA", "taban": "AD/SIFAT.BELİRTEÇ", "turev": "BELİRTEÇ/AD.SIFAT", "sayi": 8, "ornekler": "ileride, görünüşte", "tur": "A→A", "aciklama": "Yer/durum belirteci"},
    {"kod": "T176", "ek": "-TAm", "taban": "AD", "turev": "AD", "sayi": 1, "ornekler": "yöntem", "tur": "A→A", "aciklama": "İsimden isim"},
    {"kod": "T177", "ek": "-TAn", "taban": "AD/SIFAT/BELİRTEÇ", "turev": "BELİRTEÇ/SIFAT", "sayi": 32, "ornekler": "yürekten, hafiften", "tur": "A→A", "aciklama": "Ayrılma belirteci"},
    {"kod": "T178", "ek": "-TAr-", "taban": "EYLEM", "turev": "EYLEM", "sayi": 1, "ornekler": "aktar-", "tur": "E→E", "aciklama": "Fiilden fiil"},
    {"kod": "T179", "ek": "-TAş", "taban": "AD/AD.SIFAT", "turev": "AD/SIFAT", "sayi": 36, "ornekler": "yoldaş, okuldaş", "tur": "A→A", "aciklama": "Ortaklık ismi"},
    {"kod": "T180", "ek": "-TAy", "taban": "AD/EYLEM", "turev": "AD", "sayi": 5, "ornekler": "yargıtay, danıştay", "tur": "A→A", "aciklama": "Kurum ismi"},
    {"kod": "T181", "ek": "-tI", "taban": "AD/EYLEM", "turev": "AD", "sayi": 106, "ornekler": "çatırtı, güdüntü", "tur": "E→A", "aciklama": "Ses/sonuç ismi"},
    {"kod": "T182", "ek": "-(I)tI", "taban": "EYLEM", "turev": "AD", "sayi": 8, "ornekler": "alıntı", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "T183", "ek": "-(I)tIkçA", "taban": "EYLEM", "turev": "BELİRTEÇ", "sayi": 2, "ornekler": "oldukça", "tur": "E→A", "aciklama": "Fiilden belirteç"},
    {"kod": "T184", "ek": "-(D)Ir-", "taban": "EYLEM", "turev": "EYLEM", "sayi": 582, "ornekler": "yazdır-, gördür-, bildir-", "tur": "E→E", "aciklama": "Ettirgen çatı ana eki (6. en sık)"},
    {"kod": "T185", "ek": "-(D)Ir- [2]", "taban": "EYLEM", "turev": "EYLEM", "sayi": 15, "ornekler": "andır-", "tur": "E→E", "aciklama": "Ettirgen varyant"},
    # === V-Z Serisi (V186-Z191) ===
    {"kod": "V186", "ek": "-(I)y", "taban": "EYLEM", "turev": "AD", "sayi": 6, "ornekler": "görüy?", "tur": "E→A", "aciklama": "Nadir"},
    {"kod": "V187", "ek": "-vAn", "taban": "EYLEM", "turev": "SIFAT", "sayi": 1, "ornekler": "yayvan", "tur": "E→A", "aciklama": "Fiilden sıfat"},
    {"kod": "Y188", "ek": "-(I)y", "taban": "EYLEM/AD/SIFAT", "turev": "AD", "sayi": 10, "ornekler": "düşey, düzey", "tur": "E→A", "aciklama": "Fiilden isim"},
    {"kod": "Z189", "ek": "-(I)z", "taban": "AD.SIFAT", "turev": "AD.SIFAT", "sayi": 6, "ornekler": "ikiz, yıldız", "tur": "A→A", "aciklama": "İsimden isim"},
    {"kod": "Z190", "ek": "-zIk", "taban": "EYLEM", "turev": "AD", "sayi": 1, "ornekler": "emzik", "tur": "E→A", "aciklama": "Fiilden isim (alet)"},
    {"kod": "Z191", "ek": "-zIr-", "taban": "EYLEM", "turev": "EYLEM", "sayi": 1, "ornekler": "emzir-", "tur": "E→E", "aciklama": "Ettirgen"},
]

# Sıklık sırası (top 20)
TOP20 = [
    {"sira": 1, "kod": "L112", "ek": "-(l)Ik", "sayi": 3259, "oran": 22.27, "tur": "A→A"},
    {"sira": 2, "kod": "L111", "ek": "-(l)I", "sayi": 1644, "oran": 11.23, "tur": "A→A"},
    {"sira": 3, "kod": "Ç39", "ek": "-çI", "sayi": 1259, "oran": 8.60, "tur": "A→A"},
    {"sira": 4, "kod": "L95", "ek": "-(l)A-", "sayi": 840, "oran": 5.74, "tur": "A→E"},
    {"sira": 5, "kod": "S166", "ek": "-sIz", "sayi": 748, "oran": 5.11, "tur": "A→A"},
    {"sira": 6, "kod": "T184", "ek": "-(D)Ir-", "sayi": 582, "oran": 3.98, "tur": "E→E"},
    {"sira": 7, "kod": "N137", "ek": "-(I)n- [edilgen]", "sayi": 542, "oran": 3.70, "tur": "E→E"},
    {"sira": 8, "kod": "L107", "ek": "-(l)Aş-", "sayi": 518, "oran": 3.54, "tur": "A→E"},
    {"sira": 9, "kod": "I61", "ek": "-(I)l-", "sayi": 404, "oran": 2.76, "tur": "E→E"},
    {"sira": 10, "kod": "T172", "ek": "-(I)t-", "sayi": 374, "oran": 2.56, "tur": "E→E"},
    {"sira": 11, "kod": "Ç36", "ek": "-çA", "sayi": 344, "oran": 2.35, "tur": "A→A"},
    {"sira": 12, "kod": "L101", "ek": "-(l)An-", "sayi": 277, "oran": 1.89, "tur": "A→E"},
    {"sira": 13, "kod": "K78", "ek": "-(A/I)k", "sayi": 277, "oran": 1.89, "tur": "E→A"},
    {"sira": 14, "kod": "I57", "ek": "-IcI", "sayi": 255, "oran": 1.74, "tur": "E→A"},
    {"sira": 15, "kod": "G53", "ek": "-gIller", "sayi": 240, "oran": 1.64, "tur": "A→A"},
    {"sira": 16, "kod": "I64", "ek": "-(y)Im", "sayi": 202, "oran": 1.38, "tur": "E→A"},
    {"sira": 17, "kod": "S160", "ek": "-sAl", "sayi": 179, "oran": 1.22, "tur": "A→A"},
    {"sira": 18, "kod": "L103", "ek": "-(l)Ar", "sayi": 157, "oran": 1.07, "tur": "A→A"},
    {"sira": 19, "kod": "M115", "ek": "-mA", "sayi": 123, "oran": 0.84, "tur": "E→A"},
    {"sira": 20, "kod": "N136", "ek": "-(I)n- [dönüşlü]", "sayi": 121, "oran": 0.83, "tur": "E→E"},
]

# Çatı ekleri
CATI_EKLERI = [
    {"cati": "Ettirgen-1 (ana)", "ek": "-(D)Ir-", "kod": "T184", "sayi": 582, "aciklama": "Ana ettirgen eki: yazdır-, gördür-", "allomorf": "Ünsüzden sonra: -DIr-, ünlüden sonra: -t-"},
    {"cati": "Edilgen-2 (n-allomorf)", "ek": "-(I)n-", "kod": "N137", "sayi": 542, "aciklama": "Ünlüyle biten gövdelerden: oyna-n-", "allomorf": "Ünlüden sonra: -n-"},
    {"cati": "Edilgen-1 (l-allomorf)", "ek": "-(I)l-", "kod": "I61", "sayi": 404, "aciklama": "Ünsüzle biten gövdelerden: yazıl-", "allomorf": "Ünsüzden sonra: -Il-"},
    {"cati": "Ettirgen-2 (t-allomorf)", "ek": "-(I)t-", "kod": "T172", "sayi": 374, "aciklama": "Çok heceli gövdelerden: ürküt-", "allomorf": "Çok heceli veya türemiş gövde"},
    {"cati": "Dönüşlü", "ek": "-(I)n-", "kod": "N136", "sayi": 121, "aciklama": "Gerçek dönüşlü: giy-in-, yıka-n-", "allomorf": "Edilgen -(I)n- ile eşbiçimli"},
    {"cati": "İşteş", "ek": "-(I)ş-", "kod": "Ş168", "sayi": 90, "aciklama": "Karşılıklık: gülüş-, bakış-", "allomorf": "Tek biçim"},
    {"cati": "Ettirgen-3 (r-allomorf)", "ek": "-(A/I)r-", "kod": "R147", "sayi": 26, "aciklama": "Tek heceli: kop-ar-, doy-ur-", "allomorf": "Tek heceli köklerden"},
]

# UPOS çıkarım tablosu
UPOS_CIKARIM = {
    "NOUN": {
        "ekler": ["-(l)Ik", "-çI", "-(y)Im", "-mA", "-(y)I", "-In", "-(I)ntI", "-tI", "-(I)t", "-mAç", "-mAcA"],
        "guvenilirlik": "Yüksek",
        "aciklama": "Bu eklerle türeyen sözcükler yüksek olasılıkla isimdir",
    },
    "ADJ": {
        "ekler": ["-(l)I", "-sIz", "-sAl", "-(A/I)k", "-KAn", "-KIn", "-(I)msI", "-çI(l)", "-IcI"],
        "guvenilirlik": "Yüksek",
        "aciklama": "Bu eklerle türeyen sözcükler yüksek olasılıkla sıfattır",
    },
    "ADV": {
        "ekler": ["-çA", "-(s)InA", "-(I)ncA", "-leyIn", "-(y)lA", "-AsIyA"],
        "guvenilirlik": "Orta",
        "aciklama": "Bu eklerle türeyen sözcükler genellikle zarftır, ancak bağlama bağlı",
    },
    "VERB": {
        "ekler": ["-(l)A-", "-(l)An-", "-(l)Aş-", "-(D)Ir-", "-(I)l-", "-(I)n-", "-(I)ş-", "-(I)t-", "-dA-", "-sA-"],
        "guvenilirlik": "Yüksek",
        "aciklama": "Bu eklerle türeyen sözcükler fiildir (çatı ekleri dahil)",
    },
}

# Deprel çıkarım tablosu
DEPREL_CIKARIM = {
    "amod": {"kalip": "Sıfat türeten ekler", "ekler": ["-(l)I", "-sIz", "-sAl", "-(A/I)k", "-KAn", "-KIn", "-(I)msI"]},
    "advmod": {"kalip": "Zarf türeten ekler", "ekler": ["-çA", "-(s)InA", "-(I)ncA", "-leyIn"]},
    "nsubj/obj/obl": {"kalip": "Fiilden isim türetenler", "ekler": ["-(y)Im", "-mA", "-(l)Ik", "-IcI"]},
    "root/advcl/xcomp": {"kalip": "İsimden fiil türetenler", "ekler": ["-(l)A-", "-(l)An-", "-(l)Aş-"]},
    "çatı bilgisi": {"kalip": "Ettirgen/edilgen/işteş", "ekler": ["-(D)Ir-", "-(I)l-", "-(I)n-", "-(I)ş-", "-(I)t-"]},
}

# Projede eksik ekler
EKSIK_EKLER = [
    {"ek": "-çA", "kod": "Ç36", "sayi": 344, "etki": "Türkçe, boyunca, İngilizce — dil/ölçü eki", "oncelik": "Yüksek"},
    {"ek": "-sAl", "kod": "S160", "sayi": 179, "etki": "bilimsel, evrensel — ilişki sıfatı", "oncelik": "Yüksek"},
    {"ek": "-(A/I)k", "kod": "K78", "sayi": 277, "etki": "kıyak, soluk — çok verimli", "oncelik": "Yüksek"},
    {"ek": "-KAn", "kod": "K83", "sayi": 74, "etki": "konuşkan, çalışkan — eğilim sıfatı", "oncelik": "Orta"},
    {"ek": "-KIn", "kod": "K87", "sayi": 84, "etki": "seçkin, yetişkin — nitelik sıfatı", "oncelik": "Orta"},
    {"ek": "-mAn", "kod": "M122", "sayi": 14, "etki": "öğretmen, seçmen — fail/kişi ismi", "oncelik": "Orta"},
    {"ek": "-(I)msI", "kod": "M134", "sayi": 45, "etki": "mavimsi, yeşilimsi — benzerlik", "oncelik": "Orta"},
    {"ek": "-(I)mtrAk", "kod": "M135", "sayi": 8, "etki": "yeşilimtırak — yaklaşıklık", "oncelik": "Düşük"},
    {"ek": "-(I)ncI", "kod": "N141", "sayi": 8, "etki": "birinci, altıncı — sıra sayıları", "oncelik": "Orta"},
    {"ek": "-(y)Im", "kod": "I64", "sayi": 202, "etki": "yazım, çizim — fiilden isim", "oncelik": "Yüksek"},
    {"ek": "-çIk", "kod": "Ç40", "sayi": 80, "etki": "kapıcık — küçültme", "oncelik": "Düşük"},
    {"ek": "-gIller", "kod": "G53", "sayi": 240, "etki": "gülgiller — biyolojik cins", "oncelik": "Düşük"},
]


# ============================================================
# ARAÇLAR (TOOLS)
# ============================================================

@mcp.tool()
def ek_ara(sorgu: str) -> str:
    """Türetim eki ara — kod (ör. 'L112'), ek formu (ör. '-lIk', '-çI') veya
    açıklama/örnek ile arama yapar. Birden fazla sonuç dönebilir."""
    sorgu_lower = sorgu.lower().replace("ı", "i").replace("ö", "o").replace("ü", "u").replace("ç", "c").replace("ş", "s").replace("ğ", "g")
    sonuclar = []

    for e in EKLER:
        # Kod eşleşmesi
        if sorgu.upper() == e["kod"]:
            sonuclar.append(e)
            continue
        # Normalize edip arama
        ek_norm = e["ek"].lower().replace("ı", "i").replace("ö", "o").replace("ü", "u").replace("ç", "c").replace("ş", "s").replace("ğ", "g")
        orn_norm = e["ornekler"].lower().replace("ı", "i").replace("ö", "o").replace("ü", "u").replace("ç", "c").replace("ş", "s").replace("ğ", "g")
        acik_norm = e["aciklama"].lower().replace("ı", "i").replace("ö", "o").replace("ü", "u").replace("ç", "c").replace("ş", "s").replace("ğ", "g")
        if sorgu_lower in ek_norm or sorgu_lower in orn_norm or sorgu_lower in acik_norm:
            sonuclar.append(e)

    if not sonuclar:
        return f"'{sorgu}' için sonuç bulunamadı."

    lines = [f"## '{sorgu}' için {len(sonuclar)} sonuç:\n"]
    for e in sonuclar:
        lines.append(
            f"**{e['kod']}** {e['ek']}\n"
            f"  Taban: {e['taban']} → Türev: {e['turev']} ({e['tur']})\n"
            f"  Örnek sayısı: {e['sayi']} — {e['ornekler']}\n"
            f"  Açıklama: {e['aciklama']}\n"
        )
    return "\n".join(lines)


@mcp.tool()
def ek_filtrele(
    turetim_turu: str = "",
    taban_turu: str = "",
    min_sayi: int = 0,
    max_sayi: int = 999999,
) -> str:
    """Türetim eklerini filtrele.
    turetim_turu: 'A→A', 'A→E', 'E→A', 'E→E' (boş = hepsi)
    taban_turu: 'AD', 'EYLEM', 'SIFAT' vs. (boş = hepsi)
    min_sayi / max_sayi: Örnek sayısı aralığı"""
    sonuclar = []
    for e in EKLER:
        if turetim_turu and e["tur"] != turetim_turu:
            continue
        if taban_turu and taban_turu.upper() not in e["taban"].upper():
            continue
        if not (min_sayi <= e["sayi"] <= max_sayi):
            continue
        sonuclar.append(e)

    sonuclar.sort(key=lambda x: x["sayi"], reverse=True)

    if not sonuclar:
        return "Filtreye uyan ek bulunamadı."

    lines = [f"## Filtre sonucu: {len(sonuclar)} ek\n"]
    for e in sonuclar[:50]:  # İlk 50
        lines.append(f"| {e['kod']} | {e['ek']} | {e['taban']}→{e['turev']} | {e['sayi']} | {e['ornekler'][:40]} |")

    if len(sonuclar) > 50:
        lines.append(f"\n... ve {len(sonuclar) - 50} ek daha")
    return "\n".join(lines)


@mcp.tool()
def en_verimli_ekler(n: int = 20) -> str:
    """En sık kullanılan türetim eklerini sıklık sırasıyla döndürür.
    n: Kaç ek gösterilsin (varsayılan: 20, max: 191)"""
    n = min(n, len(EKLER))
    sirali = sorted(EKLER, key=lambda x: x["sayi"], reverse=True)[:n]

    lines = [f"## En Verimli {n} Türetim Eki (14.635 maddebaşından)\n"]
    lines.append("| Sıra | Kod | Ek | Sayı | Oran | Tür | Açıklama |")
    lines.append("|------|-----|-----|------|------|-----|----------|")

    toplam = 14635
    for i, e in enumerate(sirali, 1):
        oran = (e["sayi"] / toplam) * 100
        lines.append(
            f"| {i} | {e['kod']} | {e['ek']} | {e['sayi']} | %{oran:.1f} | {e['tur']} | {e['aciklama'][:50]} |"
        )

    ilk5_toplam = sum(e["sayi"] for e in sirali[:5])
    lines.append(f"\n**İlk 5 ek toplam maddebaşlarının %{(ilk5_toplam/toplam)*100:.1f}'ini kapsar.**")
    return "\n".join(lines)


@mcp.tool()
def turetim_turu_dagilimi() -> str:
    """Türetim türü dağılımını döndürür: A→A, A→E, E→A, E→E oranları."""
    dagılım = {"A→A": 0, "A→E": 0, "E→A": 0, "E→E": 0}
    for e in EKLER:
        if e["tur"] in dagılım:
            dagılım[e["tur"]] += e["sayi"]

    toplam = sum(dagılım.values())
    lines = ["## Türetim Türü Dağılımı\n"]
    lines.append("| Türetim Türü | Toplam Maddebaşı | Oran |")
    lines.append("|-------------|-----------------|------|")
    for tur, sayi in sorted(dagılım.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {tur} | {sayi} | %{(sayi/toplam)*100:.1f} |")
    lines.append(f"| **Toplam** | **{toplam}** | **%100** |")

    lines.append("\n### Açıklama:")
    lines.append("- **A→A (Adsıl→Adsıl)**: İsimden isim/sıfat — en büyük grup")
    lines.append("- **E→E (Eylemcil→Eylemcil)**: Fiilden fiil — çatı ekleri burada")
    lines.append("- **E→A (Eylemcil→Adsıl)**: Fiilden isim/sıfat")
    lines.append("- **A→E (Adsıl→Eylemcil)**: İsimden fiil")
    return "\n".join(lines)


@mcp.tool()
def cati_ekleri_analiz() -> str:
    """Türkçe çatı eklerinin (ettirgen, edilgen, dönüşlü, işteş) detaylı analizi.
    Allomorf dağılımı ve disambiguasyon kurallarını içerir."""
    lines = ["## Türkçe Çatı Ekleri Analizi\n"]
    lines.append("| Çatı | Ek | Kod | Sayı | Allomorf | Açıklama |")
    lines.append("|------|-----|-----|------|----------|----------|")

    for c in CATI_EKLERI:
        lines.append(
            f"| {c['cati']} | {c['ek']} | {c['kod']} | {c['sayi']} | {c['allomorf']} | {c['aciklama']} |"
        )

    lines.append("\n### Ettirgen Allomorf Seçim Kuralları:")
    lines.append("1. **-(D)Ir-** (T184): Ana ek — ünsüzle biten gövdelerden: yaz→yazdır-")
    lines.append("2. **-(I)t-** (T172): Çok heceli veya türemiş gövdelerden: ürk→ürküt-")
    lines.append("3. **-(A/I)r-** (R147): Tek heceli köklerden: kop→kopar-")

    lines.append("\n### Edilgen Allomorf Seçim Kuralları:")
    lines.append("1. **-(I)l-** (I61): Ünsüzle biten gövdelerden: yaz→yazıl-")
    lines.append("2. **-(I)n-** (N137): Ünlüyle biten gövdelerden: oyna→oynan-")

    lines.append("\n### ⚠️ Disambiguasyon Sorunu:")
    lines.append("Edilgen -(I)n- (N137, 542 örnek) ile dönüşlü -(I)n- (N136, 121 örnek)")
    lines.append("biçimsel olarak aynıdır — ayrıştırma sözlüksel bağlam gerektirir.")
    return "\n".join(lines)


@mcp.tool()
def upos_cikarimi(ek: str = "") -> str:
    """Türetim ekinden UPOS (Universal Part-of-Speech) çıkarım tablosu.
    ek parametresi verilirse o ekin UPOS tahmini döner.
    Boş bırakılırsa tüm tablo gösterilir."""
    if ek:
        ek_clean = ek.lower().strip("-").replace("(", "").replace(")", "").replace("/", "")
        for upos, bilgi in UPOS_CIKARIM.items():
            for e in bilgi["ekler"]:
                e_clean = e.lower().strip("-").replace("(", "").replace(")", "").replace("/", "")
                if ek_clean in e_clean or e_clean in ek_clean:
                    return (
                        f"## '{ek}' → {upos}\n"
                        f"Güvenilirlik: {bilgi['guvenilirlik']}\n"
                        f"{bilgi['aciklama']}\n"
                        f"Aynı gruptaki ekler: {', '.join(bilgi['ekler'])}"
                    )
        return f"'{ek}' için UPOS çıkarım bilgisi bulunamadı."

    lines = ["## UPOS Çıkarım Tablosu\n"]
    for upos, bilgi in UPOS_CIKARIM.items():
        lines.append(f"### {upos} (güvenilirlik: {bilgi['guvenilirlik']})")
        lines.append(f"Ekler: {', '.join(bilgi['ekler'])}")
        lines.append(f"{bilgi['aciklama']}\n")
    return "\n".join(lines)


@mcp.tool()
def deprel_cikarimi(ek: str = "") -> str:
    """Türetim ekinden dependency relation (deprel) çıkarım tablosu.
    ek parametresi verilirse o ekin olası deprel'leri döner."""
    if ek:
        ek_clean = ek.lower().strip("-").replace("(", "").replace(")", "").replace("/", "")
        for deprel, bilgi in DEPREL_CIKARIM.items():
            for e in bilgi["ekler"]:
                e_clean = e.lower().strip("-").replace("(", "").replace(")", "").replace("/", "")
                if ek_clean in e_clean or e_clean in ek_clean:
                    return (
                        f"## '{ek}' → olası deprel: {deprel}\n"
                        f"Kalıp: {bilgi['kalip']}\n"
                        f"Aynı gruptaki ekler: {', '.join(bilgi['ekler'])}"
                    )
        return f"'{ek}' için deprel çıkarım bilgisi bulunamadı."

    lines = ["## Deprel Çıkarım Tablosu\n"]
    for deprel, bilgi in DEPREL_CIKARIM.items():
        lines.append(f"### {deprel}")
        lines.append(f"Kalıp: {bilgi['kalip']}")
        lines.append(f"Ekler: {', '.join(bilgi['ekler'])}\n")
    return "\n".join(lines)


@mcp.tool()
def eksik_ek_analizi(oncelik: str = "") -> str:
    """Projede eksik veya geliştirilebilir ekleri gösterir.
    oncelik: 'Yüksek', 'Orta', 'Düşük' (boş = hepsi)"""
    ekler = EKSIK_EKLER
    if oncelik:
        ekler = [e for e in ekler if e["oncelik"] == oncelik]

    if not ekler:
        return f"'{oncelik}' önceliğinde eksik ek bulunamadı."

    lines = [f"## Projede Eksik Ekler ({len(ekler)} adet)\n"]
    lines.append("| Ek | Kod | Örnek Sayısı | Öncelik | Etki |")
    lines.append("|----|-----|-------------|---------|------|")
    for e in sorted(ekler, key=lambda x: x["sayi"], reverse=True):
        lines.append(f"| {e['ek']} | {e['kod']} | {e['sayi']} | {e['oncelik']} | {e['etki'][:60]} |")

    lines.append("\n### Proje-Kitap Etiket Eşleştirmesi:")
    lines.append("| Proje Etiketi | Kitap # | Sayı |")
    lines.append("|---------------|---------|------|")
    lines.append("| ETTİRGEN | T184 | 582 |")
    lines.append("| EDİLGEN | I61 | 404 |")
    lines.append("| İŞTEŞ | Ş168 | 90 |")
    lines.append("| YAPIM_-CI | Ç39 | 1259 |")
    lines.append("| YAPIM_-lI | L111 | 1644 |")
    lines.append("| YAPIM_-lIk | L112 | 3259 |")
    lines.append("| YAPIM_-sIz | S166 | 748 |")
    lines.append("| YAPIM_-lAn | L101 | 277 |")
    lines.append("| YAPIM_-lAş | L107 | 518 |")
    return "\n".join(lines)


@mcp.tool()
def istatistikler() -> str:
    """Kitaptaki tüm istatistiksel bulguları döndürür:
    yerli/yabancı, yaş, taban yapısı, toplam rakamlar."""
    return """## Uzun et al. (1992) — İstatistiksel Bulgular

### Genel
- **Toplam türetimsel maddebaşı:** 14.635
- **Toplam türetim eki:** 191
- **Kaynak:** TDK Türkçe Sözlük, 7. Baskı

### Taban Türü Dağılımı
| Taban Türü | Oran |
|-----------|------|
| AD (isim) | %50.15 |
| EYLEM (fiil) | %28.49 |
| SIFAT | ~%10 |
| AD.SIFAT | ~%5.5 |
| BELİRTEÇ | ~%2.7 |
| Diğer (ADIL, İLGEÇ, BAĞLAÇ) | ~%2.9 |

### Taban Yapısı
| Yapı | Oran |
|------|------|
| Yalın taban | ~%85 |
| Bileşik taban | ~%10 |
| Yansıma taban | ~%5 |

### Köken Dağılımı
| Köken | Oran |
|-------|------|
| Yerli (öz Türkçe) | ~%70-75 |
| Yabancı (alıntı) | ~%25-30 |

### Yaş Dağılımı
| Yaş | Oran |
|-----|------|
| Eski (yerleşik) | ~%76.6 |
| Yeni (son dönem) | ~%23.4 |

### Verimlilik
- İlk 5 ek toplam maddebaşlarının **%52.3**'ünü kapsar
- İlk 20 ek toplam maddebaşlarının **%82.3**'ünü kapsar
- Kalan 171 ek yalnızca **%17.7** oranında

### İÇİ (İçerikle Çözümleme İlişkisi)
Kitabın temel yöntemi: Taban ve türev arasında sözlüksel içerik
ilişkisi kurulabilmelidir. Taban, sözlükte yalın maddebaşı olarak
bulunmalıdır. Bu kısıtlama sezgisel çözümlemeyi dışlar.
"""


@mcp.tool()
def morfolojik_kurallar() -> str:
    """Kitaba dayalı morfolojik çözümleme kuralları:
    taban belirleme, ek sıralama, allomorf seçimi."""
    return """## Morfolojik Çözümleme Kuralları

### 1. Taban Belirleme İlkeleri
1. **İÇİ zorunluluğu**: Taban-türev arasında sözlüksel içerik ilişkisi kurulabilmeli
2. **Sözlük doğrulaması**: Taban, sözlükte yalın maddebaşı olarak bulunmalı
3. **İkili kök sorunu**: göç/göç-, ekşi/ekş- gibi durumda İÇİ tercih belirler
4. **Bileşik ek**: Parçalar başka yerde yoksa tümü tek ek sayılır
5. **Koruyucu ünsüzler**: (y), (n), (s), (ş) — ünlüyle biten tabandan sonra

### 2. Çatı Eki Sıralama
```
[KÖK] + [DÖNÜŞLÜ/EDİLGEN] + [İŞTEŞ] + [ETTİRGEN] + [OLUMSUZ] + [çekim]
```

Zincirleme örnekler:
- yaz + dır + ıl + abil + ecek → yazdırılabilecek (E→E→E→çekim)
- güzel + leş + tir → güzelleştir (A→E→E)
- ev + len + dir + il → evlendiril (A→E→E→E)

### 3. Allomorf Seçim Tablosu
| Ek | Ortam | Allomorf |
|----|-------|----------|
| -(D)Ir- | Tek heceli + ünlü sonu | -(I)r- (kop-ar-) |
| -(D)Ir- | Çok heceli gövde | -(I)t- (ürk-üt-) |
| -(I)l- | Ünsüzle biten | -(I)l- (yaz-ıl-) |
| -(I)n- | Ünlüyle biten | -(I)n- (oyna-n-) |
| -(l)A- | Ünlüyle biten | -lA- (taş-la-) |
| -(l)I | Ünlüyle biten | -lI (tuz-lu) |
| -(l)Ik | Ünlüyle biten | -lIk (güzel-lik) |

### 4. BÜU (Büyük Ünlü Uyumu) — 2 yönlü
| Son ünlü | Ek ünlüsü |
|----------|-----------|
| a, ı, o, u (kalın) | a |
| e, i, ö, ü (ince) | e |

### 5. KÜU (Küçük Ünlü Uyumu) — 4 yönlü
| Son ünlü | Ek ünlüsü |
|----------|-----------|
| a, e | ı, i |
| ı, i | ı, i |
| o, ö | u, ü |
| u, ü | u, ü |
"""


@mcp.tool()
def tam_envanter_ozet() -> str:
    """191 ekin seri bazında özet tablosu (her seriyi sayıları ile gösterir)."""
    seriler = {}
    for e in EKLER:
        seri = e["kod"][0]
        if seri not in seriler:
            seriler[seri] = {"adet": 0, "toplam_ornek": 0, "ekler": []}
        seriler[seri]["adet"] += 1
        seriler[seri]["toplam_ornek"] += e["sayi"]
        seriler[seri]["ekler"].append(e["ek"])

    lines = ["## 191 Türetim Eki — Seri Bazında Özet\n"]
    lines.append("| Seri | Ek Sayısı | Toplam Örnek | Örnekler |")
    lines.append("|------|-----------|-------------|----------|")

    for seri in sorted(seriler.keys()):
        info = seriler[seri]
        ornek_ekler = ", ".join(info["ekler"][:5])
        if len(info["ekler"]) > 5:
            ornek_ekler += f" ... (+{len(info['ekler'])-5})"
        lines.append(f"| {seri} | {info['adet']} | {info['toplam_ornek']} | {ornek_ekler} |")

    lines.append(f"\n**Toplam: {len(EKLER)} ek, {sum(e['sayi'] for e in EKLER)} maddebaşı**")
    lines.append("\n### Kaynak")
    lines.append("Uzun, N.E., Uzun, L.S., Aksan, Y.K. & Aksan, M. (1992).")
    lines.append("*Türkiye Türkçesinin Türetim Ekleri — Bir Döküm Denemesi*. Ankara.")
    return "\n".join(lines)


# ============================================================
# AKTİF MOTOR ARAÇLARI (ACTIVE ANALYSIS TOOLS)
# ============================================================

@mcp.tool()
def sozcuk_cozumle(sozcuk: str) -> str:
    """Bir Türkçe sözcüğü morfolojik olarak çözümler.
    Kök, ekler, lemma bilgisini döndürür.
    Örnek: sozcuk_cozumle('evlerinden') → ev + ÇOĞUL + İYELİK + AYRILMA"""
    try:
        analyzer = _get_analyzer()
        result = analyzer.analyze(sozcuk)
        lemma = result.lemma or result.root or result.stem

        lines = [f"## Morfolojik Çözümleme: {sozcuk}\n"]
        lines.append(f"**Kök (stem):** {result.stem}")
        lines.append(f"**Lemma:** {lemma}")

        if result.suffixes:
            lines.append(f"\n**Ekler ({len(result.suffixes)}):**")
            for form, label in result.suffixes:
                lines.append(f"  - `{form}` → {label}")
            chain = f"{result.stem} + " + " + ".join(
                f"-{form}({label})" for form, label in result.suffixes
            )
            lines.append(f"\n**Zincir:** {chain}")
        else:
            lines.append("\n**Ekler:** (yok — kök sözcük)")

        # Parçalar
        parts = result.parts if hasattr(result, "parts") else [result.stem] + [s[0] for s in result.suffixes]
        lines.append(f"\n**Parçalar:** {' | '.join(parts)}")

        return "\n".join(lines)
    except Exception as e:
        return f"Çözümleme hatası: {e}"


@mcp.tool()
def sozcuk_tum_cozumlemeler(sozcuk: str, max_sonuc: int = 5) -> str:
    """Bir sözcüğün tüm olası morfolojik çözümlemelerini döndürür.
    Belirsizlik durumlarında (ör. 'yazar' = yaz+ar mı, yazar mı?)
    tüm alternatifleri sıralı gösterir."""
    try:
        analyzer = _get_analyzer()
        results = analyzer.analyze_all(sozcuk, max_results=max_sonuc)

        if not results:
            return f"'{sozcuk}' için çözümleme bulunamadı."

        lines = [f"## '{sozcuk}' için {len(results)} çözümleme\n"]
        for i, r in enumerate(results, 1):
            lemma = r.lemma or r.root or r.stem
            if r.suffixes:
                suffix_str = " + ".join(f"-{f}({l})" for f, l in r.suffixes)
                lines.append(f"**{i}.** `{r.stem}` + {suffix_str}  (lemma: {lemma})")
            else:
                lines.append(f"**{i}.** `{r.stem}` (kök sözcük, lemma: {lemma})")
        return "\n".join(lines)
    except Exception as e:
        return f"Çözümleme hatası: {e}"


@mcp.tool()
def cumle_analiz(cumle: str) -> str:
    """Bir Türkçe cümleyi tam olarak analiz eder:
    morfolojik çözümleme + bağlam kuralları + dependency ağacı + CoNLL-U.
    Örnek: cumle_analiz('Ali güzel kitabı okudu.')"""
    try:
        sa = _get_sentence_analyzer()
        dp = _get_dep_parser()

        # Cümle morfoloji
        tokens = sa.analyze(cumle)
        # Dependency parsing
        dep_tokens = dp.parse(tokens, text=cumle)

        lines = [f"## Cümle Analizi: {cumle}\n"]

        # Morfoloji tablosu
        lines.append("### Morfoloji")
        lines.append("| # | Sözcük | Kök | Lemma | Ekler | Bağlam Kuralları |")
        lines.append("|---|--------|-----|-------|-------|-----------------|")
        for i, t in enumerate(tokens, 1):
            lemma = t.analysis.lemma or t.analysis.root or t.analysis.stem
            if t.analysis.suffixes:
                ek_str = ", ".join(f"-{f}({l})" for f, l in t.analysis.suffixes)
            else:
                ek_str = "—"
            ctx = ", ".join(t.context_applied) if t.context_applied else "—"
            lines.append(f"| {i} | {t.word} | {t.analysis.stem} | {lemma} | {ek_str} | {ctx} |")

        # Dependency ağacı
        lines.append("\n### Dependency Ağacı")
        lines.append("```")
        tree = dp.to_tree(dep_tokens)
        lines.append(tree)
        lines.append("```")

        # Dependency tablosu
        lines.append("\n### Dependency Detay")
        lines.append("| ID | Form | Lemma | UPOS | Head | Deprel | Feats |")
        lines.append("|----|------|-------|------|------|--------|-------|")
        for t in dep_tokens:
            feats_s = t.feats_str if hasattr(t, "feats_str") else str(t.feats)
            lines.append(
                f"| {t.id} | {t.form} | {t.lemma} | {t.upos} | {t.head} | {t.deprel} | {feats_s} |"
            )

        # CoNLL-U
        lines.append("\n### CoNLL-U Çıktı")
        lines.append("```conllu")
        lines.append(dp.to_conllu(dep_tokens, text=cumle))
        lines.append("```")

        return "\n".join(lines)
    except Exception as e:
        return f"Cümle analiz hatası: {e}"


@mcp.tool()
def cumle_conllu(cumle: str) -> str:
    """Bir cümleyi CoNLL-U (Universal Dependencies) formatında döndürür.
    Direkt pipeline çıktısı — başka araçlara girdi olarak kullanılabilir."""
    try:
        sa = _get_sentence_analyzer()
        dp = _get_dep_parser()
        tokens = sa.analyze(cumle)
        dep_tokens = dp.parse(tokens, text=cumle)
        return dp.to_conllu(dep_tokens, text=cumle)
    except Exception as e:
        return f"Hata: {e}"


@mcp.tool()
def sozcuk_karsilastir(sozcuk: str, beklenen_lemma: str = "", beklenen_upos: str = "") -> str:
    """Bir sözcüğün çözümlemesini beklenen değerlerle karşılaştırır.
    Gold standard ile pred karşılaştırması yapmak için kullanılır.
    Örnek: sozcuk_karsilastir('kitabı', beklenen_lemma='kitap', beklenen_upos='NOUN')"""
    try:
        analyzer = _get_analyzer()
        result = analyzer.analyze(sozcuk)
        pred_lemma = result.lemma or result.root or result.stem

        lines = [f"## Karşılaştırma: {sozcuk}\n"]
        lines.append(f"**Predicted lemma:** {pred_lemma}")
        lines.append(f"**Predicted stem:** {result.stem}")

        if result.suffixes:
            lines.append(f"**Predicted ekler:** {', '.join(l for _, l in result.suffixes)}")

        if beklenen_lemma:
            match = "✅" if pred_lemma == beklenen_lemma else "❌"
            lines.append(f"\n**Lemma:** {match} beklenen=`{beklenen_lemma}` → tahmin=`{pred_lemma}`")

        if beklenen_upos:
            # UPOS için cümle bağlamı gerekir — tek sözcükle en iyi tahmin
            sa = _get_sentence_analyzer()
            dp = _get_dep_parser()
            tokens = sa.analyze(sozcuk)
            dep_tokens = dp.parse(tokens, text=sozcuk)
            pred_upos = dep_tokens[0].upos if dep_tokens else "?"
            match = "✅" if pred_upos == beklenen_upos else "❌"
            lines.append(f"**UPOS:** {match} beklenen=`{beklenen_upos}` → tahmin=`{pred_upos}`")

        return "\n".join(lines)
    except Exception as e:
        return f"Karşılaştırma hatası: {e}"


@mcp.tool()
def benchmark_calistir(max_cumle: int = 50) -> str:
    """BOUN Treebank üzerinde benchmark çalıştırır.
    UAS, LAS, UPOS doğruluğu ve deprel bazlı kırılımı döndürür.
    max_cumle: Kaç cümle değerlendirilsin (varsayılan: 50, max: 979)"""
    try:
        conllu_path = _PROJECT_ROOT / "benchmark" / "test.conllu"
        if not conllu_path.exists():
            return "benchmark/test.conllu bulunamadı."

        # Import lazily
        sys.path.insert(0, str(_PROJECT_ROOT))
        from benchmark.eval_dep import evaluate

        sa = _get_sentence_analyzer()
        dp = _get_dep_parser()

        max_cumle = min(max(max_cumle, 1), 979)

        results = evaluate(
            conllu_path=conllu_path,
            sa=sa,
            dp=dp,
            max_sentences=max_cumle,
            deprel_matrix=True,
        )

        total = results["total_tokens"]
        if total == 0:
            return "Değerlendirme yapılamadı (0 token)."

        uas_pct = results["uas"] / total * 100
        las_pct = results["las"] / total * 100
        upos_pct = results["upos_correct"] / total * 100
        deprel_pct = results["deprel_correct"] / total * 100

        lines = [f"## Benchmark Sonuçları ({max_cumle} cümle, {total} token)\n"]
        lines.append("| Metrik | Doğru | Toplam | Oran |")
        lines.append("|--------|-------|--------|------|")
        lines.append(f"| **UAS** | {results['uas']} | {total} | **%{uas_pct:.1f}** |")
        lines.append(f"| **LAS** | {results['las']} | {total} | **%{las_pct:.1f}** |")
        lines.append(f"| **UPOS** | {results['upos_correct']} | {total} | **%{upos_pct:.1f}** |")
        lines.append(f"| **Deprel** | {results['deprel_correct']} | {total} | **%{deprel_pct:.1f}** |")
        lines.append(f"\n**Perfect LAS cümle:** {results['sent_perfect_las']}/{results['sent_count']}")
        lines.append(f"**Token uyuşmazlık:** {results['token_mismatch_sents']} cümle")

        # Deprel kırılımı (en büyük 15)
        if results["deprel_counts"]:
            lines.append("\n### Deprel Bazlı Kırılım (Top 15)")
            lines.append("| Deprel | Gold | Hit | Oran |")
            lines.append("|--------|------|-----|------|")
            sorted_deprels = sorted(
                results["deprel_counts"].items(), key=lambda x: x[1], reverse=True
            )
            for deprel, count in sorted_deprels[:15]:
                hits = results["deprel_hits"].get(deprel, 0)
                pct = hits / count * 100 if count else 0
                lines.append(f"| {deprel} | {count} | {hits} | %{pct:.1f} |")

        # UPOS kırılımı
        if results["upos_counts"]:
            lines.append("\n### UPOS Bazlı Kırılım")
            lines.append("| UPOS | Gold | Hit | Oran |")
            lines.append("|------|------|-----|------|")
            sorted_upos = sorted(
                results["upos_counts"].items(), key=lambda x: x[1], reverse=True
            )
            for upos, count in sorted_upos[:10]:
                hits = results["upos_hits"].get(upos, 0)
                pct = hits / count * 100 if count else 0
                lines.append(f"| {upos} | {count} | {hits} | %{pct:.1f} |")

        return "\n".join(lines)
    except Exception as e:
        return f"Benchmark hatası: {e}"


@mcp.tool()
def hata_analizi(max_cumle: int = 30, max_hata: int = 10) -> str:
    """Benchmark'teki hataları analiz eder: hangi sözcüklerde, hangi deprel'lerde
    hata yapılıyor? Geliştirme öncelikleri için kullanılır."""
    try:
        conllu_path = _PROJECT_ROOT / "benchmark" / "test.conllu"
        if not conllu_path.exists():
            return "benchmark/test.conllu bulunamadı."

        from benchmark.eval_dep import evaluate

        sa = _get_sentence_analyzer()
        dp = _get_dep_parser()

        results = evaluate(
            conllu_path=conllu_path,
            sa=sa,
            dp=dp,
            max_sentences=min(max_cumle, 979),
            collect_errors=max_hata,
        )

        lines = ["## Hata Analizi\n"]

        # Confusion matrisi: en sık karıştırılan deprel çiftleri
        if results["deprel_confusion"]:
            lines.append("### Deprel Karıştırma (Gold → Pred, Top 15)")
            lines.append("| Gold | Pred | Sayı |")
            lines.append("|------|------|------|")
            pairs = []
            for gold_dep, pred_counts in results["deprel_confusion"].items():
                for pred_dep, cnt in pred_counts.items():
                    if gold_dep != pred_dep:
                        pairs.append((gold_dep, pred_dep, cnt))
            pairs.sort(key=lambda x: x[2], reverse=True)
            for gold, pred, cnt in pairs[:15]:
                lines.append(f"| {gold} | {pred} | {cnt} |")

        # En büyük kayıplar (gold çok ama hit az)
        if results["deprel_counts"]:
            lines.append("\n### En Büyük Kayıplar (Düşük Recall)")
            lines.append("| Deprel | Gold | Hit | Kayıp | Oran |")
            lines.append("|--------|------|-----|-------|------|")
            losses = []
            for deprel, count in results["deprel_counts"].items():
                hits = results["deprel_hits"].get(deprel, 0)
                loss = count - hits
                pct = hits / count * 100 if count else 0
                losses.append((deprel, count, hits, loss, pct))
            losses.sort(key=lambda x: x[3], reverse=True)
            for dep, cnt, hit, loss, pct in losses[:10]:
                lines.append(f"| {dep} | {cnt} | {hit} | {loss} | %{pct:.0f} |")

        # Toplanan hatalar
        if results.get("errors"):
            lines.append(f"\n### Örnek Hatalar ({len(results['errors'])} adet)")
            for err in results["errors"][:max_hata]:
                lines.append(f"\n**Token:** `{err.get('form', '?')}` (sent: {err.get('sent_id', '?')})")
                lines.append(f"  Gold: head={err.get('gold_head')}, deprel={err.get('gold_deprel')}, upos={err.get('gold_upos')}")
                lines.append(f"  Pred: head={err.get('pred_head')}, deprel={err.get('pred_deprel')}, upos={err.get('pred_upos')}")

        return "\n".join(lines)
    except Exception as e:
        return f"Hata analizi hatası: {e}"


@mcp.tool()
def coklu_cumle_analiz(metin: str) -> str:
    """Birden fazla cümle içeren bir metni analiz eder.
    Her cümle için ayrı dependency ağacı oluşturur.
    Nokta, soru/ünlem işaretlerinden cümle ayrımı yapar."""
    try:
        import re
        sa = _get_sentence_analyzer()
        dp = _get_dep_parser()

        # Basit cümle bölme
        cumleler = [c.strip() for c in re.split(r'(?<=[.!?])\s+', metin) if c.strip()]
        if not cumleler:
            cumleler = [metin]

        lines = [f"## Metin Analizi ({len(cumleler)} cümle)\n"]

        for i, cumle in enumerate(cumleler, 1):
            tokens = sa.analyze(cumle)
            dep_tokens = dp.parse(tokens, text=cumle)

            lines.append(f"### Cümle {i}: {cumle}")
            lines.append("```")
            lines.append(dp.to_tree(dep_tokens))
            lines.append("```")

            # Özet tablo
            for t in dep_tokens:
                lines.append(f"  {t.id}. {t.form} → {t.lemma} ({t.upos}) --{t.deprel}--> {t.head}")
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"Metin analizi hatası: {e}"

@mcp.resource("turetim://skill/full")
def skill_full() -> str:
    """skill_turetim.md dosyasının tam içeriği."""
    skill_path = _PROJECT_ROOT / "skill_turetim.md"
    try:
        with open(skill_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "skill_turetim.md bulunamadı."


@mcp.resource("turetim://ekler/json")
def ekler_json() -> str:
    """Tüm 191 ekin JSON formatı."""
    return json.dumps(EKLER, ensure_ascii=False, indent=2)


# ============================================================
# PROMPTS
# ============================================================

@mcp.prompt()
def turetim_analiz(sozcuk: str) -> str:
    """Bir Türkçe sözcüğün türetimsel çözümlemesini yapmak için prompt."""
    return f"""Aşağıdaki Türkçe sözcüğün türetimsel çözümlemesini yap:

Sözcük: {sozcuk}

Uzun et al. (1992) kitabındaki 191 türetim ekini referans alarak:
1. Olası kök(ler)i belirle
2. Türetim ek(ler)ini tanımla (kitap kodu ile, ör. L112 -(l)Ik)
3. Türetim türünü belirt (A→A, A→E, E→A, E→E)
4. UPOS çıkarımı yap
5. Olası deprel'leri öner

Morfolojik kurallar:
- İÇİ (İçerikle Çözümleme İlişkisi): Taban sözlükte bulunmalı
- BÜU ve KÜU uyumu kontrol edilmeli
- Koruyucu ünsüz (y, n, s, ş) durumu değerlendirilmeli
"""


# ============================================================
# ANA GİRİŞ
# ============================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")
