#!/usr/bin/env node
/**
 * TDK'nın (Türk Dil Kurumu) kamuya açık "Atasözleri ve Deyimler Sözlüğü"
 * uç noktasından tüm atasözü + deyim kayıtlarını çekip
 * `kind,text,meaning,topic,min_grade,source` CSV'sine dönüştürür.
 *
 * NOT (2026-09-06): sozluk.gov.tr çok yakın zamanda (bkz. sunucu
 * Last-Modified: 2026-08-31) yeniden tasarlandı — eski `sozluk.gov.tr/atasozu?ara=`
 * artık JSON değil SPA kabuğu döndürüyor. Yeni uç nokta ayrı bir alt alan
 * adında: `api.sozluk.gov.tr/ads?ara=&searchType=web`, `Origin`/`Referer:
 * sozluk.gov.tr` header'ları zorunlu (yoksa `{"error":"FORBIDDEN"}`).
 * Şema aynı kaldı (`soz_id/sozum/anlami/turu2`) — bu keşif
 * github.com/clydeofficial/tdk-sozluk (npm `tdk-sozluk`, 2026-09-02
 * güncel) kütüphanesinin kaynağından doğrulandı.
 *
 * Uç nokta bir ALT-DİZGE (substring, `LIKE '%q%'` tarzı — önek değil: "eli"
 * sorgusu "d-eli-arlanmaz" gibi ortasında geçenleri de buluyor) araması ve
 * **sayfa başına 50 sonuçla sınırlı** (offset/limit parametresi yok — denendi,
 * hiçbiri etkilemedi). Tek harfli sorguların neredeyse tamamı bu sınıra
 * takılıyor. Bu yüzden **uyarlanabilir dizge genişletme** kullanılır: bir
 * sorgu tam 50 sonuç döndürürse (kırpılmış olabilir) sorgu bir karakter
 * uzatılıp alt-sorgulara bölünür (MAX_DEPTH'e kadar); 50'nin altında dönerse
 * o dal tamamlanmış sayılır. Genişletme karakter kümesi harflerin yanı sıra
 * BOŞLUK da içerir — kelime sınırını atlamak gerekiyor: "eli" tıkanmışken
 * "elia" (bitişik) 0 sonuç veriyor ama "eli a" (boşluklu) "eli açık" dahil
 * 16 sonuç veriyor. Sonuçlar `soz_id`'ye göre tekilleştirilir (bir dal
 * başka bir dalın üst/alt kümesi olabilir, alt-dizge eşleşmesi çapraz
 * dal çakışmasına yol açar — zararsız, yalnız tekrar isteği). Garantili
 * %100 kapsama DEĞİLDİR (MAX_DEPTH/MAX_REQUESTS sınırının ötesi kalır) ama
 * pratikte tam derinlik nadiren gerekiyor. Uzun sürebileceğinden
 * CHECKPOINT_EVERY istek sayısında ara-CSV yazılır (kesilirse veri kaybolmaz).
 *
 * DizgeBERT-Idiom bağlamı: PARSEME-TR yalnız fiil-merkezli deyimleri (VID)
 * kapsıyor; TDK sözlüğü fiil/isim/sıfat ayrımı yapmadan hepsini içerir
 * ("eli açık" gibi isim/sıfat deyimlerini de) — bu, DizgeBERT-Idiom'un
 * kapsam dışı bıraktığı kategoriyi kapatacak ham lexicon.
 *
 * Köken: corpus_engine reposunun kaldırılmış "ÖSP" alt sisteminden
 * (apps/osp-api/scripts/fetch-tdk-atasozu-deyim.mjs, commit 1f2b0f72)
 * bağımsız script olarak taşındı — orijinali kendi Postgres şemasına
 * (osp.mwe_entries) içe aktarmak için osp-web/import sayfasını
 * hedefliyordu; burada yalnızca CSV üretimi kullanılıyor.
 *
 * Kullanım:
 *   node fetch_tdk_deyim.mjs [--out=idiom_data/raw/tdk_atasozu_deyim.csv]
 *
 * Çıktı dosyası .gitignore'da (idiom_data/*) — üretilen veri commit'lenmez.
 */

const ALPHABET = [
    'a', 'b', 'c', 'ç', 'd', 'e', 'f', 'g', 'ğ', 'h', 'ı', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'ö', 'p', 'r', 's', 'ş', 't', 'u', 'ü', 'v', 'y', 'z',
];
// Genişletme sırasında harflere ek olarak BOŞLUK da denenir — API alt-dizge (substring)
// araması yapıyor ve kelime sınırını atlamıyor: "eli" (50'ye tıkanmış) → "elia" 0 sonuç
// (bitişik eşleşme yok) ama "eli a" (boşluklu) 16 sonuç ("eli açık" dahil). Bir sonraki
// kelimeye geçmek için boşluk şart.
const EXPAND_CHARS = [...ALPHABET, ' '];

const KIND_MAP = {
    'Atasözü': 'proverb',
    'Deyim': 'idiom',
};

const HTML_ENTITY_MAP = { amp: '&', quot: '"', apos: "'", lt: '<', gt: '>', nbsp: ' ' };

function stripHtml(s) {
    if (!s) return '';
    return s
        .replace(/<[^>]+>/g, ' ')
        .replace(/&(amp|quot|apos|lt|gt|nbsp);/g, (_, name) => HTML_ENTITY_MAP[name])
        .replace(/\s+/g, ' ')
        .trim();
}

function csvField(value) {
    const s = String(value ?? '');
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

const PAGE_CAP = 50;       // /ads sayfa başına sonuç sınırı (deneysel — sabit)
const MAX_DEPTH = 12;      // önek uzunluğu üst sınırı (boşluk dahil — birkaç kelimeyi hecelemeye yeter)
const MAX_REQUESTS = 60000; // güvenlik sınırı — TDK sunucusuna nazik davranmak için

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// TDK: RateLimit-Policy: 3000;w=900 (15 dakikada 3000 istek) + 429'da Retry-After header'ı.
async function fetchQuery(query, retryLeft = 5) {
    const url = `https://api.sozluk.gov.tr/ads?ara=${encodeURIComponent(query)}&searchType=web`;
    let res;
    try {
        res = await fetch(url, {
            headers: {
                'Accept': 'application/json',
                'Origin': 'https://sozluk.gov.tr',
                'Referer': 'https://sozluk.gov.tr/',
                'User-Agent': 'Mozilla/5.0 (DizgeBERT-Idiom data import; educational use)',
            },
        });
    } catch (err) {
        // ağ hatası (DNS/bağlantı kopması) — HTTP durumu yok, aynı yeniden-deneme yolunu kullan
        if (retryLeft <= 0) throw new Error(`Ağ hatası "${query}": ${err.message} (tüm denemeler tükendi)`);
        console.warn(`[tdk-import] Ağ hatası "${query}": ${err.message}, yeniden deneniyor (${retryLeft} hak kaldı)...`);
        await sleep(3000);
        return fetchQuery(query, retryLeft - 1);
    }
    if (res.status === 429) {
        const retryAfter = Number(res.headers.get('retry-after')) || 60;
        console.warn(`[tdk-import] 429 (rate limit) "${query}" — ${retryAfter + 5}s bekleniyor...`);
        await sleep((retryAfter + 5) * 1000);
        return fetchQuery(query, retryLeft); // rate-limit denemesi sayaca dahil değil
    }
    if (!res.ok) {
        if (retryLeft <= 0) throw new Error(`HTTP ${res.status} for "${query}" (tüm denemeler tükendi)`);
        console.warn(`[tdk-import] HTTP ${res.status} for "${query}", yeniden deneniyor (${retryLeft} hak kaldı)...`);
        await sleep(3000);
        return fetchQuery(query, retryLeft - 1);
    }
    const data = await res.json().catch(() => null);
    return Array.isArray(data) ? data : [];
}

function writeCsv(seen, outPath) {
    const rows = [];
    let skippedUnknownKind = 0;
    for (const item of seen.values()) {
        const kind = KIND_MAP[item.turu2];
        if (!kind) { skippedUnknownKind++; continue; }
        rows.push({
            kind,
            text: item.sozum,
            meaning: stripHtml(item.anlami),
            topic: '',
            min_grade: '',
            source: 'TDK Atasözleri ve Deyimler Sözlüğü',
        });
    }
    const header = 'kind,text,meaning,topic,min_grade,source';
    const lines = rows.map((r) =>
        [r.kind, r.text, r.meaning, r.topic, r.min_grade, r.source].map(csvField).join(',')
    );
    fsMod.mkdirSync(pathMod.dirname(outPath), { recursive: true });
    fsMod.writeFileSync(outPath, [header, ...lines].join('\n') + '\n', 'utf8');
    return { rows, skippedUnknownKind };
}

let fsMod, pathMod;

function stateForOut(outPath) {
    return pathMod.join(pathMod.dirname(outPath), '_tdk_crawl_state.json');
}

function loadState(outPath) {
    const p = stateForOut(outPath);
    if (!fsMod.existsSync(p)) return null;
    const d = JSON.parse(fsMod.readFileSync(p, 'utf8'));
    return { queue: d.queue, seen: new Map(d.seenEntries.map((e) => [e.soz_id, e])) };
}

function saveState(outPath, queue, seen) {
    const p = stateForOut(outPath);
    fsMod.mkdirSync(pathMod.dirname(p), { recursive: true });
    fsMod.writeFileSync(p, JSON.stringify({ queue, seenEntries: [...seen.values()] }), 'utf8');
}

// Her çalıştırma en fazla bu kadar istek atar, sonra durumu kaydedip çıkar — bu ortamda
// uzun arka-plan işleri bir noktada dıştan kesilebiliyor (görüldü: ~1.5-2 saat civarı).
// Kesintiye uğrarsa da (öldürülme/hata) ara-kayıt CHECKPOINT_EVERY'de zaten yazılmış olur;
// bir sonraki `node fetch_tdk_deyim.mjs` çalıştırması otomatik kaldığı yerden devam eder.
const REQUESTS_PER_RUN = 8000;
const CHECKPOINT_EVERY = 200;

async function crawl(outPath) {
    const prior = loadState(outPath);
    const seen = prior ? prior.seen : new Map();
    const queue = prior ? prior.queue : [...ALPHABET];
    if (prior) console.log(`[tdk-import] devam ediliyor: ${seen.size} kayıt, kuyrukta ${queue.length} önek`);

    let nReq = 0;
    while (queue.length) {
        if (nReq >= REQUESTS_PER_RUN || nReq >= MAX_REQUESTS) {
            console.warn(`[tdk-import] bu çalıştırmanın istek sınırına ulaşıldı (${nReq}), durum kaydedilip çıkılıyor. `
                + `Devam etmek için tekrar çalıştır: node fetch_tdk_deyim.mjs`);
            saveState(outPath, queue, seen);
            return { seen, done: false };
        }
        const query = queue.shift();
        const items = await fetchQuery(query);
        nReq++;
        for (const item of items) seen.set(item.soz_id, item);
        const truncated = items.length >= PAGE_CAP;
        console.log(`[tdk-import] "${query}" (derinlik ${query.length}) → ${items.length} sonuç` +
            `${truncated ? ' [KIRPILMIŞ]' : ''}, toplam tekil: ${seen.size}, kalan kuyruk: ${queue.length}, istek: ${nReq}`);
        if (truncated && query.length < MAX_DEPTH) {
            // çift boşluğa düşme ("eli  a" gibi) — anlamsız, sonuç vermez
            const children = query.endsWith(' ')
                ? EXPAND_CHARS.filter((c) => c !== ' ')
                : EXPAND_CHARS;
            for (const c of children) queue.push(query + c);
        }
        if (nReq % CHECKPOINT_EVERY === 0) {
            saveState(outPath, queue, seen);
            const { rows } = writeCsv(seen, outPath);
            console.log(`[tdk-import] ara-kayıt: ${outPath} (${rows.length} satır, ${nReq} istek)`);
        }
        // TDK sınırı 15 dk'da 3000 istek (RateLimit-Policy: 3000;w=900) — 350ms ~2571/15dk,
        // güvenli marj bırakır.
        await sleep(350);
    }
    return { seen, done: true };
}

async function main() {
    fsMod = await import('fs');
    pathMod = await import('path');
    const outArg = process.argv.find((a) => a.startsWith('--out='));
    const outPath = outArg ? outArg.slice('--out='.length) : 'idiom_data/raw/tdk_atasozu_deyim.csv';

    const { seen, done } = await crawl(outPath);
    const { rows, skippedUnknownKind } = writeCsv(seen, outPath);

    console.log(`[tdk-import] Yazıldı: ${outPath} (${rows.length} satır, ${skippedUnknownKind} bilinmeyen tür atlandı)`);
    const byKind = rows.reduce((acc, r) => ((acc[r.kind] = (acc[r.kind] ?? 0) + 1), acc), {});
    console.log('[tdk-import] Tür dağılımı:', byKind);
    if (done) {
        const p = stateForOut(outPath);
        if (fsMod.existsSync(p)) fsMod.unlinkSync(p);
        console.log('[tdk-import] Tarama tamamlandı (kuyruk boş).');
    }
}

main().catch((err) => {
    console.error('[tdk-import] Hata:', err.message);
    process.exit(1);
});
