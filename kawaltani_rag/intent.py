"""
Intent Classifier — Menentukan jalur Hybrid Retrieval (Gambar 3 Proposal).

Perbaikan v2 atas versi Claude AI:
  1. Mendukung rentang 2 tanggal Bahasa Indonesia ('21 Januari 2025 s/d 25 Januari 2025')
     termasuk rentang tanpa tahun pada tanggal pertama ('13 Januari - 20 Januari 2026').
  2. Mendukung kueri perbandingan 2 tanggal eksplisit ('antara tanggal 20 Januari 2026
     dan 20 April 2026') menjadi 2 jendela waktu terpisah (`compare_windows`).
  3. Mendukung acuan Hari Setelah Tanam / HST ('H60 s/d H69 dan selama H70 s/d H76', '30 HST').
  4. Mengenali istilah umum ('nutrisi', 'unsur hara', 'kesuburan', 'kondisi lahan/sawah',
     'jadwal pemupukan') sehingga kueri seperti 'Bagaimana kondisi nutrisi tanah di area 1?'
     tetap mengambil data sensor SQL area 1, bukan salah belok ke Vector murni.
"""
import re
import datetime as dt
from dataclasses import dataclass, field

SQL, VECTOR, HYBRID = "sql", "vector", "hybrid"

PARAM_SYNONYMS = {
    "nitro": ["nitrogen", " n ", "urea", "za ", "unsur n"],
    "phos": ["fosfor", "fosfat", "phosphor", "unsur p", "sp-36", "sp36", "tsp"],
    "pot": ["kalium", "potasium", "potassium", "unsur k", "kcl"],
    "ph": ["ph tanah", "ph", "keasaman", "kemasaman", "asam", "basa", "alkalin", "dolomit"],
    "temp": ["suhu", "temperatur", "panas", "dingin"],
    "hum": ["kelembapan", "kelembaban", "lembab", "lengas", "kadar air"],
    "con": ["konduktivitas", "ec", "daya hantar", "dhl"],
    "salin": ["salinitas", "garam", "asin", "salin"],
    "tds": ["tds", "zat terlarut", "padatan terlarut"],
}

GENERAL_SOIL_KEYWORDS = [
    "nutrisi", "unsur hara", "hara", "kesuburan", "npk",
    "kondisi tanah", "kondisi sawah", "kondisi lahan", "data sensor", "sensor terakhir",
]

ADVISORY_KEYWORDS = [
    "apa yang harus", "tindakan", "rekomendasi", "saran", "sebaiknya",
    "perlu", "memerlukan", "haruskah", "bagaimana cara", "mengatasi", "solusi",
    "dampak", "dampaknya", "pengaruh", "akibat", "kenapa", "mengapa", "penyebab",
    "normal?", "aman", "berbahaya", "gejala", "pupuk", "pemupukan", "dosis", "jadwal",
]

COMPARE_KEYWORDS = [
    "bandingkan", "perbandingan", "dibanding", "dibandingkan", "versus", " vs ",
    "antara tanggal", "selisih", "berubah", "perubahan",
]

REL_DAYS = {
    "hari ini": 0, "sekarang": 0, "saat ini": 0, "terkini": 0, "terbaru": 0, "terakhir": 0,
    "kemarin": 1, "kemaren": 1,
}

BULAN = {
    "januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6,
    "juli": 7, "agustus": 8, "september": 9, "oktober": 10, "november": 11, "desember": 12,
}
BULAN_RE = "|".join(BULAN)

ANGKA_KATA = {
    "satu": 1, "se": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5, "enam": 6,
    "tujuh": 7, "delapan": 8, "sembilan": 9, "sepuluh": 10, "sebelas": 11,
    "duabelas": 12, "dua belas": 12, "empatbelas": 14, "empat belas": 14,
    "limabelas": 15, "lima belas": 15, "duapuluh": 20, "dua puluh": 20,
    "tigapuluh": 30, "tiga puluh": 30,
}


def _normalize_number_words(q: str) -> str:
    for kata, angka in sorted(ANGKA_KATA.items(), key=lambda x: -len(x[0])):
        q = re.sub(rf"\b{kata}\b(?=\s*(hari|minggu|bulan))", str(angka), q)
    return q


@dataclass
class Intent:
    route: str
    params: list = field(default_factory=list)
    areas: list = field(default_factory=list)
    devices: list = field(default_factory=list)
    start: dt.datetime = None
    end: dt.datetime = None
    is_comparison: bool = False
    needs_advice: bool = False
    has_general_soil: bool = False
    raw_query: str = ""
    compare_windows: list = field(default_factory=list)   # [(start1, end1), (start2, end2)]
    hst_windows: list = field(default_factory=list)       # [(h_start, h_end), ...]

    @property
    def has_spatiotemporal(self) -> bool:
        has_target = bool(self.params) or self.has_general_soil
        has_loc_or_time = (
            bool(self.areas) or bool(self.devices) or self.start is not None
        )
        return has_target and has_loc_or_time


def _find_params(q: str):
    found = []
    padded = f" {q} "
    for key, syns in PARAM_SYNONYMS.items():
        for s in syns:
            if s in padded:
                found.append(key)
                break
    if "ph" in found and not re.search(r"\bph\b|keasaman|kemasaman|asam|basa|alkalin|dolomit", q):
        found.remove("ph")
    if "npk" in q:
        for p in ("nitro", "phos", "pot"):
            if p not in found:
                found.append(p)
    return found


def _find_areas(q: str):
    areas = sorted({int(n) for n in re.findall(r"area\s*(\d+)", q) if 1 <= int(n) <= 6})
    if not areas:
        areas = sorted({int(n) for n in re.findall(r"(?:lahan|petak|blok)\s*(\d+)", q)
                        if 1 <= int(n) <= 6})
    return areas


def _find_devices(q: str):
    return sorted({d.replace(" ", "").upper() for d in re.findall(r"(telu\s*0?\d{3})", q)})


def _day_bounds(d: dt.datetime):
    return (
        d.replace(hour=0, minute=0, second=0, microsecond=0),
        d.replace(hour=23, minute=59, second=59, microsecond=0),
    )


def _extract_hst_windows(q: str):
    """Deteksi pola 'H60 s/d H69' atau 'HST 30 - 40'."""
    ranges = re.findall(r"\bh(?:st)?\s*(\d{1,3})\s*(?:s/d|sd|sampai|-|hingga)\s*h(?:st)?\s*(\d{1,3})\b", q)
    if ranges:
        return [(int(a), int(b)) for a, b in ranges]
    single = re.findall(r"\b(\d{1,3})\s*hst\b|\bhst\s*(\d{1,3})\b", q)
    if single:
        vals = [int(a or b) for a, b in single]
        return [(v, v) for v in vals]
    return []


def _extract_id_dates(q: str):
    """
    Ekstrak semua tanggal Bahasa Indonesia dalam kueri, termasuk pola
    '13 Januari - 20 Januari 2026' di mana tanggal pertama tidak menulis tahun.
    """
    # Pola khusus: DD Bulan (- / s/d / sampai) DD Bulan YYYY
    partial_range = re.search(
        rf"(\d{{1,2}})\s+({BULAN_RE})\s*(?:-|–|s/d|sd|sampai|hingga)\s*(\d{{1,2}})\s+({BULAN_RE})\s+(\d{{4}})",
        q,
    )
    if partial_range:
        d1, m1, d2, m2, yr = partial_range.groups()
        yr_i = int(yr)
        dt1 = dt.datetime(yr_i, BULAN[m1], int(d1))
        dt2 = dt.datetime(yr_i, BULAN[m2], int(d2))
        if dt1 > dt2:
            dt1 = dt.datetime(yr_i - 1, BULAN[m1], int(d1))
        return [dt1, dt2], True  # True = eksplisit rentang kontigu

    # Pola umum: satu atau lebih 'DD Bulan YYYY'
    full_matches = list(re.finditer(rf"(\d{{1,2}})\s+({BULAN_RE})\s+(\d{{4}})", q))
    if len(full_matches) >= 2:
        dts = [
            dt.datetime(int(m.group(3)), BULAN[m.group(2)], int(m.group(1)))
            for m in full_matches
        ]
        between = q[full_matches[0].end():full_matches[1].start()]
        is_range = bool(re.search(r"s/d|sd|sampai|hingga|-|–", between))
        return dts, is_range
    if len(full_matches) == 1:
        m = full_matches[0]
        return [dt.datetime(int(m.group(3)), BULAN[m.group(2)], int(m.group(1)))], False

    return [], False


def _parse_compare_windows(q: str, now: dt.datetime):
    today_end = now.replace(hour=23, minute=59, second=59, microsecond=0)
    for unit, days in (("minggu", 7), ("bulan", 30)):
        if re.search(rf"{unit} ini", q) and re.search(rf"{unit} (lalu|kemarin|sebelumnya)", q):
            recent = (now - dt.timedelta(days=days), today_end)
            prior = (now - dt.timedelta(days=days * 2), now - dt.timedelta(days=days))
            return [prior, recent]

    # Dua tanggal ISO terpisah untuk dibandingkan ("2025-01-20 vs 2025-04-20")
    iso = re.findall(r"(\d{4})-(\d{2})-(\d{2})", q)
    if len(iso) >= 2 and any(k in q for k in COMPARE_KEYWORDS):
        d1 = dt.datetime(*map(int, iso[0]))
        d2 = dt.datetime(*map(int, iso[1]))
        return [_day_bounds(d1), _day_bounds(d2)]

    # Dua tanggal Bahasa Indonesia terpisah ("antara tanggal 20 Januari 2026 dan 20 April 2026")
    dts, is_range = _extract_id_dates(q)
    if len(dts) >= 2 and not is_range:
        return [_day_bounds(dts[0]), _day_bounds(dts[1])]

    return None


def _parse_time(q: str, now: dt.datetime = None):
    now = now or dt.datetime.now()
    q_norm = _normalize_number_words(q)
    today_end = now.replace(hour=23, minute=59, second=59, microsecond=0)

    # 1. Tanggal ISO ("2026-03-15")
    iso = re.findall(r"(\d{4})-(\d{2})-(\d{2})", q_norm)
    if len(iso) >= 2:
        a = dt.datetime(*map(int, iso[0]))
        b = dt.datetime(*map(int, iso[1])).replace(hour=23, minute=59, second=59)
        return (a, b) if a <= b else (b, a)
    if len(iso) == 1:
        return _day_bounds(dt.datetime(*map(int, iso[0])))

    # 2. Tanggal Bahasa Indonesia (tunggal maupun rentang)
    dts, _ = _extract_id_dates(q_norm)
    if len(dts) >= 2:
        a, b = min(dts[0], dts[1]), max(dts[0], dts[1])
        return _day_bounds(a)[0], _day_bounds(b)[1]
    if len(dts) == 1:
        return _day_bounds(dts[0])

    # 3. "N hari/minggu/bulan terakhir/belakangan/ini"
    m = re.search(r"(\d+)\s*(hari|minggu|bulan)\s*(terakhir|belakangan|ini|lalu)", q_norm)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        days = n * {"hari": 1, "minggu": 7, "bulan": 30}[unit]
        return now - dt.timedelta(days=days), today_end

    # 4. Frasa relatif ("minggu ini", "bulan lalu", "hari ini", "kemarin")
    if re.search(r"minggu (ini|lalu|kemarin)", q_norm):
        days = 7 if "ini" in q_norm else 14
        return now - dt.timedelta(days=days), today_end
    if re.search(r"bulan (ini|lalu|kemarin)", q_norm):
        days = 30 if "ini" in q_norm else 60
        return now - dt.timedelta(days=days), today_end

    for kw, days_ago in REL_DAYS.items():
        if kw in q_norm:
            d = now - dt.timedelta(days=days_ago)
            return _day_bounds(d)

    return None, None


def classify(query: str, now: dt.datetime = None) -> Intent:
    now_ref = now or dt.datetime.now()
    q = query.lower().strip()
    params = _find_params(q)
    has_general_soil = any(k in q for k in GENERAL_SOIL_KEYWORDS)
    areas = _find_areas(q)
    devices = _find_devices(q)
    hst_windows = _extract_hst_windows(q)
    needs_advice = any(k in q for k in ADVISORY_KEYWORDS)
    is_comparison = any(k in q for k in COMPARE_KEYWORDS) or len(hst_windows) >= 2

    compare_windows = []
    if is_comparison:
        cw = _parse_compare_windows(_normalize_number_words(q), now_ref)
        if cw:
            compare_windows = cw

    if compare_windows:
        start, end = compare_windows[0][0], compare_windows[-1][1]
    else:
        start, end = _parse_time(q, now_ref)

    intent = Intent(
        route=VECTOR,
        params=params,
        areas=areas,
        devices=devices,
        start=start,
        end=end,
        is_comparison=is_comparison,
        needs_advice=needs_advice,
        has_general_soil=has_general_soil,
        raw_query=query,
        compare_windows=compare_windows,
        hst_windows=hst_windows,
    )

    if intent.has_spatiotemporal:
        intent.route = HYBRID if needs_advice else SQL
    elif (params or has_general_soil) and not needs_advice and (areas or devices):
        intent.route = SQL
    elif (params or has_general_soil) and not needs_advice and any(k in q for k in ("berapa", "cek ", "nilai ", "kadar ")):
        intent.route = SQL
    else:
        intent.route = VECTOR

    return intent

