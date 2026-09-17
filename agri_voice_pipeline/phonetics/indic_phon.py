"""
indic_phon.py — Indic normalisation + ASR-oriented phonetic encoding.

A rule-based grapheme-to-phoneme (G2P) encoder for Indic script, instantiated here
for Devanagari (Hindi). The phoneme inventory is deliberately script-agnostic — the
same uppercase-consonant / lowercase-vowel representation is what Telugu and Odia map
into — so one phonetic distance serves every language once its grapheme table is added.
Unlike Soundex (English/ASCII only), consonants collapse across the confusions Indic
ASR actually makes, word-final and rule-governed medial schwa is deleted, and
substitution costs come from phonological feature distance (see `pdist`).

Three levels of representation:
  norm_word(w)  -> orthographic canonical form (for dedup / merging rows)
  phon(w)       -> phoneme string, ASR-confusion-aware (for clustering / matching)
  skeleton(w)   -> consonant-only blocking key

Prior art for edit-distance-with-phonetic-cost: Kondrak (NAACL 2000);
Zobel & Dart (SIGIR 1996).
"""
import re
import unicodedata as ud

ZW = "\u200c\u200d\ufeff"

# ---------- nukta folding (क़->क etc.) ----------
NUKTA_MAP = {
    "\u0958": "\u0915",  # क़ -> क
    "\u0959": "\u0916",  # ख़ -> ख
    "\u095A": "\u0917",  # ग़ -> ग
    "\u095B": "\u091C",  # ज़ -> ज
    "\u095C": "\u0921",  # ड़ -> ड
    "\u095D": "\u0922",  # ढ़ -> ढ
    "\u095E": "\u092B",  # फ़ -> फ
    "\u095F": "\u092F",  # य़ -> य
}
NUKTA = "\u093C"

def strip_nukta(s: str) -> str:
    s = ud.normalize("NFD", s)
    s = s.replace(NUKTA, "")
    return ud.normalize("NFC", s)


def norm_word(w: str) -> str:
    """Canonical orthographic form used as the dedup key."""
    if not isinstance(w, str):
        return ""
    s = ud.normalize("NFC", w)
    for ch in ZW:
        s = s.replace(ch, "")
    s = strip_nukta(s)
    for k, v in NUKTA_MAP.items():
        s = s.replace(k, v)
    s = s.replace("\u0901", "\u0902")          # chandrabindu -> anusvara
    s = s.replace("\u0945", "\u0947")          # candra e -> e
    s = s.replace("\u0949", "\u094B")          # candra o -> o
    s = s.replace("\u0972", "\u0905")          # candra A -> a
    s = re.sub(r"[\u0964\u0965]", " ", s)      # danda
    s = re.sub(r"[!?,;:\"'`~*_]", " ", s)      # stray punctuation, keep hyphen
    s = re.sub(r"\s*-\s*", "-", s)             # tighten reduplication hyphens
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------- phonetic encoding ----------
# Consonants collapse across the confusions Hindi ASR actually makes:
#   aspirated ~ unaspirated,  retroflex ~ dental,  all sibilants,  v ~ b
CONS = {
    "क": "K", "ख": "K", "ग": "K", "घ": "K", "ङ": "N", "क़": "K", "ख़": "K", "ग़": "K",
    "च": "C", "छ": "C", "ज": "J", "झ": "J", "ञ": "N", "ज़": "J",
    "ट": "T", "ठ": "T", "ड": "D", "ढ": "D", "ण": "N",
    "त": "T", "थ": "T", "द": "D", "ध": "D", "न": "N",
    "प": "P", "फ": "P", "ब": "B", "भ": "B", "म": "M", "फ़": "P",
    "य": "Y", "र": "R", "ल": "L", "व": "B", "ळ": "L",
    "श": "S", "ष": "S", "स": "S", "ह": "H",
    "ड़": "R", "ढ़": "R", "ऱ": "R",
}

# Independent vowels
IVOW = {
    "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u",
    "ए": "e", "ऐ": "e", "ओ": "o", "औ": "o", "ऋ": "ri", "ॠ": "ri",
    "ऍ": "e", "ऑ": "o", "ऒ": "o", "ऎ": "e",
}
# Dependent matras
MVOW = {
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u",
    "े": "e", "ै": "e", "ो": "o", "ौ": "o", "ृ": "ri",
    "ॅ": "e", "ॉ": "o", "ॊ": "o", "ॆ": "e",
}
VIRAMA = "\u094D"
ANUSVARA = "\u0902"
CHANDRABINDU = "\u0901"
VISARGA = "\u0903"


def phon(w: str, drop_final_nasal: bool = True) -> str:
    """Phoneme string. Consonants uppercase, vowels lowercase.

    Inherent 'a' is inserted after a consonant that has no matra and no virama,
    except word-finally (Hindi schwa deletion)."""
    s = norm_word(w)
    s = s.replace(" ", "").replace("-", "")
    out = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch in CONS:
            out.append(CONS[ch])
            j = i + 1
            # look at what follows
            if j < n and s[j] == VIRAMA:
                i = j + 1
                continue
            if j < n and s[j] in MVOW:
                out.append(MVOW[s[j]])
                i = j + 1
            else:
                if j < n:                      # non-final -> inherent schwa
                    out.append("a")
                i = j
            continue
        if ch in IVOW:
            out.append(IVOW[ch])
            i += 1
            continue
        if ch in (ANUSVARA, CHANDRABINDU):
            out.append("n")   # nasalisation, distinct from consonant न
            i += 1
            continue
        if ch == VISARGA:
            out.append("H")
            i += 1
            continue
        if ch.isdigit():
            out.append(ch)
            i += 1
            continue
        if ch.isascii() and ch.isalpha():
            out.append(ch.upper())
            i += 1
            continue
        i += 1

    p = "".join(out)
    p = re.sub(r"(.)\1+", r"\1", p)             # degeminate: मक्का -> MaKa
    p = re.sub(r"H$", "", p)                    # weak final h
    if drop_final_nasal:
        p = re.sub(r"n$", "", p)                # गेहूं ~ गेहू
    return p


def skeleton(w: str) -> str:
    """Consonant-only key — used for blocking, not for final judgement."""
    return re.sub(r"[^A-Z]", "", phon(w))


# ---------- weighted phonetic edit distance ----------
# cheap substitutions: these confuse constantly in noisy Hindi ASR
NEAR = [
    {"K", "H"}, {"T", "D"}, {"P", "B"}, {"B", "M"}, {"N", "M"},
    {"J", "C"}, {"J", "Y"}, {"R", "L"}, {"S", "C"}, {"S", "H"},
    {"a", "o"}, {"i", "e"}, {"u", "o"}, {"a", "e"}, {"e", "i"},
    {"n", "N"}, {"n", "M"},
]
_NEARSET = set()
for pair in NEAR:
    a, b = tuple(pair)
    _NEARSET.add((a, b))
    _NEARSET.add((b, a))

_VOW = set("aeiou")


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if (a, b) in _NEARSET:
        return 0.4
    if a in _VOW and b in _VOW:
        return 0.6
    if (a in _VOW) != (b in _VOW):
        return 1.2
    return 1.0


def _indel_cost(c: str) -> float:
    return 0.5 if c in _VOW or c == "n" else 1.0


def pdist(a: str, b: str) -> float:
    """Weighted Levenshtein over phoneme strings."""
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    if la == 0:
        return sum(_indel_cost(c) for c in b)
    if lb == 0:
        return sum(_indel_cost(c) for c in a)
    prev = [0.0] * (lb + 1)
    for j in range(1, lb + 1):
        prev[j] = prev[j - 1] + _indel_cost(b[j - 1])
    for i in range(1, la + 1):
        cur = [prev[0] + _indel_cost(a[i - 1])] + [0.0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            bj = b[j - 1]
            cur[j] = min(
                prev[j] + _indel_cost(ai),
                cur[j - 1] + _indel_cost(bj),
                prev[j - 1] + _sub_cost(ai, bj),
            )
        prev = cur
    return prev[lb]


def psim(a: str, b: str) -> float:
    """Normalised similarity in [0,1]."""
    if not a and not b:
        return 1.0
    d = pdist(a, b)
    return max(0.0, 1.0 - d / max(len(a), len(b), 1))
