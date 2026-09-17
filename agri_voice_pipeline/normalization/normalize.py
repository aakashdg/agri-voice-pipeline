"""Hindi / Telugu / Odia ASR text normalizer for word-error-rate comparison.

The Hindi normalizer (`normalize_hindi_text`) is a faithful port of IISc's reference
Hindi WER normalizer (calculateWER.py) — credit to IISc; this module carries only the
normalization logic, no secrets. Telugu and Odia (`normalize_telugu_text` /
`normalize_odia_text`) apply the same pipeline with per-language number→word tables.

Coverage:
  - HINDI: NFC, nukta-fold, native digits→Hindi words, strip bracketed content +
    punctuation, collapse whitespace.
  - TELUGU / ODIA: same pipeline, with COMPOSITIONAL number-word tables (tens + units,
    Indian numbering) cross-referenced with authoritative sources and AI4Bharat's
    indic-numtowords. They give a CONSISTENT canonicalization — so relative
    model-vs-model WER is valid — but the Te/Od number tables should be validated by a
    native speaker before publishing ABSOLUTE Te/Od WER.
  - `normalize(text, lang)` dispatches by language code or name (hi/te/or or the full
    names). Corpus-level (micro-average) helpers `corpus_wer` / `corpus_wer_by_lang`
    match the convention jiwer.wer(list_refs, list_hyps) over all clips.
"""
import re
import unicodedata


def number_to_hindi_words(n):
    """Convert an integer to Hindi words (Indian numbering system)."""
    units = ["", "एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "नौ"]
    teens = ["दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस"]
    tens = ["", "", "बीस", "तीस", "चालीस", "पचास", "साठ", "सत्तर", "अस्सी", "नब्बे"]

    def two_digit_words(num):
        if num < 10:
            return units[num]
        elif num < 20:
            return teens[num - 10]
        return tens[num // 10] + (" " + units[num % 10] if num % 10 != 0 else "")

    def convert(num):
        if num == 0:
            return "शून्य"
        parts = []
        for divisor, label in [(10**7, "करोड़"), (10**5, "लाख"), (1000, "हज़ार"), (100, "सौ")]:
            q = num // divisor
            if q:
                parts.append(convert(q) + f" {label}")
                num %= divisor
        if num:
            parts.append(two_digit_words(num))
        return " ".join(parts)

    return convert(int(n)).strip()


def normalize_hindi_text(text):
    """Paper's Hindi ASR normalizer: NFC, nukta-fold, digits→Hindi words,
    strip bracketed content + punctuation, collapse whitespace."""
    try:
        text = unicodedata.normalize('NFC', str(text))
        # Fold nukta: NFC decomposes precomposed forms (क़ etc.) to base + U+093C,
        # so stripping the combining nukta yields क़→क, ज़→ज, ड़→ड … (matches the
        # paper's intent; avoids the multi-codepoint str.maketrans pitfall).
        text = text.replace('़', '')
        text = text.translate(str.maketrans('०१२३४५६७८९', '0123456789'))
        text = re.sub(r'\(.*?\)|\[.*?\]|\{.*?\}|<.*?>|❴.*?❵', '', text)   # bracketed content
        text = re.sub(r'\d+', lambda m: number_to_hindi_words(int(m.group())), text)
        text = re.sub(r'[,:.\-!?;\"\'“”‘’%()\[\]{}<>₹^&*—\|\\/~`+=\n\t]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text if text else "NA"
    except Exception:
        return "NA"


# ============================================================ TELUGU / ODIA
# Same pipeline as normalize_hindi_text, with per-language number words.
# Number-word tables cross-referenced with Omniglot / Wikipedia (Odia numerals) /
# languagesandnumbers + AI4Bharat's indic-numtowords compositional convention.
_TE_DIGITS = str.maketrans("".join(chr(0x0C66 + i) for i in range(10)), "0123456789")
_OR_DIGITS = str.maketrans("".join(chr(0x0B66 + i) for i in range(10)), "0123456789")
_BRACKET_RE = re.compile(r'\(.*?\)|\[.*?\]|\{.*?\}|<.*?>|❴.*?❵')
# punctuation set = the Hindi port's set + danda (।) and double danda (॥)
_PUNCT_RE = re.compile(r'[।॥,:.\-!?;"\'“”‘’%()\[\]{}<>₹^&*—\|\\/~`+=\n\t]')


def _indian_num_to_words(n, units, teens, tens, scale, zero):
    """Integer → compositional words, Indian numbering (…crore, lakh, thousand, hundred),
    mirroring number_to_hindi_words. e.g. 25 → '<twenty> <five>' (tens + units)."""
    def two_digit(num):
        if num < 10: return units[num]
        if num < 20: return teens[num - 10]
        return tens[num // 10] + ((" " + units[num % 10]) if num % 10 else "")
    def convert(num):
        if num == 0: return zero
        parts = []
        for divisor, label in scale:
            q = num // divisor
            if q:
                parts.append(convert(q) + " " + label)
                num %= divisor
        if num:
            parts.append(two_digit(num))
        return " ".join(parts)
    return convert(int(n)).strip()


_TE_UNITS = ["", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది"]
_TE_TEENS = ["పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు", "పద్దెనిమిది", "పంతొమ్మిది"]
_TE_TENS  = ["", "", "ఇరవై", "ముప్పై", "నలభై", "యాభై", "అరవై", "డెబ్బై", "ఎనభై", "తొంభై"]
_TE_SCALE = [(10**7, "కోటి"), (10**5, "లక్ష"), (1000, "వెయ్యి"), (100, "వంద")]
def number_to_telugu_words(n):
    return _indian_num_to_words(n, _TE_UNITS, _TE_TEENS, _TE_TENS, _TE_SCALE, "సున్న")


_OR_UNITS = ["", "ଏକ", "ଦୁଇ", "ତିନି", "ଚାରି", "ପାଞ୍ଚ", "ଛଅ", "ସାତ", "ଆଠ", "ନଅ"]
_OR_TEENS = ["ଦଶ", "ଏଗାର", "ବାର", "ତେର", "ଚଉଦ", "ପନ୍ଦର", "ଷୋହଳ", "ସତର", "ଅଠର", "ଊଣେଇଶ"]
_OR_TENS  = ["", "", "କୋଡିଏ", "ତିରିଶ", "ଚାଳିଶ", "ପଚାଶ", "ଷାଠିଏ", "ସତୁରି", "ଅଶୀ", "ନବେ"]
_OR_SCALE = [(10**7, "କୋଟି"), (10**5, "ଲକ୍ଷ"), (1000, "ହଜାର"), (100, "ଶହ")]
def number_to_odia_words(n):
    return _indian_num_to_words(n, _OR_UNITS, _OR_TEENS, _OR_TENS, _OR_SCALE, "ଶୂନ୍ୟ")


def normalize_telugu_text(text):
    """Telugu ASR normalizer — mirrors the Hindi pipeline: NFC, native digits→words,
    strip bracketed content + punctuation (incl. danda), collapse whitespace.
    (Telugu has no nukta.)"""
    try:
        text = unicodedata.normalize('NFC', str(text))
        text = text.translate(_TE_DIGITS)
        text = _BRACKET_RE.sub('', text)
        text = re.sub(r'\d+', lambda m: number_to_telugu_words(int(m.group())), text)
        text = _PUNCT_RE.sub('', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text if text else "NA"
    except Exception:
        return "NA"


def normalize_odia_text(text):
    """Odia ASR normalizer — mirrors the Hindi pipeline: NFC, native digits→words,
    fold nukta (U+0B3C) AFTER number substitution (so inserted words fold too, unlike
    the Hindi port), strip bracketed content + punctuation (incl. danda), collapse."""
    try:
        text = unicodedata.normalize('NFC', str(text))
        text = text.translate(_OR_DIGITS)
        text = _BRACKET_RE.sub('', text)
        text = re.sub(r'\d+', lambda m: number_to_odia_words(int(m.group())), text)
        text = text.replace('଼', '')          # fold Odia nukta (after substitution)
        text = _PUNCT_RE.sub('', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text if text else "NA"
    except Exception:
        return "NA"


_LANG2FN = {
    "hi": normalize_hindi_text, "hin": normalize_hindi_text, "hindi": normalize_hindi_text,
    "te": normalize_telugu_text, "tel": normalize_telugu_text, "telugu": normalize_telugu_text,
    "or": normalize_odia_text, "od": normalize_odia_text, "ori": normalize_odia_text,
    "odia": normalize_odia_text, "oriya": normalize_odia_text,
}
def normalize(text, lang):
    """Dispatch to the language normalizer by code or name (hi/te/or, or hindi/telugu/odia).
    Unknown languages fall back to the Hindi pipeline (this corpus is only Hi/Te/Od).

    Apply to plain transcript text. If your text still carries diarization markers
    (e.g. 'S1:' speaker labels), strip them first — otherwise the digits would be
    converted to number words too (S1 → S<one>)."""
    return _LANG2FN.get(str(lang).strip().lower(), normalize_hindi_text)(text)


def corpus_wer(refs, hyps, normalizer=normalize_hindi_text):
    """Corpus-level (micro-average) WER over aligned lists, matching the paper."""
    from jiwer import wer
    R, H = [], []
    for r, h in zip(refs, hyps):
        rn, hn = normalizer(r), normalizer(h)
        if rn and rn != "NA" and hn and hn != "NA":
            R.append(rn); H.append(hn)
    if not R:
        return float("nan"), 0
    return wer(R, H), len(R)


def corpus_wer_by_lang(refs, hyps, langs):
    """Corpus-level (micro-average) WER where each row is normalized by ITS OWN
    language (hi/te/or). Pass aligned iterables of refs, hyps, and language codes/names."""
    from jiwer import wer
    R, H = [], []
    for r, h, lg in zip(refs, hyps, langs):
        rn, hn = normalize(r, lg), normalize(h, lg)
        if rn and rn != "NA" and hn and hn != "NA":
            R.append(rn); H.append(hn)
    if not R:
        return float("nan"), 0
    return wer(R, H), len(R)
