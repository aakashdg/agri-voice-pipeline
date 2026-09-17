"""Precision-first, attested-only agricultural term correction — no LLM, WER-safe.

DESIGN.  An earlier version canonicalized valid words (variant -> family head, e.g.
दवाई -> दवा). Because a faithful reference keeps the speaker's ACTUAL words (not a
canonical form), most such edits rewrote a token that was already correct — net-negative
on WER. The fix is a change of objective, matching the high-precision post-editing
literature ("Tag and correct", arXiv:2406.07589): only touch tokens that are almost
certainly ASR ERRORS, never rewrite a valid word.

WHAT IT DOES (no LLM):
  OOV agri-term recovery.  A token is a correction candidate only if it is essentially
    UNSEEN in the mining corpus (bg <= oov_max => not a real word, a genuine ASR garble),
    is not already a domain term, and has a UNIQUE feature-weighted phoneme match
    (indic_phon.pdist) to a high-criticality agri term (relevance >= rel_min) at
    sim >= min_sim, where that target is itself attested in the corpus (>= target_min_bg).
    Homophone collisions, non-unique matches, and same-word diacritic variants are left
    alone. These guards keep it WER-safe across recognizers without a per-stratum knob.
  Meaning-changing confusions (e.g. दवा/दावा) are NOT touched here — they need sentence
    context and are deferred to an optional contextual pass, kept out of the no-LLM path.

LANGUAGE.  The phoneme primitives are Devanagari, so recovery runs on Hindi; Telugu and
Odia tokens pass through unchanged until their lexica/phonetics mature. Non-Devanagari
text is returned untouched, so applying this to a Te/Od transcript is a safe no-op.

Also exposes domain_terms(): the crop/pest/chemical terms present in a token list, folded
so SAME_* spelling variants count as the same term — the basis for a domain-term-recall
metric (did the farmer's term survive?), applied symmetrically to reference and hypothesis.
"""
import re, json, csv, unicodedata
from pathlib import Path
from .._paths import data_dir

_DATA = data_dir()
LEXICON = _DATA / "lexicon_clean.csv"
PAIR_LABELS = _DATA / "pair_labels.csv"
BG_COUNTS = _DATA / "form_bg_counts.json"

DEVANAGARI = re.compile(r"[ऀ-ॿ]")

def _norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s or "")).strip())

def _is_devanagari(tok):
    return bool(DEVANAGARI.search(tok))

# diacritic fold: chandrabindu->anusvara, candra-O->O, drop nukta/virama. Two tokens equal under
# this fold are the SAME word spelled differently (e.g. पाँच/पांच, रूटस्टॉक/रूटस्टोक) — not an ASR
# error, so "recovering" one to the other only churns WER against a reference that kept the original.
_FOLD = str.maketrans({"ँ": "ं", "ॉ": "ो", "ॲ": "अ"})
def _fold(tok):
    return unicodedata.normalize("NFKC", tok).translate(_FOLD).replace("़", "").replace("्", "")

def _trivial_variant(a, b):
    return _fold(a) == _fold(b)

class AgriCorrector:
    def __init__(self, oov_max=0, rel_min=3, min_sim=0.90, target_min_bg=1):
        self.oov_max = oov_max        # a token seen <= this many times in the corpus is treated as OOV/garble
        self.rel_min = rel_min        # only recover toward domain terms of at least this criticality
        self.min_sim = min_sim
        self.target_min_bg = target_min_bg  # only recover TOWARD an agri term the ASR actually produces
                                            # (attested >= this many times in the corpus). This is what
                                            # makes it WER-safe on every recognizer WITHOUT a stratum gate:
                                            # a "recovery" toward a term the recognizer never emits (bg=0)
                                            # is almost always spurious (e.g. dialectal एगो -> unseen एको).
        # --- lexicon: relevance per token ---
        self.rel = {}
        for r in csv.DictReader(open(LEXICON)):
            t = (r.get("word_norm") or r.get("word") or "").strip()
            if not t:
                continue
            try:
                v = int(float(r.get("relevance_mode") or 1))
            except Exception:
                v = 1
            self.rel[t] = max(self.rel.get(t, 1), v)
        self.agri = {w for w, v in self.rel.items() if v >= rel_min and " " not in w}
        # --- corpus frequency (OOV gate denominator) ---
        self.bg = {}
        if BG_COUNTS.exists():
            try:
                self.bg = json.loads(BG_COUNTS.read_text())
            except Exception:
                self.bg = {}
        # --- phonetic matcher over the high-criticality agri vocabulary ---
        from ..phonetics.phonetic_matcher import PhoneticMatcher
        self.pm = PhoneticMatcher(sorted(self.agri), min_sim=min_sim)
        # --- SAME_* variant->canonical map, for the domain-term-recall metric ONLY ---
        self.vmap = {}
        for r in csv.DictReader(open(PAIR_LABELS)):
            if r["label"] not in ("SAME_WORD", "SAME_MEANING"):
                continue
            c, a = _norm(r["candidate"]), _norm(r["anchor"])
            if c and a and c != a and " " not in c and " " not in a and self.rel.get(a, 1) >= rel_min:
                self.vmap.setdefault(c, a)

    def _tokens(self, text):
        return [w for w in re.sub(r"[^\w\sऀ-ॿ]", " ", _norm(text)).split() if w.strip()]

    def _apply(self, text, frm, to):
        return re.sub(rf"(?<!\S){re.escape(frm)}(?!\S)", to, text, count=1)

    def correct(self, text, lang="hindi", **_):
        """OOV-gated agri-term recovery. Returns {corrected, changed, edits, ...}. No LLM, WER-safe."""
        out = _norm(text); edits = []
        if not out or out == "NA":
            return {"corrected": text, "changed": False, "edits": []}
        for t in self._tokens(out):
            if not _is_devanagari(t):            # Hindi-only mechanism; Te/Od + latin pass through
                continue
            if self.bg.get(t, 0) > self.oov_max:  # a real/common word — never touch
                continue
            if t in self.agri:                    # already a valid domain term
                continue
            m = self.pm.match(t)
            if not m:
                continue
            surf, sim, collision = m
            if collision or surf == t or sim < self.min_sim:
                continue                          # ambiguous or no real change -> leave it
            if _trivial_variant(t, surf):         # same word, different diacritics -> not an error
                continue
            if self.bg.get(surf, 0) < self.target_min_bg:  # only recover toward an attested agri term
                continue
            new = self._apply(out, t, surf)
            if new != out:
                out = new
                edits.append({"from": t, "to": surf, "sim": round(sim, 3), "tier": "oov_recovery"})
        return {"corrected": out, "changed": bool(edits), "edits": edits,
                "n_recovered": len(edits)}

    # ---- evaluation helper: domain-term recall (the metric this component targets) ----
    def domain_terms(self, normed_tokens):
        """agri (rel>=rel_min) terms in a token list, folded to canonical (variants count as matches)."""
        out = []
        for t in normed_tokens:
            if t in self.agri:
                out.append(self.vmap.get(t, t))
            elif t in self.vmap:
                out.append(self.vmap[t])
        return out


if __name__ == "__main__":
    import sys
    c = AgriCorrector()
    print(f"agri vocab (rel>={c.rel_min}) {len(c.agri)} | variant map {len(c.vmap)} "
          f"| oov_max={c.oov_max} min_sim={c.min_sim}")
    # default recovers खरपतवा (OOV ASR garble) -> खरपतवार ("weed", attested)
    demo = sys.argv[1] if len(sys.argv) > 1 else "खेत में खरपतवा बहुत हो गया है"
    print(json.dumps(c.correct(demo), ensure_ascii=False, indent=2))
