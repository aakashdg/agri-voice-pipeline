"""Runtime phoneme-space fuzzy matcher over a controlled vocabulary — Devanagari-native.

A vocabulary corrector adapted to Devanagari: instead of Soundex (English/ASCII-only),
each token is converted to a phoneme string by an Indic G2P and scored against the
controlled agri lexicon by a FEATURE-WEIGHTED phoneme edit distance
(place/manner/voicing/aspiration/vowel-length via `indic_phon.pdist`). An n-gram window
(1-3 tokens) recovers terms the recognizer fragmented; a length gate suppresses spurious
matches. Homophone collisions are surfaced (not silently resolved) so a caller can leave
meaning-changing pairs alone.

Primitives are in `agri_voice_pipeline.phonetics.indic_phon` (phon, pdist, psim, skeleton).
Prior art for edit-distance-with-phonetic-cost: Kondrak (NAACL 2000); Zobel & Dart (SIGIR 1996).
"""
from collections import defaultdict
from .indic_phon import phon, skeleton, pdist, psim, norm_word  # noqa

class PhoneticMatcher:
    def __init__(self, vocab_words, min_sim=0.84, len_gate=3):
        # canonical vocabulary: {phon: set(surface canonicals)} — one-to-many keeps homophones
        self.min_sim = min_sim; self.len_gate = len_gate
        self.entries = []            # (surface, phon)
        self.homophone = defaultdict(set)
        self.by_first = defaultdict(list)
        seen = set()
        for w in vocab_words:
            w = norm_word(w); p = phon(w)
            if not p or w in seen:
                continue
            seen.add(w); self.entries.append((w, p)); self.homophone[p].add(w)
            self.by_first[p[0]].append((w, p))
        self._cache = {}

    def match(self, token):
        """Best phonetic match for a single token -> (surface, sim, is_collision) or None."""
        t = norm_word(token)
        if t in self._cache:
            return self._cache[t]
        pt = phon(t)
        res = None
        if pt:
            # exact phoneme hit (incl. homophone collisions) is the strongest signal
            if pt in self.homophone:
                cands = self.homophone[pt]
                surf = t if t in cands else sorted(cands)[0]
                res = (surf, 1.0, len(cands) > 1 or t not in cands and len(cands) >= 1 and any(c != surf for c in cands))
                res = (surf, 1.0, len(cands) > 1)
            else:
                best, best_sim = None, 0.0; ties = 0
                lo = pt[0]
                # candidate pool: same first phoneme (cheap, high recall for onset-preserving errors)
                pool = self.by_first.get(lo, [])
                # plus a light scan of near-length entries for onset errors (small vocab, ok)
                for w, p in (pool if pool else self.entries):
                    if abs(len(p) - len(pt)) > self.len_gate:
                        continue
                    s = psim(pt, p)
                    if s > best_sim + 1e-9:
                        best, best_sim, ties = w, s, 1
                    elif abs(s - best_sim) < 1e-9:
                        ties += 1
                if best is not None and best_sim >= self.min_sim:
                    res = (best, best_sim, ties > 1)
        self._cache[t] = res
        return res

    def correct_windows(self, tokens):
        """Return proposals [(i, span, token_text, surface, sim, collision)] over 1..3-grams,
        longest window first so fragmented terms win over their pieces."""
        props = []; used = [False] * len(tokens)
        for span in (3, 2, 1):
            for i in range(len(tokens) - span + 1):
                if any(used[i:i + span]):
                    continue
                joined = "".join(tokens[i:i + span])
                m = self.match(joined)
                if m and m[0] != norm_word(joined):
                    props.append((i, span, " ".join(tokens[i:i + span]), m[0], m[1], m[2]))
                    for k in range(i, i + span):
                        used[k] = True
        return props
