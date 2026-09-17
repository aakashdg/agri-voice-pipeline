"""awwer.py — deterministic Agriculture-Weighted Word Error Rate (AWWER).

AWWER = sum(weight over error tokens) / sum(weight over reference tokens), with each
token weighted by agricultural criticality (4/3/2/1). A reference token's weight is its
lexicon `relevance_mode` (4/3/2) when the normalized token is in the agri lexicon, else 1.

  - Substitutions and deletions are charged the REFERENCE token's weight.
  - Insertions are charged the INSERTED (hypothesis) token's own criticality weight —
    symmetric with sub/del. This penalizes a hallucinated agri term while ignoring a
    hallucinated function word, and avoids the insertion-dilution artifact that charging
    every insertion weight 1 produces on an over-inserting recognizer.

Text is normalized with the same normalizer used for plain WER, so AWWER and WER are
directly comparable. This module is deterministic: no model, no hand-typed numbers.
"""
import csv
from ..normalization.normalize import normalize as N
from .._paths import data_dir


def load_weights(lexicon_csv=None):
    """token -> agricultural criticality weight (4/3/2), from lexicon_clean.csv.
    Tokens absent from the lexicon are weight 1 (handled by `weight_of`)."""
    path = lexicon_csv or (data_dir() / "lexicon_clean.csv")
    w = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        t = (r.get("word_norm") or r.get("word") or "").strip()
        try:
            rel = int(float(r.get("relevance_mode") or 1))
        except Exception:
            rel = 1
        if t:
            w[t] = max(w.get(t, 1), rel)
    return w


def weight_of(token, weights):
    return weights.get(token, 1)


def awwer(refs, hyps, langs, weights=None):
    """Corpus-level AWWER and plain WER over aligned iterables.

    refs, hyps, langs : aligned sequences of reference text, hypothesis text, and language
                        code/name (hi/te/or or the full names) per clip.
    weights           : token->weight dict (defaults to the bundled lexicon).

    Returns (awwer, wer), each rounded to 4 dp. Both use the same normalizer, so the gap
    between them is purely the agricultural re-weighting.
    """
    import jiwer
    if weights is None:
        weights = load_weights()
    num = den = 0.0          # weighted (AWWER)
    wnum = wden = 0          # unweighted (plain WER), for comparison
    for ref, hyp, lang in zip(refs, hyps, langs):
        rn = N(ref, lang)
        if not rn or rn == "NA":
            continue
        hn = N(hyp, lang) if (hyp and str(hyp).strip() and not str(hyp).startswith("__ERROR__")) else ""
        if hn == "NA":
            hn = ""
        rt = rn.split()
        ht = (hn or " ").split()
        o = jiwer.process_words([rn], [hn if hn else " "])
        den += sum(weight_of(t, weights) for t in rt)
        wden += len(rt)
        for ch in o.alignments[0]:
            if ch.type == "equal":
                continue
            if ch.type in ("substitute", "delete"):
                seg = rt[ch.ref_start_idx:ch.ref_end_idx]      # charge the intended (ref) token
                num += sum(weight_of(t, weights) for t in seg)
                wnum += len(seg)
            if ch.type == "insert":
                seg = ht[ch.hyp_start_idx:ch.hyp_end_idx]      # charge the inserted (hyp) token
                num += sum(weight_of(t, weights) for t in seg)
                wnum += len(seg)
    return (round(num / max(den, 1), 4), round(wnum / max(wden, 1), 4))


if __name__ == "__main__":
    # A hallucinated agri term (कीटनाशक, weight 3) costs more than a dropped function word.
    refs = ["किसान खेत में यूरिया डाल रहा है"]
    hyps = ["किसान खेत में यूरिया कीटनाशक डाल है"]     # + inserted agri term, - one function word
    aw, we = awwer(refs, hyps, ["hindi"])
    print(f"AWWER={aw}  WER={we}   (AWWER > WER because the hallucinated agri term is up-weighted)")
