"""End-to-end demo of the reusable methods. Runs with no ASR and no audio files:

    python examples/demo.py

Touches every module on tiny in-repo inputs (the bundled lexicon + synthetic audio).
"""
import sys
from pathlib import Path

# Run from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from agri_voice_pipeline.normalization import normalize
from agri_voice_pipeline.phonetics import phon, psim, PhoneticMatcher
from agri_voice_pipeline.correction import AgriCorrector
from agri_voice_pipeline.evaluation import awwer
from agri_voice_pipeline.selection import select_best_speaker


def hr(title):
    print("\n" + title)
    print("-" * len(title))


hr("1. Normalization (Hi / Te / Od) — digits become words, punctuation drops")
print("hi:", normalize("2 बोरी यूरिया डालें?", "hindi"))
print("te:", normalize("3 ఎకరాలు.", "telugu"))
print("od:", normalize("୫ କିଲୋ ବିହନ।", "odia"))

hr("2. Phonetics — sound-alikes score high, meaning-changers score low")
for a, b, note in [("दवा", "दवाई", "same meaning (medicine)"),
                   ("कीड़ा", "खीरा", "DIFFERENT meaning (insect vs cucumber)")]:
    print(f"{a} ~ {b:6s} phon {phon(a):6s}/{phon(b):6s}  sim={psim(phon(a), phon(b)):.2f}   ({note})")

hr("3. Phonetic matcher — recover a garble against a small controlled vocabulary")
pm = PhoneticMatcher(["खरपतवार", "कीटनाशक", "यूरिया", "सिंचाई"], min_sim=0.84)
print("खरपतवा ->", pm.match("खरपतवा"))     # (surface, similarity, is_collision)

hr("4. Correction — precision-first OOV agri-term recovery (uses the bundled lexicon)")
c = AgriCorrector()
out = c.correct("खेत में खरपतवा बहुत हो गया है")
print("corrected:", out["corrected"])
print("edits    :", out["edits"])

hr("5. AWWER vs WER — a hallucinated agri term is up-weighted")
refs = ["किसान खेत में यूरिया डाल रहा है"]
hyps = ["किसान खेत में यूरिया कीटनाशक डाल है"]
aw, we = awwer(refs, hyps, ["hindi"])
print(f"AWWER={aw}  WER={we}")

hr("6. Speaker selection — pick the louder (near-field) speaker")
sr = 16000
audio = np.zeros(sr * 6, dtype="float32")
audio[0:sr * 3] = 0.05 * np.random.randn(sr * 3)       # speaker A, quiet, 0-3s
audio[sr * 3:sr * 6] = 0.30 * np.random.randn(sr * 3)  # speaker B, loud,  3-6s
segments = [{"start": 0.0, "end": 3.0, "speaker": "A"},
            {"start": 3.0, "end": 6.0, "speaker": "B"}]
spk, sliced = select_best_speaker(segments, audio, sr)
print(f"chosen speaker: {spk}  (sliced {len(sliced) / sr:.1f}s of the louder speaker)")
