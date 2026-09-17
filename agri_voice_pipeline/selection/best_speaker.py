"""best_speaker.py — energy-based dominant (farmer) speaker selection.

After diarization gives per-speaker segments, the service still has to answer ONE
person's question — the farmer holding the phone. This picks that speaker so a
recognizer is not penalized for also transcribing bystanders.

The rule is deliberately simple and recognizer-invariant (it sees only segments +
audio, never any ASR output): among speakers who hold the floor for at least
max(1.0s, 0.25 x the longest speaker's total), choose the one with the highest
mean RMS energy (the near-field speaker into the mic); fall back to the
longest-duration speaker when none clear the duration floor. It carries no learned
parameters, so it transfers across languages unchanged.

Depends only on numpy. `select_best_speaker` returns the chosen speaker label;
`slice_speaker` concatenates that speaker's audio (with a small boundary pad) for
re-transcription.
"""
from collections import defaultdict
import numpy as np

PAD = 0.3            # boundary padding (seconds) when slicing the chosen speaker
MIN_SPK_S = 1.0      # absolute duration floor for "holding the floor"
REL_SPK = 0.25       # relative floor: 0.25 x the longest speaker's total duration


def qualifying_speakers(segments, min_s=MIN_SPK_S):
    """Speakers whose total speaking time is at least `min_s` seconds."""
    dur = defaultdict(float)
    for s in segments:
        dur[s["speaker"]] += s["end"] - s["start"]
    return [sp for sp, d in dur.items() if d >= min_s]


def robust_pick(segments, audio, sr):
    """Return the dominant speaker label, or None if there are no segments.

    segments : list of {"start", "end", "speaker"} (seconds).
    audio    : 1-D numpy array (mono).
    sr       : sample rate (Hz).
    """
    dur = defaultdict(float)
    energy = defaultdict(float)
    for s in segments:
        a = int(s["start"] * sr)
        b = min(int(s["end"] * sr), len(audio))
        dur[s["speaker"]] += s["end"] - s["start"]
        energy[s["speaker"]] += float(np.sum(audio[a:b] ** 2))
    if not dur:
        return None
    floor = max(MIN_SPK_S, REL_SPK * max(dur.values()))
    eligible = [sp for sp in dur if dur[sp] >= floor] or [max(dur, key=dur.get)]
    rms = {sp: (energy[sp] / max(dur[sp] * sr, 1)) ** 0.5 for sp in eligible}
    return max(rms, key=rms.get)


def slice_speaker(segments, speaker, audio, sr, pad=PAD):
    """Concatenate `speaker`'s segments (+/- `pad`); fall back to the whole clip if
    the result is too short to transcribe reliably."""
    pieces = []
    for s in segments:
        if s["speaker"] != speaker:
            continue
        a = max(0, int((s["start"] - pad) * sr))
        b = min(len(audio), int((s["end"] + pad) * sr))
        if b > a:
            pieces.append(audio[a:b])
    y = np.concatenate(pieces) if pieces else audio
    if len(y) < int(0.2 * sr):
        y = audio
    return y


def select_best_speaker(segments, audio, sr):
    """Convenience wrapper: returns (speaker_label, sliced_audio) for the dominant
    speaker. If there are no segments, returns (None, audio) unchanged."""
    spk = robust_pick(segments, audio, sr)
    if spk is None:
        return None, audio
    return spk, slice_speaker(segments, spk, audio, sr)
