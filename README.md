# agri-voice-pipeline

Reusable methods from the **Model-Agnostic, Language-Agnostic Voice Pipeline for Agriculture** — the
public code companion to the paper. This repository packages the *novel, reusable components* as a clean,
importable library. It contains **no farmer audio, transcripts, or production data** — only methods, plus
the openly-released agricultural lexicon (mirrored from [DigiGreen/agri-lexicon-hindi](https://huggingface.co/datasets/DigiGreen/agri-lexicon-hindi)).

> **Status: code being added.** This README is the planned layout for review; modules land next.

## What's here (planned contents)

| Module | Novel contribution | Source |
|---|---|---|
| `phonetics/indic_phon.py` | **Script-unified Indic G2P + feature-weighted phoneme edit distance** — rule-based grapheme→phoneme with word-final and rule-governed medial **schwa deletion**, a shared phoneme inventory across Devanagari/Telugu/Odia, and substitution costs from phonological feature distance (not Soundex). | `hindi_phon.py` |
| `phonetics/phonetic_matcher.py` | Runtime fuzzy matcher over a controlled vocabulary in phoneme space, with n-gram windowing and homophone-collision handling. | `phonetic_matcher.py` |
| `normalization/normalize.py` | **Hindi / Telugu / Odia ASR text normalizer** — NFC, nukta-fold, bracket/punctuation stripping, and **compositional Indian-numbering number-words** (…crore/lakh/thousand) per language. | `normalize.py` |
| `correction/corrector.py` | **Precision-first, attested-only domain-term correction (no LLM)** — recovers corpus-unseen garbles that uniquely phoneme-match an attested high-criticality term; leaves valid words and meaning-changing homophones alone. | `corrector_v3.py` |
| `evaluation/awwer.py` | **Deterministic Agriculture-Weighted WER** — weights each error by its agricultural criticality (4/3/2/1), charging inserted tokens by their own criticality (symmetric with sub/del). | `eval_awwer_det.py` |
| `selection/best_speaker.py` | **Energy-based dominant-speaker selection** heuristic with a duration floor — pick the farmer's turns after diarization. | pipeline_core selection logic |
| `lexicon/build/` | The **seven-step lexicon construction** (clean → cluster by phonetic similarity → adjudicate sound-alike pairs into same-meaning vs meaning-changing). | `clean_lexicon.py`, `cluster_lexicon.py`, `adjudicate.py` |
| `data/form_bg_counts.json` | Corpus token-frequency table used by the corrector's OOV gate (counts only, no text/PII). | — |
| `examples/demo.py` | Runnable end-to-end example on a tiny bundled sample (no PII). | new |

## Not included (by design)
- No farmer audio, transcripts, or the evaluation corpus (privacy). The evaluation set is released
  separately as [DigiGreen/agri-voice-eval](https://huggingface.co/datasets/DigiGreen/agri-voice-eval).
- No API keys, provider credentials, or internal paths.
- The full experiment/pipeline runner lives in the project's working repository; this repo is the reusable
  methods only.

## Related open-source releases
- Fine-tuned diarization segmenter — [DigiGreen/pyannote-segmentation-agri-indic](https://huggingface.co/DigiGreen/pyannote-segmentation-agri-indic)
- Agricultural lexicon — [DigiGreen/agri-lexicon-hindi](https://huggingface.co/datasets/DigiGreen/agri-lexicon-hindi)
- Evaluation set — [DigiGreen/agri-voice-eval](https://huggingface.co/datasets/DigiGreen/agri-voice-eval)
- Demo — [DigiGreen/farmerchat-voice-pipeline-demo](https://huggingface.co/spaces/DigiGreen/farmerchat-voice-pipeline-demo)

## License
MIT (code). The bundled lexicon is CC-BY-4.0. Please cite the paper.
