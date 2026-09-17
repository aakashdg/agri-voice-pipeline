# Bundled data

These files are a verbatim copy of the openly-released agricultural lexicon, so the
`correction` and `evaluation` modules run out of the box. The canonical source is:

**`DigiGreen/agri-lexicon-hindi`** — https://huggingface.co/datasets/DigiGreen/agri-lexicon-hindi

| File | Rows | What it is |
|---|---|---|
| `lexicon_clean.csv` | 18,646 | Controlled agricultural vocabulary — each term with an agronomic category and a 4/3/2 criticality weight. Drives correction targeting and AWWER weighting. |
| `pair_labels.csv` | 13,904 | Sound-alike term pairs adjudicated `SAME_WORD` / `SAME_MEANING` / `DIFF_MEANING` / `UNRELATED`, with cues. Used to fold spelling variants and to keep meaning-changing pairs apart. |
| `form_bg_counts.json` | 20,847 | Corpus token-frequency table (counts only — no transcripts, no text spans). The corrector's out-of-vocabulary gate: a token unseen here is treated as a likely ASR garble. |

**License:** CC-BY-4.0, © 2026 Digital Green. If you use this lexicon, please cite Digital
Green's agricultural voice-pipeline paper and attribute the dataset above.

No personal data: these are agricultural vocabulary terms, sound-alike pairs, and frequency
counts — not farmer transcripts or recordings.
