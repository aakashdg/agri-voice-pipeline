"""agri_voice_pipeline — reusable methods for a model- and language-agnostic
agricultural voice pipeline (Hindi / Telugu / Odia).

Public code companion to the paper. Submodules:
  phonetics     — script-unified Indic G2P + feature-weighted phoneme distance
  normalization — Hindi / Telugu / Odia ASR text normalizers
  correction    — precision-first, attested-only domain-term recovery (no LLM)
  evaluation    — deterministic Agriculture-Weighted WER (AWWER)
  selection     — energy-based dominant (farmer) speaker selection
"""
__version__ = "0.1.0"
