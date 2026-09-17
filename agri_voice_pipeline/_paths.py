"""Locate the bundled data directory (lexicon + corpus counts).

Search order: the ``AVP_DATA_DIR`` environment variable, then ``data/`` beside the
repository, then ``./data`` in the current working directory. Set ``AVP_DATA_DIR``
if you pip-install the package away from the checked-out ``data/`` folder.
"""
import os
from pathlib import Path


def data_dir() -> Path:
    env = os.environ.get("AVP_DATA_DIR")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parents[1]      # parent of the package dir
    d = repo_root / "data"
    if (d / "lexicon_clean.csv").exists():
        return d
    cwd = Path.cwd() / "data"
    if (cwd / "lexicon_clean.csv").exists():
        return cwd
    return d
