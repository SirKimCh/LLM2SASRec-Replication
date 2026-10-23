import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lib  # noqa: F401  (puts vendor/llmseqrec on sys.path)
import models
from main.transformer.sasrec.sasrec_with_embeddings import SASRecWithEmbeddings


def test_shuffled_builds(tmp_path, monkeypatch):
    """The vendor __init__ logs extract_config(self), which reads every __init__ argument back as an attribute."""
    p = tmp_path / "emb.csv"
    pd.DataFrame({"ItemId": [1, 2, 3], "embedding": [json.dumps([float(i), 1.0]) for i in range(3)]}).to_csv(p, index=False)
    monkeypatch.setattr(SASRecWithEmbeddings, "product_embeddings", None)
    model, cfg = models.build("llm2sasrec_shuf", str(p), "palm", 1, 1, seed=7)
    assert model.shuffle_seed == 7 and cfg["shuffle_seed"] == 7
