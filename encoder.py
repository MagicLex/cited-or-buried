"""Frozen text encoder, shared by F2 (page + query embeddings) and the predictor
(live query embedding). Embed-at-the-door: the store holds 384-d vectors, never
raw page text on the serving path.

BGE-small-en-v1.5 via fastembed (ONNX, no torch): a small CPU-only install that
builds a lean env. Loaded once per process (inert weights, not a lifecycle-owned
resource). Kept out of cited_features.py so structure-only paths stay dep-free.
"""

from __future__ import annotations

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384

_model = None


def _get():
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(model_name=MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """(n,) texts -> (n, 384) float32, L2-normalized. Blank text embeds to a zero
    vector so a failed page fetch does not poison the batch."""
    clean = [t if isinstance(t, str) and t.strip() else "" for t in texts]
    nonblank = [(i, t) for i, t in enumerate(clean) if t]
    out = np.zeros((len(clean), DIM), dtype="float32")
    if nonblank:
        vecs = np.array(list(_get().embed([t for _, t in nonblank])), dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.clip(norms, 1e-9, None)
        for (i, _), v in zip(nonblank, vecs):
            out[i] = v
    return out


def embed_one(text: str) -> list[float]:
    return embed([text])[0].tolist()
