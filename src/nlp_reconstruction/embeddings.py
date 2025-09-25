
from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def mean_pool_word_vectors(tokens: List[str], kv) -> np.ndarray:
    vecs = []
    for w in tokens:
        if w in kv:
            vecs.append(kv[w])
    if not vecs:
        return np.zeros(kv.vector_size, dtype=float)
    return np.mean(vecs, axis=0)

def tokenize_basic(text: str) -> List[str]:
    import re
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    toks = text.split()
    return toks

def sent_embedding_wordspace(text: str, kv) -> np.ndarray:
    toks = tokenize_basic(text)
    return mean_pool_word_vectors(toks, kv)

def pairwise_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(1, -1); b = b.reshape(1, -1)
    cs = cosine_similarity(a, b)[0,0]
    return float(cs)

def get_gensim_vectors(name: str):
    import gensim.downloader as api
    return api.load(name)

def bert_sentence_embedding(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, normalize_embeddings=True)
    return np.asarray(embs)

def cosine_between_texts_wordspace(original: str, reconstructed: str, kv) -> float:
    a = sent_embedding_wordspace(original, kv)
    b = sent_embedding_wordspace(reconstructed, kv)
    return pairwise_cosine(a, b)
