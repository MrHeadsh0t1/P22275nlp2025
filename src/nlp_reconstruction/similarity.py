
from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(1, -1); b = b.reshape(1, -1)
    return float(cosine_similarity(a, b)[0,0])

def project_2d(X: np.ndarray, method: str = "pca", perplexity: int | None = 15, random_state: int = 42) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(X)
    elif method == "tsne":
        n = X.shape[0]
        # t-SNE απαιτεί perplexity < n_samples
        if perplexity is None:
            per = max(2, min(30, n // 3))
        else:
            per = min(perplexity, n - 1 if n > 1 else 1)
            per = max(2, per)
        return TSNE(n_components=2, perplexity=per, random_state=random_state, init="random").fit_transform(X)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")
