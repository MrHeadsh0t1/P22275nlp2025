
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from nlp_reconstruction.data import TEXT1, TEXT2
from nlp_reconstruction.preprocess import sentencize_spacy
from nlp_reconstruction.embeddings import get_gensim_vectors, bert_sentence_embedding, cosine_between_texts_wordspace, sent_embedding_wordspace
from nlp_reconstruction.pipelines import CustomRuleRewriter, LanguageToolCorrector, T5Paraphraser
from nlp_reconstruction.similarity import project_2d

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Προσπαθήστε να φορτώσετε μικρού μεγέθους προ-εκπαιδευμένους διανύσματα λέξεων
WIKI_GLOVE = "glove-wiki-gigaword-100"         # 128MB
WIKI_FASTTEXT = "fasttext-wiki-news-subwords-300"  # 1GB 
GOOGLE_W2V = "word2vec-google-news-300"        # 1.5GB 

def safe_load_vectors(name: str):
    try:
        kv = get_gensim_vectors(name)
        print(f"Loaded vectors: {name}")
        return kv
    except Exception as e:
        print(f"Could not load {name} due to: {e}")
        return None

def sentence_level_matrix(original: str, reconstructed: str, kv) -> float:
    orig_sents = sentencize_spacy(original)
    reco_sents = sentencize_spacy(reconstructed)
    n = min(len(orig_sents), len(reco_sents))
    if n == 0:
        return 0.0
    sims = []
    for i in range(n):
        a = sent_embedding_wordspace(orig_sents[i], kv)
        b = sent_embedding_wordspace(reco_sents[i], kv)
        # cosine
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            sims.append(0.0)
        else:
            sims.append(float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))))
    return float(np.mean(sims))

def main():
    texts = [TEXT1, TEXT2]

    # Εκτέλεση Pipelines για την απόκτηση ανακατασκευών
    custom = CustomRuleRewriter()
    lt = LanguageToolCorrector()
    t5 = T5Paraphraser()

    recon = {
        "custom_rules": [custom(TEXT1), custom(TEXT2)],
        "languagetool": [lt(TEXT1), lt(TEXT2)],
        "t5_paraphrase": [t5(TEXT1), t5(TEXT2)],
    }

    # Φόρτωση ενσωματώσεων
    glove = safe_load_vectors(WIKI_GLOVE)
    ftext = safe_load_vectors(WIKI_FASTTEXT)
    
    w2v = safe_load_vectors(GOOGLE_W2V)

    # Ενσωματώσεις BERT/SentenceTransformer (σε επίπεδο πρότασης)
    bert_texts = []
    labels = []
    for idx, t in enumerate(texts):
        bert_texts.append(f"ORIG_{idx+1}: " + t)
    for k, vs in recon.items():
        for idx, t in enumerate(vs):
            bert_texts.append(f"{k.upper()}_{idx+1}: " + t)

    # Εξαγωγή μόνο του κειμένου για ενσωμάτωση
    bert_inputs = [bt.split(": ", 1)[1] for bt in bert_texts]
    bert_embs = bert_sentence_embedding(bert_inputs)  # normalized

    
    rows = []
    for name, kv in [("glove", glove), ("fasttext", ftext), ("word2vec", w2v)]:
        if kv is None: 
            continue
        for i, orig in enumerate(texts):
            for k, vs in recon.items():
                sim_doc = cosine_between_texts_wordspace(orig, vs[i], kv)
                sim_sent = sentence_level_matrix(orig, vs[i], kv)
                rows.append({
                    "embedding": name,
                    "text": f"TEXT{i+1}",
                    "pipeline": k,
                    "cosine_doc": sim_doc,
                    "cosine_sent_avg": sim_sent,
                })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "similarity_scores.csv", index=False)
    print("Saved", OUT_DIR / "similarity_scores.csv")


    labels = []
    for i in range(len(texts)):
        labels.append(f"ORIG_{i+1}")
    for k in recon.keys():
        for i in range(len(texts)):
            labels.append(f"{k.upper()}_{i+1}")

    for method in ("pca", "tsne"):
        coords = project_2d(bert_embs, method=method)
        plt.figure()
        xs, ys = coords[:,0], coords[:,1]
        for lbl, x, y in zip(labels, xs, ys):
            plt.scatter(x, y)
            plt.text(x, y, lbl, fontsize=8)
        plt.title(f"BERT sentence embeddings ({method.upper()})")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"embeddings_2d_{method}.png", dpi=160)
        plt.close()
        print("Saved", OUT_DIR / f"embeddings_2d_{method}.png")

if __name__ == "__main__":
    main()
