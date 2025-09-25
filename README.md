# Ανακατασκευή Κειμένων NLP

## Τι περιλαμβάνει
- Τρία pipelines ανακατασκευής: custom rules (spaCy), LanguageTool (γραμματική), και T5 paraphrasing.  
- Ανάλυση με embeddings και cosine similarity χρησιμοποιώντας **GloVe**, **FastText**, (προαιρετικά) **Word2Vec**, και **BERT/Sentence-Transformers**.  
- Διάσταση 2D με **PCA** και **t-SNE**.  
- Αναπαραγωγή αποτελεσμάτων με **Conda** + **Poetry**.  

## 0) Προαπαιτούμενα
- Python **3.10–3.12**  
- [Conda](https://docs.conda.io/en/latest/miniconda.html)  
- [Poetry](https://python-poetry.org/docs/#installation)  
- (Προαιρετικά) **Java 11+** αν θέλετε να τρέχει το LanguageTool τοπικά.  

## 1) Δημιουργία περιβάλλοντος
```bash
conda create -n nlp-recon python=3.11 -y
conda activate nlp-recon

# Αν δεν έχετε εγκατεστημένο το poetry:
# pipx install poetry   # ή: pip install --user poetry
poetry install
```

> Αν το `torch` αποτύχει στην εγκατάσταση στο σύστημά σας, εγκαταστήστε πρώτα μια συμβατή έκδοση και ξανατρέξτε `poetry install`.

## 2) (Προαιρετικό) LanguageTool
Το LanguageTool χρησιμοποιείται για έλεγχο γραμματικής. Απαιτεί Java.  
Αν η Java λείπει, ο κώδικας θα **παρακάμψει** αυτό το pipeline αυτόματα.
```bash
java -version  # πρέπει να εμφανίσει κάτι σαν 'openjdk version "17..."'
```

## 3) Κατέβασμα μικρού αγγλικού μοντέλου spaCy
```bash
python -m spacy download en_core_web_sm
```

## 4) Εκτέλεση Παραδοτέου 1 (Ανακατασκευές)
```bash
poetry run python scripts/run_pipelines.py
```
Τα αποτελέσματα αποθηκεύονται σε:
- `outputs/reconstructions.json`  
- `outputs/reconstructed_texts.txt`

## 5) Εκτέλεση Παραδοτέου 2 (Embeddings + Ομοιότητα + Γραφήματα)
```bash
poetry run python scripts/run_embeddings.py
```
Θα παραχθούν:
- `outputs/similarity_scores.csv`  
- `outputs/embeddings_2d_pca.png`  
- `outputs/embeddings_2d_tsne.png`

## 6) (Προαιρετικό) Εκτέλεση χωρίς βαριά μοντέλα
Το `run_embeddings.py` χρησιμοποιεί **GloVe** και **FastText** μέσω `gensim.downloader`.  
Αν δεν είναι δυνατή η λήψη (π.χ. περιορισμένο δίκτυο), μπορείτε να κατεβάσετε τα μοντέλα χειροκίνητα ή να προσαρμόσετε τον κώδικα ώστε να χρησιμοποιεί τοπικές διαδρομές.
