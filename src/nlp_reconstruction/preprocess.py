
from __future__ import annotations
import re
from typing import List

def sentencize_spacy(text: str) -> List[str]:
    """Διαχωρίζει το κείμενο σε προτάσεις με χρήση του spaCy, 
    αν είναι διαθέσιμο· διαφορετικά γίνεται απλός διαχωρισμός με regex.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        return [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    except Exception:
       
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

def normalize_spaces(text: str) -> str:
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([(\[\{])\s+', r'\1', text)
    text = re.sub(r'\s+([)\]\}])', r'\1', text)
    return text

def capitalize_sentences(text: str) -> str:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s[0:1].upper() + s[1:] if s else s for s in sents]
    return " ".join(sents).strip()

def simple_clean(text: str) -> str:
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.replace(' ,', ',').replace(' .', '.').replace(' ?', '?')
    text = text.replace('“', '"').replace('”', '"').replace('—', '-')
    return normalize_spaces(text)

def pick_two_sentences(text: str) -> List[str]:
    sents = sentencize_spacy(text)
    if len(sents) >= 2:
        return [sents[0], sents[-1]]
    return sents
