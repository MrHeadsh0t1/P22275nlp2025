
from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import shutil
from .preprocess import simple_clean, normalize_spaces, capitalize_sentences, sentencize_spacy

class CustomRuleRewriter:
    """Ένας απλός, ντετερμινιστικός αναδιατυπωτής που χρησιμοποιεί έτοιμους κανόνες
    για αντικατάσταση φράσεων και διορθώσεις σημείων στίξης/γραμματικής"""
    PHRASES = {
        "Thank your message": "Thank you for your message",
        "final discuss": "final discussion",
        "the updates was": "the updates were",
        "as it not included": "as they did not include",
        "bit delay": "a slight delay",
        "at recent days": "in recent days",
        "tried best": "did their best",
        "the doctor": "the doctor",
        "he sending again": "he sends it again",
        "I am very appreciated": "I greatly appreciate",
        "our lives": "our lives",
        "I mean": "",
        "I think": "",
        "Anyway,": "",
        "Overall,": "Overall,",
    }

    def rewrite_sentence(self, sent: str) -> str:
        s = simple_clean(sent)
        for k, v in self.PHRASES.items():
            s = s.replace(k, v)
        # Μικροί ευριστικοί κανόνες για γραμματική/στίξη
        s = s.replace("Hope you too, to enjoy it", "I hope you enjoy it too")
        s = s.replace("with all safe and great", "wishing everyone safety and well-being")
        s = s.replace("as his next contract checking", "for his next contract review")
        s = s.replace("I got this message to see the approved message", "I received confirmation of the approval")
        s = s.replace("to show me, this, a couple of days ago", "a couple of days ago")
        s = s.replace("Springer proceedings publication", "Springer Proceedings publication")
        s = s.replace("less communication", "less communication")
        s = s.replace("acknowledgments section", "Acknowledgments section")
        s = s.replace("make sure all are safe", "make sure everyone is safe")
        s = normalize_spaces(s)
        return s

    def __call__(self, text: str) -> str:
        sents = sentencize_spacy(text)
        out = [self.rewrite_sentence(s) for s in sents]
        return capitalize_sentences(" ".join(out))

class LanguageToolCorrector:
    """Διόρθωση γραμματικής με χρήση του LanguageTool (απαιτεί Java).
    Αν δεν υπάρχει διαθέσιμη Java, κάνει αυτόματο graceful skip.
    """
    def __init__(self, language: str = "en-US"):
        self.language = language
        self.available = shutil.which("java") is not None
        self.lt = None
        if self.available:
            try:
                import language_tool_python as lt
                self.lt = lt.LanguageToolPublicAPI(language) if os.environ.get("LT_PUBLIC") else lt.LanguageTool(language)
            except Exception:
                self.available = False

    def __call__(self, text: str) -> str:
        if not self.available or self.lt is None:
             # Επιστρέφει απλώς καθαρισμένο κείμενο αν το LanguageTool δεν είναι διαθέσιμο
            return capitalize_sentences(simple_clean(text))
        corrected = self.lt.correct(text)
        return capitalize_sentences(simple_clean(corrected))

class T5Paraphraser:
    """Παραφραστής κειμένου με χρήση μοντέλου T5 (αν υπάρχει paraphrase-tuned το φορτώνει,
    αλλιώς χρησιμοποιεί το γενικό T5 με paraphrase prompt).
    """
    def __init__(self, model_name: Optional[str] = None, device: Optional[int] = None):
        from transformers import pipeline
        self.model_name = model_name or "Vamsi/T5_Paraphrase_Paws"
        try:
            self.pipe = pipeline("text2text-generation", model=self.model_name, device=device)
            self.prompt_template = "paraphrase: {text}"
        except Exception:
            # Αν αποτύχει, fallback σε βασικό μοντέλο t5-base
            self.pipe = pipeline("text2text-generation", model="t5-base", device=device)
            self.prompt_template = "paraphrase: {text}"

    def __call__(self, text: str, max_new_tokens: int = 128) -> str:
        """Δημιουργεί παραφρασμένο κείμενο με T5.
        Χρησιμοποιούμε μόνο το max_new_tokens για να αποφύγουμε warnings.
        """
        prompt = self.prompt_template.format(text=text)
        out = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.92,
            top_k=50,
        )
        cand = out[0]["generated_text"]
        return cand.strip()


def run_all_pipelines(texts: List[str]) -> Dict[str, List[str]]:
    """Τρέχει και τα τρία pipelines (κανόνες, LanguageTool, T5 paraphrase)
    και επιστρέφει τα αποτελέσματα για κάθε κείμενο.
    """
    cr = CustomRuleRewriter()
    lt = LanguageToolCorrector()
    t5 = T5Paraphraser()

    results = {
        "custom_rules": [cr(t) for t in texts],
        "languagetool": [lt(t) for t in texts],
        "t5_paraphrase": [t5(t) for t in texts],
    }
    return results
