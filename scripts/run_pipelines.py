
from __future__ import annotations
import json
from pathlib import Path
from nlp_reconstruction.data import TEXT1, TEXT2
from nlp_reconstruction.preprocess import pick_two_sentences
from nlp_reconstruction.pipelines import CustomRuleRewriter, LanguageToolCorrector, T5Paraphraser

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True, parents=True)

def main():
    texts = [TEXT1, TEXT2]
   # Μέρος Α: επιλέξτε δύο προτάσεις (μία από κάθε κείμενο για ποικιλία)
    s1 = pick_two_sentences(TEXT1)[0]
    s2 = pick_two_sentences(TEXT2)[0]

    custom = CustomRuleRewriter()
    lt = LanguageToolCorrector()
    t5 = T5Paraphraser()

    recon = {
        "partA": {
            "input": [s1, s2],
            "custom_rules": [custom(s1), custom(s2)],
            "languagetool": [lt(s1), lt(s2)],
            "t5_paraphrase": [t5(s1), t5(s2)],
        },
        "partB": {
            "input": [TEXT1, TEXT2],
            "custom_rules": [custom(TEXT1), custom(TEXT2)],
            "languagetool": [lt(TEXT1), lt(TEXT2)],
            "t5_paraphrase": [t5(TEXT1), t5(TEXT2)],
        },
    }

    with open(OUT_DIR / "reconstructions.json", "w", encoding="utf-8") as f:
        json.dump(recon, f, ensure_ascii=False, indent=2)

    # αποθηκεύστε επίσης ένα αρχείο txt για γρήγορη προβολή
    with open(OUT_DIR / "reconstructed_texts.txt", "w", encoding="utf-8") as f:
        for k in ("custom_rules", "languagetool", "t5_paraphrase"):
            f.write(f"==== {k.upper()} ====\n\n")
            f.write("TEXT1:\n")
            f.write(recon["partB"][k][0] + "\n\n")
            f.write("TEXT2:\n")
            f.write(recon["partB"][k][1] + "\n\n")

    print("Wrote outputs to", OUT_DIR)

if __name__ == "__main__":
    main()
