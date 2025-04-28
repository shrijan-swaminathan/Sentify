import nltk
from nltk.tokenize import sent_tokenize
from models.formality.predict_formality import getformality

# Download once (safe if already downloaded)
nltk.download("punkt_tab", quiet=True)

# Map labels to scores
label_to_score = {"informal": 0, "neutral": 0.5, "formal": 1.0}


def get_sentence_formality(text: str):
    sentences = sent_tokenize(text)
    total_length = sum(len(s) for s in sentences)

    results = []
    weighted_score = 0.0

    for sent in sentences:
        label = getformality(
            sent
        ).lower()  # called pipeline from existing formality code
        score = label_to_score.get(label, 0.5)
        weight = len(sent) / total_length if total_length else 0
        weighted_score += score * weight
        results.append((sent, label, round(score, 2)))

    classification = (
        "formal"
        if weighted_score > 0.75
        else "neutral" if weighted_score > 0.4 else "informal"
    )

    return {
        "sentences": results,
        "weighted_formality_score": round(weighted_score, 3),
        "classification": classification,
    }


def get_nomatch_formality(text: str, desired_formality: str):
    sentences = sent_tokenize(text)
    flagged_sentences = []

    for sent in sentences:
        detected_formality = getformality(sent).lower()
        desired_formality = desired_formality.lower()

        if detected_formality != desired_formality:
            flagged_sentences.append(
                {
                    "sentence": sent,
                    "detected_formality": detected_formality,
                    "issue": f"Sentence formality '{detected_formality}' does not match desired formality '{desired_formality}'.",
                }
            )

    return flagged_sentences


# Example usage
if __name__ == "__main__":
    email_text = """Hey! Just checking — can you shoot over the deck? Appreciate it. Thanks so much!"""
    from pprint import pprint

    pprint(get_sentence_formality(email_text))
