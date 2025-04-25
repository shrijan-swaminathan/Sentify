import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from models.formality.sentence_level_formality import get_sentence_formality

# Download VADER lexicon if not already downloaded
nltk.download("vader_lexicon")
ZSC_MODEL = "facebook/bart-large-mnli"
_intent_clf = pipeline("zero-shot-classification", model=ZSC_MODEL)
_audience_clf = pipeline("zero-shot-classification", model=ZSC_MODEL)

def analyze(text):
    # get sentiment, intent, formality, and audience
    sentiment, sentiment_category = analyze_sentiment(text)
    intent = analyze_intent(text)
    formality = analyze_formality(text)
    audience = analyze_audience(text)
    result = {
        "sentiment_scores": sentiment,
        "sentiment_category": sentiment_category,
        "intent": intent,
        "formality": formality,
        "audience": audience,
    }
    return result

def analyze_sentiment(text):
    # initial vader sentiment
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)

    # determine overall sentiment
    if sentiment["compound"] >= 0.05:
        sentiment_category = "positive"
    elif sentiment["compound"] <= -0.05:
        sentiment_category = "negative"
    else:
        sentiment_category = "negative"
    return sentiment, sentiment_category


def analyze_intent(text):
    labels = ["inform", "request", "follow-up"]
    out = _intent_clf([text], candidate_labels=labels, batch_size=1, multi_label=False)[0]
    return out["labels"][0]
    # return None


def analyze_formality(text):
    formality = get_sentence_formality(text)['classification']
    return formality


def analyze_audience(text):
    labels = ["professional", "personal", "general"]
    out = _audience_clf(
        [text], candidate_labels=labels, batch_size=1, multi_label=False
    )[0]
    return out["labels"][0]
    # return None


if __name__ == "__main__":
    # Example usage
    email_text = "I am very happy with the service."
    sentiment = analyze_sentiment(email_text)
    print(sentiment)
