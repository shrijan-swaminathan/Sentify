import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from models.formality.sentence_level_formality import get_sentence_formality, get_nomatch_formality
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Download VADER lexicon if not already downloaded
nltk.download("vader_lexicon")

def analyze(text):
    # get sentiment, intent, formality, and audience
    sentiment, sentiment_category = analyze_sentiment(text)
    intent, i_confidence = analyze_intent(text)
    formality = analyze_formality(text)
    audience, a_confidence = analyze_audience(text)
    result = {
        "sentiment_scores": sentiment,
        "sentiment_category": sentiment_category,
        "intent": intent,
        "intent_confidence": i_confidence,
        "formality": formality,
        "audience": audience,
        "audience_confidence": a_confidence,
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
        sentiment_category = "neutral"
    return sentiment, sentiment_category


def analyze_intent(text):
    tokenizer = AutoTokenizer.from_pretrained("parvk11/intent_classification_model")
    model = AutoModelForSequenceClassification.from_pretrained("parvk11/intent_classification_model")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()
    reverse_label_map = {0: 'follow-up', 1: 'request', 2: 'inform'}
    return reverse_label_map[pred], confidence


def analyze_formality(text):
    formality = get_sentence_formality(text)['classification']
    return formality

def get_sentence_formality_match(text, desired_formality):
    return get_nomatch_formality(text, desired_formality)

def analyze_audience(text):
    tokenizer = AutoTokenizer.from_pretrained("parvk11/audience_classifier_model")
    model = AutoModelForSequenceClassification.from_pretrained("parvk11/audience_classifier_model")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()
    reverse_label_map = {0: 'professional', 1: 'personal', 2: 'general'}
    return reverse_label_map[pred], confidence


if __name__ == "__main__":
    # Example usage
    email_text = "I am very happy with the service."
    sentiment = analyze(email_text)
    print(sentiment)
