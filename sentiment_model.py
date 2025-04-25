import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

def analyze_sentiment(text):
    # initial vader sentiment
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    
    # determine overall sentiment
    if sentiment['compound'] >= 0.05:
        sentiment_category="positive"
    elif sentiment['compound'] <= -0.05:
        sentiment_category="negative"
    else:
        sentiment_category="negative"

    # get intent, formality, and audience
    intent = analyze_intent(text)
    formality = analyze_formality(text)
    audience = analyze_audience(text)
    result = {
        "sentiment_scores": sentiment,
        "sentiment_category": sentiment_category,
        "intent": intent,
        "formality": formality,
        "audience": audience
    }
    return result

def analyze_intent(text):
    # classifier = pipeline("zero-shot-classification")
    # labels = ["inform", "request", "follow-up"]
    # result = classifier(text, labels)
    # intent = result['labels'][0]
    # return intent
    tokenizer = AutoTokenizer.from_pretrained("parvk11/intent_classification_model")
    model = AutoModelForSequenceClassification.from_pretrained("parvk11/intent_classification_model")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    reverse_label_map = {0: 'follow-up', 1: 'request', 2: 'inform'}
    return reverse_label_map[pred]
    
def analyze_formality(text):
    classifier = pipeline("zero-shot-classification")
    labels = ["formal", "unformal", "neutral"]
    result = classifier(text, labels)
    formality = result['labels'][0]
    return formality
def analyze_audience(text):
    # classifier = pipeline("zero-shot-classification")
    # labels = ["professional", "personal", "general"]
    # result = classifier(text, labels)
    # audience = result['labels'][0]
    # return audience
    tokenizer = AutoTokenizer.from_pretrained("parvk11/audience_classifier_model")
    model = AutoModelForSequenceClassification.from_pretrained("parvk11/audience_classifier_model")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    reverse_label_map = {0: 'professional', 1: 'personal', 2: 'general'}
    return reverse_label_map[pred]
if __name__ == "__main__":
    # Example usage
    email_text = "I am very happy with the service."
    sentiment = analyze_sentiment(email_text)
    print(sentiment)