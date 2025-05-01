import nltk
from models.formality.sentence_level_formality import get_sentence_formality, get_nomatch_formality
from models.intent.intent_model import get_intent
from models.sentiment.sentiment_model import get_sentiment
from models.audience.audience_model import get_audience

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
    return get_sentiment(text)

def analyze_intent(text):
   return get_intent(text)    

def analyze_formality(text):
    formality = get_sentence_formality(text)['classification']
    return formality

def get_sentence_formality_match(text, desired_formality):
    return get_nomatch_formality(text, desired_formality)

def analyze_audience(text):
    return get_audience(text)

if __name__ == "__main__":
    # Example usage
    email_text = "I am very happy with the service."
    sentiment = analyze(email_text)
    print(sentiment)
