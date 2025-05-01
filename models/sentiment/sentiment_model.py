import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

def get_sentiment(text):
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