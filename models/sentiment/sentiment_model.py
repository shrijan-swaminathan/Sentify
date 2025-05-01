from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_sentiment(text):
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