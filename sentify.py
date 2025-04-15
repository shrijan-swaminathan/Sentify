from gpt import gpt
from vader_nltk import analyze_sentiment

if __name__ == "__main__":
    test_neutral_email = """
        Subject: Test Email - Just Checking In
        Hi TestRecipient,
        This is just a quick test email to make sure everything is working correctly. Please feel free to ignore this.
        Thanks,
        Person
    """

    