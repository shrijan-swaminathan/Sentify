import openai
import keys

instructions="""
    You are an email coach assistant that provides constructive feedback on emails.
    
    The input will include sentiment analysis across multiple dimensions:
    1. Basic sentiment (positive/negative/neutral scores and category)
    2. Intent classification (informative, request, persuasive, gratitude, complaint)
    3. Formality level (formal, informal, neutral)
    4. Audience assessment (professional, personal, general)
    
    For each email, provide the following feedback:
    1. Overall tone assessment based on the sentiment scores
    2. Appropriateness of tone for the detected audience and intent
    3. Suggestions for improving effectiveness based on intent
    4. Formality adjustments if needed for the audience
    5. Specific wording or phrasing recommendations
    
    Keep your feedback concise, constructive, and actionable. Focus on 2-3 key improvements that would have the most impact.
"""

speech_key = keys.azure_key
service_region = keys.azure_region
client = openai.AzureOpenAI(azure_endpoint=keys.azure_openai_endpoint, api_key = keys.azure_openai_key, api_version=keys.azure_openai_api_version)
discourse = [{"role": "system", "content":instructions}]

def gpt(text, sentiment_data, discourse=discourse):
    input_format = f"""
Email Text:
------
{text}
------

Sentiment Analysis:
- Sentiment Category: {sentiment_data['sentiment_category']}
- Sentiment Scores: pos={sentiment_data['sentiment_scores']['pos']:.2f}, neg={sentiment_data['sentiment_scores']['neg']:.2f}, neu={sentiment_data['sentiment_scores']['neu']:.2f}, compound={sentiment_data['sentiment_scores']['compound']:.2f}
- Intent: {sentiment_data['intent']}
- Formality: {sentiment_data['formality']}
- Audience: {sentiment_data['audience']}

Please provide specific, actionable feedback to improve this email.
"""
    discourse.append({"role": "user", "content": input_format})
    response = client.chat.completions.create(model="gpt-4", messages=discourse)
    reply = response.choices[0].message.content
    # trim discourse to not be too large
    if len(discourse) > 10:
        discourse.pop(1)
    return reply