import openai
import keys

instructions="""
    You are a chatbot expected to provide feedback on how to improve emails given sentiment values across various parameters. 
    The input for you would be formatted in the following way: 'SENTIMENT: x; TEXT: y'
    You are expected to do the following:
    1. Provide feedback on the emails I provide you with on a step by step basis
    2. Make to be as concise and easy to understand as possible. I don't want overly complicated words or paragraph long explanations
"""

speech_key = keys.azure_key
service_region = keys.azure_region
client = openai.AzureOpenAI(azure_endpoint=keys.azure_openai_endpoint, api_key = keys.azure_openai_key, api_version=keys.azure_openai_api_version)
discourse = [{"role": "system", "content":instructions}]

def gpt(text, discourse=discourse):
    discourse.append({"role": "user", "content": text})
    response = client.chat.completions.create(model="gpt-4", messages=discourse)
    reply = response.choices[0].message.content
    return reply


