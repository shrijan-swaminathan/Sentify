import openai
import keys
import json
from json_repair import repair_json

feedback_instructions = """
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
gen_email_instructions = """
    You are an email generator. Write a complete, professional email based on the user's request.
    Focus on clarity, appropriate tone, and effectiveness.
"""

edit_email_instructions = (
    "You are an email‑editing assistant. Edit the email draft based on the metrics provided. "
    "Generate exactly 4 subject lines, 3 salutations, 3 closings, and 3 updated bodies. "
    "You'll operate in one of two modes:"
    "\n- Auto Mode: When only detected metrics are provided, use them as a baseline and make improvements you deem appropriate."
    "\n- Guided Mode: When both detected and target metrics are provided, transform the email to match the target metrics."
    "\nIn both modes, maintain the core message while adapting style and tone as needed."
)

functions = [
    {
        "name": "edit_email",
        "description": "Given an email, suggest exactly 4 subject lines, 3 salutations, 3 closings, and return an edited body.",
        "parameters": {
            "type": "object",
            "properties": {
                "Subjects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "Exactly four recommended subject lines",
                },
                "Salutations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Exactly three recommended salutations",
                },
                "Closings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Exactly three recommended closings",
                },
                "Bodies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Exactly three recommended bodies",
                },
            },
            "required": ["Subjects", "Salutations", "Closings", "Bodies"],
        },
    }
]

speech_key = keys.azure_key
service_region = keys.azure_region
client = openai.AzureOpenAI(
    azure_endpoint=keys.azure_openai_endpoint,
    api_key=keys.azure_openai_key,
    api_version=keys.azure_openai_api_version,
)
feedback_discourse = [{"role": "system", "content": feedback_instructions}]
generation_discourse = [{"role": "system", "content": gen_email_instructions}]

def gpt_feedback(text, sentiment_data, discourse=feedback_discourse):
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

def gpt_generate_and_analyze(text, analyze_function, discourse=generation_discourse):
    discourse.append({"role": "user", "content": text})
    response = client.chat.completions.create(model="gpt-4o", messages = discourse)
    generated_email = response.choices[0].message.content.strip()
    sentiment_data = analyze_function(generated_email)
    return generated_email, sentiment_data

def gpt_edit_email(text, detected_sentiment_data, target_sentiment_data=None):
    messages = [
        {"role": "system", "content": edit_email_instructions},
    ]

    # Common email information for both modes
    email_info = f"""
        Original Email:
        {text}

        Detected Baseline Metrics (current state of the email):
        - Sentiment: {detected_sentiment_data['sentiment_category']} (pos={detected_sentiment_data['sentiment_scores']['pos']:.2f}, neg={detected_sentiment_data['sentiment_scores']['neg']:.2f}, neu={detected_sentiment_data['sentiment_scores']['neu']:.2f}, compound={detected_sentiment_data['sentiment_scores']['compound']:.2f})
        - Intent: {detected_sentiment_data['intent']}
        - Formality: {detected_sentiment_data['formality']}
        - Audience: {detected_sentiment_data['audience']}
        """

    # Determine mode: Auto or Guided
    if target_sentiment_data is None:
        # Auto Mode: use detected sentiment only
        user_prompt = f"""
            You are an intelligent email-editing assistant. Please analyze and improve the email below.

            {email_info}

            Since no target metrics were specified, you're in AUTO MODE. Please:
            1. Use the detected metrics ONLY as a baseline understanding of the email's current state
            2. Make appropriate improvements to enhance clarity, tone, and effectiveness
            3. Maintain the original message's core intent
            4. Explain the key changes you made and why
            """
    else:
        # Guided Mode: Use user-specified target
        user_prompt = f"""
            You are an intelligent email-editing assistant. Please revise the email below according to the user's specifications.

            {email_info}

            Target Metrics (User-Specified):
            - Intent: {target_sentiment_data['intent']}
            - Formality: {target_sentiment_data['formality']}
            - Audience: {target_sentiment_data['audience']}

            You're in GUIDED MODE. Please:
            1. Use the detected metrics ONLY as a baseline to understand the email's current state
            2. Transform the email to match the user's specified target metrics
            3. Focus on shifting the intent, formality, and audience-appropriateness as requested
            4. Maintain the core message while adapting tone and style
            5. Explain the specific changes you made to achieve the requested metrics
        """

    messages.append({"role": "user", "content": user_prompt})
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        functions=functions,
        function_call={"name": "edit_email"},
    )
    fn_call = response.choices[0].message.function_call
    try:
        output = json.loads(fn_call.arguments)
    except json.JSONDecodeError:
        output = repair_json(fn_call.arguments)
    return output
