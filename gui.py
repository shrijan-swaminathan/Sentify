# import streamlit as st
# from gpt import gpt
# # from vader_nltk import analyze_sentiment
# from predict import analyse_email

# st.title("Email Sentiment Analysis and Feedback")
# input_text = st.text_area("Enter your email content here")
# intent = st.text_input("Enter your intent here (optional)")
# if st.button("Analyze"):
#   # sentiment = analyze_sentiment(input_text)
#   sentiment = analyse_email(input_text)
#   gpt_input = f"SENTIMENT: {sentiment}; INTENT: {intent}; TEXT: {input_text}"
#   # print("gpt_input: "+ gpt_input)
#   feedback = gpt(gpt_input)
#   st.write("Sentiment:", sentiment)
#   st.write("Feedback:", feedback)

import streamlit as st
from gpt import gpt
from vader_nltk import analyze_sentiment

def chatbot_response(input_text):
    # Call the GPT function with the input text
    sentiment = analyze_sentiment(input_text)
    gpt_input = f"SENTIMENT: {sentiment}; TEXT: {input_text}"
    feedback = gpt(gpt_input)
    return feedback

st.title("Email Sentiment Analysis and Feedback")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_prompt = st.chat_input("Enter your email content here...")

if user_prompt:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Get bot response
    bot_reply = chatbot_response(user_prompt)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)