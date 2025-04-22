# # import streamlit as st
# # from gpt import gpt
# # # from vader_nltk import analyze_sentiment
# # from predict import analyse_email

# # st.title("Email Sentiment Analysis and Feedback")
# # input_text = st.text_area("Enter your email content here")
# # intent = st.text_input("Enter your intent here (optional)")
# # if st.button("Analyze"):
# #   # sentiment = analyze_sentiment(input_text)
# #   sentiment = analyse_email(input_text)
# #   gpt_input = f"SENTIMENT: {sentiment}; INTENT: {intent}; TEXT: {input_text}"
# #   # print("gpt_input: "+ gpt_input)
# #   feedback = gpt(gpt_input)
# #   st.write("Sentiment:", sentiment)
# #   st.write("Feedback:", feedback)

# import streamlit as st
# from gpt import gpt
# from vader_nltk import analyze_sentiment

# def chatbot_response(input_text):
#     # Call the GPT function with the input text
#     sentiment = analyze_sentiment(input_text)
#     gpt_input = f"SENTIMENT: {sentiment}; TEXT: {input_text}"
#     feedback = gpt(gpt_input)
#     return feedback

# st.title("Email Sentiment Analysis and Feedback")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# user_prompt = st.chat_input("Enter your email content here...")

# if user_prompt:
#     # Save user message
#     st.session_state.messages.append({"role": "user", "content": user_prompt})
#     with st.chat_message("user"):
#         st.markdown(user_prompt)

#     # Get bot response
#     bot_reply = chatbot_response(user_prompt)
#     st.session_state.messages.append({"role": "assistant", "content": bot_reply})
#     with st.chat_message("assistant"):
#         st.markdown(bot_reply)

import streamlit as st
from gpt import gpt
from sentiment_model import analyze_sentiment

# Set page config to cover the screen fully
st.set_page_config(
    page_title="Sentify - Email Analysis and Feedback",
    page_icon="@",
    layout="wide"
)

st.title("Sentify - Email Sentiment Analysis and Feedback")
st.markdown("""
This tool analyzes your email content for sentiment, intent, and formality. 
With this information, it provides AI-generated feedback to help improve your message.
""")
# col1, col2 = st.columns(2)
# # first column for inserting text
# with col1:
#     st.subheader("Enter Your Email")
#     email_subject = st.text_input("Subject (optional)")
#     email_content = st.text_area("Email Content", height=250, placeholder="Type or paste your email content here")
#     analyze_button=st.button("Analyze Email")
# # second column for feedback
# with col2:
#     st.subheader("Analysis Results")
#     if analyze_button:
#         if not email_content:# check if there is email content
#             st.info("Enter your email content and click 'Analyze Email' to see results.")
#         with st.spinner("Analyzing email sentiment..."):
#           full_text = f"Subject: {email_subject}\n\n{email_content}" if email_subject else email_content
#           sentiment_results = analyze_sentiment(full_text)
#           feedback = gpt(full_text, sentiment_results)
#           st.write()

input_text = st.text_area("Enter your email content here")
if st.button("Analyze"):
  sentiment = analyze_sentiment(input_text)
  feedback = gpt(input_text, sentiment)
  st.write("Sentiment:", sentiment)
  st.write("Feedback:", feedback)