import streamlit as st
from gpt import gpt
from vader_nltk import analyze_sentiment

st.title("Email Sentiment Analysis and Feedback")
input_text = st.text_area("Enter your email content here")
if st.button("Analyze"):
  sentiment = analyze_sentiment(input_text)
  gpt_input = f"SENTIMENT: {sentiment}; TEXT: {input_text}"
  # print("gpt_input: "+ gpt_input)
  feedback = gpt(gpt_input)
  st.write("Sentiment:", sentiment)
  st.write("Feedback:", feedback)