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