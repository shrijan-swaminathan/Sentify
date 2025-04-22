import streamlit as st
from gpt import gpt, gpt_edit_email
from sentiment_model import analyze_sentiment

def chatbot_response(input_text):
    sentiment = analyze_sentiment(input_text)
    return gpt(input_text, sentiment)


def get_edits(input_text):
    sentiment = analyze_sentiment(input_text)
    sentiment_data = gpt_edit_email(input_text, sentiment)
    return sentiment_data

st.set_page_config(page_title="Email Assistant", layout="wide")

tab_chat, tab_emailassistant = st.tabs(["Chatbot & Feedback", "Email Assistant"])

# === Tab 1: Chatbot & Feedback ===
with tab_chat:
    st.header("Email Sentiment Analysis and Feedback")

    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Enter your email content here…")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        reply = chatbot_response(user_input)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

# === Tab 2: Email Editor ===
with tab_emailassistant:
    # Initialize session state
    for key in (
        "persistent_subject",
        "persistent_sender",
        "persistent_recipient",
        "persistent_body",
        "sentiment",
        "opt_subject",
        "opt_salutation",
        "opt_closing",
        "opt_body",
    ):
        if key not in st.session_state:
            st.session_state[key] = "" if "sentiment" not in key else {}

    # Main layout with 2 columns: Compose & Suggestions
    col_inputs, col_suggestions = st.columns([2, 1])

    with col_inputs:
        st.header("Compose Email")
        st.text_input(
            "Subject",
            key="persistent_subject",
            value=st.session_state.persistent_subject,
        )
        st.text_input(
            "From (required)",
            key="persistent_sender",
            value=st.session_state.persistent_sender,
        )
        st.text_input(
            "To (required)",
            key="persistent_recipient",
            value=st.session_state.persistent_recipient,
        )
        st.text_area(
            "Body (required)",
            key="persistent_body",
            height=200,
            value=st.session_state.persistent_body,
        )

        # If suggestions button is clicked
        if st.button("Generate suggestions", type="primary"):
            to_check = ["persistent_sender", "persistent_recipient", "persistent_body"]
            missing = [
                key.split("_")[1] for key in to_check if not st.session_state.get(key)
            ]

            if missing:
                st.error(f"Missing field(s): {', '.join(missing)}", icon="🚨")
            else:
                # Reset options
                st.session_state.opt_subject = ""
                st.session_state.opt_salutation = ""
                st.session_state.opt_closing = ""
                st.session_state.opt_body = ""

                # Clear raw selections
                for raw_key in (
                    "opt_subject_raw",
                    "opt_salutation_raw",
                    "opt_closing_raw",
                    "opt_body_raw",
                ):
                    if raw_key in st.session_state:
                        del st.session_state[raw_key]

                # Prepare email text for processing
                email_text = "\n\n".join(
                    [
                        f"Subject: {st.session_state.persistent_subject}".strip(),
                        f"From: {st.session_state.persistent_sender}".strip(),
                        st.session_state.persistent_body.strip(),
                        f"To: {st.session_state.persistent_recipient}".strip(),
                    ]
                )

                # Generate sentiment scores and suggestions
                with st.spinner("Generating suggestions..."):
                    st.session_state.sentiment = get_edits(email_text)
                st.session_state.opt_subject = ""
                st.session_state.opt_salutation = ""
                st.session_state.opt_closing = ""
                st.rerun()

    with col_suggestions:
        st.header("Subject/Greeting/Closing Suggestions")

        if st.session_state.sentiment:
            # Subject suggestions
            opts = st.session_state.sentiment["Subjects"]
            choice = st.selectbox(
                "Subject:",
                ["(keep mine)"] + opts,
                index=0,
                key="opt_subject_raw",
            )
            if choice != "(keep mine)":
                st.session_state.opt_subject = choice
            else:
                st.session_state.opt_subject = st.session_state.persistent_subject

            # Salutation suggestions
            salutations = st.session_state.sentiment["Salutations"]
            choice = st.selectbox(
                "Salutation:",
                ["(none)"] + salutations,
                index=0,
                key="opt_salutation_raw",
            )
            if choice != "(none)":
                st.session_state.opt_salutation = choice
            else:
                st.session_state.opt_salutation = ""

            # Closing suggestions
            closings = st.session_state.sentiment["Closings"]
            choice = st.selectbox(
                "Closing:",
                ["(none)"] + closings,
                index=0,
                key="opt_closing_raw",
            )
            if choice != "(none)":
                st.session_state.opt_closing = choice
            else:
                st.session_state.opt_closing = ""
        else:
            st.info("Click **Generate suggestions** to see options.")
    
    st.subheader("📄 Body Suggestions")
    bodies = st.session_state.sentiment["Bodies"] if st.session_state.sentiment else []

    if bodies:
        ## append the original body to the list of suggestions
        bodies.append(st.session_state.persistent_body)
        cols = st.columns(4)
        for i, (col, text) in enumerate(zip(cols, bodies)):
            with col:
                if i == len(bodies) - 1:
                    st.markdown("**Original Body**")
                else: 
                    st.markdown(f"**Option {i+1}**")
                st.text_area("Suggestion text", text, height=150, disabled=True, key=f"preview_{i}")
                if st.button(
                    f"Select Option {i+1}", key=f"select_{i}", type="secondary"
                ):
                    st.session_state.opt_body = text
    else:
        st.info("Generate suggestions to see body options.")

    preview_tab, scores_tab = st.tabs(["Email Preview", "Sentiment Analysis"])

    # Prepare the preview content
    subject = st.session_state.opt_subject or st.session_state.persistent_subject
    salutation = st.session_state.opt_salutation
    body = st.session_state.opt_body or st.session_state.persistent_body
    closing = st.session_state.opt_closing
    sender = st.session_state.persistent_sender

    preview_lines = []
    if subject:
        preview_lines.append(f"Subject: {subject}")
    if salutation:
        preview_lines.append(salutation)
    preview_lines.append(body)
    if closing:
        preview_lines.append(closing)
    preview_lines.append(sender)

    preview_text = "\n\n".join(preview_lines)

    # Preview tab content
    with preview_tab:
        st.subheader("Final Email Preview")
        st.code(preview_text, language="text")

    # Sentiment analysis tab content
    with scores_tab:
        if st.session_state.sentiment:
            sentiment_data = analyze_sentiment(preview_text)

            # Overall sentiment with visual indicator
            sent_cat = sentiment_data["sentiment_category"]
            cat_colors = {"positive": "green", "negative": "red", "neutral": "gray"}

            st.subheader("Overall Sentiment")
            sentiment_col, gauge_col = st.columns([1, 1])

            with sentiment_col:
                st.markdown(
                    f"### <span style='color:{cat_colors.get(sent_cat, 'black')}'>{sent_cat.capitalize()}</span>",
                    unsafe_allow_html=True,
                )

            # Sentiment scores in a more visual way
            st.subheader("Sentiment Breakdown")
            pos = sentiment_data["sentiment_scores"]["pos"]
            neg = sentiment_data["sentiment_scores"]["neg"]
            neu = sentiment_data["sentiment_scores"]["neu"]
            comp = sentiment_data["sentiment_scores"]["compound"]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Positive", f"{pos:.2f}")
            col2.metric("Negative", f"{neg:.2f}")
            col3.metric("Neutral", f"{neu:.2f}")
            col4.metric("Compound", f"{comp:.2f}")

            # Message attributes in a more organized way
            st.subheader("Email Characteristics")

            attributes_col1, attributes_col2, attributes_col3 = st.columns(3)

            with attributes_col1:
                intent = sentiment_data["intent"]
                st.metric("Intent", intent.capitalize() if intent else "N/A")

            with attributes_col2:
                formality = sentiment_data["formality"]
                st.metric("Formality", formality.capitalize() if formality else "N/A")

            with attributes_col3:
                audience = sentiment_data["audience"]
                st.metric("Audience", audience.capitalize() if audience else "N/A")
