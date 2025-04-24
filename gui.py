import streamlit as st

st.set_page_config(page_title="Email Assistant", layout="wide")


# Cache resource loading
@st.cache_resource
def load_resources():
    from sentiment_model import analyze_sentiment
    from gpt import gpt, gpt_edit_email

    return analyze_sentiment, gpt, gpt_edit_email


# Load all resources at once
analyze_sentiment, gpt, gpt_edit_email = load_resources()


def chatbot_response(input_text):
    sentiment = analyze_sentiment(input_text)
    return gpt(input_text, sentiment)


def get_edits(input_text, mode="Auto", target=None):
    sentiment = analyze_sentiment(input_text)
    if mode == "Auto":
        return gpt_edit_email(input_text, sentiment)
    return gpt_edit_email(input_text, sentiment, target)


tab_chat, tab_emailassistant = st.tabs(["Chatbot & Feedback", "Email Assistant"])


def initialize_session_state():
    """Initialize all session state variables if they don't exist"""
    default_keys = {
        "persistent_subject": "",
        "persistent_sender": "",
        "persistent_recipient": "",
        "persistent_body": "",
        "sentiment": {},
        "opt_subject": "",
        "opt_salutation": "",
        "opt_closing": "",
        "opt_body": "",
        "mode": "",
        "target_intent": "",
        "target_formality": "",
        "target_audience": "",
        "messages": [],
    }

    for key, default_value in default_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# Initialize session state variables
initialize_session_state()

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

    compose_tab, preview_tab, analysis_tab = st.tabs(["Compose", "Preview", "Analysis"])

    with compose_tab:
        st.header("Compose Email")

        # Mode selection
        current_mode = st.radio(
            "Choose Mode", ["Auto", "Guided"], index=0, horizontal=True
        )

        # Check if the mode changed from previous selection
        if st.session_state.get("prev_mode", "") != current_mode:
            for k in (
                "opt_subject",
                "opt_salutation",
                "opt_closing",
                "opt_body",
                "sentiment",
            ):
                st.session_state[k] = "" if k != "sentiment" else {}
            for raw_k in (
                "opt_subject_raw",
                "opt_salutation_raw",
                "opt_closing_raw",
                "opt_body_raw",
            ):
                if raw_k in st.session_state:
                    del st.session_state[raw_k]

        st.session_state["mode"] = current_mode
        st.session_state["prev_mode"] = current_mode

        # Email input and suggestions layout
        col_inputs, col_suggestions = st.columns([3, 2])

        # Email input form
        with col_inputs:
            st.text_input("Subject", key="persistent_subject")
            st.text_input("From (required)", key="persistent_sender")
            st.text_input("To (required)", key="persistent_recipient")
            st.text_area("Body (required)", key="persistent_body", height=200)
            if current_mode == "Guided":
                with st.expander("Tone Settings", expanded=True):
                    st.markdown("##### Set your desired email tone")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.selectbox(
                            "Intent",
                            ["follow-up", "inform", "request"],
                            index=0,
                            key="target_intent",
                            help="What is the main purpose of your email?",
                        )
                    with col2:
                        st.selectbox(
                            "Formality",
                            ["formal", "neutral", "informal"],
                            key="target_formality",
                            help="How formal should your email be?",
                        )
                    with col3:
                        st.selectbox(
                            "Audience",
                            ["professional", "personal", "general"],
                            key="target_audience",
                            help="Who will be reading your email?",
                        )

            if st.button("Generate suggestions", type="primary"):
                # Validate required fields
                required_fields = {
                    "sender": st.session_state.persistent_sender,
                    "recipient": st.session_state.persistent_recipient,
                    "body": st.session_state.persistent_body,
                }

                missing = [
                    field for field, value in required_fields.items() if not value
                ]

                if missing:
                    st.error(f"Missing field(s): {', '.join(missing)}", icon="🚨")
                else:
                    # Reset suggestion state
                    for k in (
                        "opt_subject",
                        "opt_salutation",
                        "opt_closing",
                        "opt_body",
                    ):
                        st.session_state[k] = ""
                    for raw_k in (
                        "opt_subject_raw",
                        "opt_salutation_raw",
                        "opt_closing_raw",
                        "opt_body_raw",
                    ):
                        if raw_k in st.session_state:
                            del st.session_state[raw_k]

                    # Format email for analysis
                    email_text = "\n\n".join(
                        [
                            f"Subject: {st.session_state.persistent_subject}".strip(),
                            f"From: {st.session_state.persistent_sender}".strip(),
                            st.session_state.persistent_body.strip(),
                            f"To: {st.session_state.persistent_recipient}".strip(),
                        ]
                    )

                    # Generate suggestions
                    with st.spinner("Generating suggestions..."):
                        if current_mode == "Auto":
                            result = get_edits(email_text)
                        else:
                            target = {
                                "intent": st.session_state.target_intent,
                                "formality": st.session_state.target_formality,
                                "audience": st.session_state.target_audience,
                            }
                            result = get_edits(email_text, mode="Guided", target=target)

                        st.session_state.sentiment = result
                        st.rerun()

        # Suggestions panel
        with col_suggestions:
            st.header("Suggestions")

            if st.session_state.sentiment:
                with st.container(border=True):
                    # Subject suggestions
                    opts = st.session_state.sentiment["Subjects"]
                    choice = st.selectbox(
                        "Subject:",
                        ["(keep mine)"] + opts,
                        index=0,
                        key="opt_subject_raw",
                    )
                    st.session_state.opt_subject = (
                        choice
                        if choice != "(keep mine)"
                        else st.session_state.persistent_subject
                    )
                    salutations = st.session_state.sentiment["Salutations"]
                    choice = st.selectbox(
                        "Salutation:",
                        ["(none)"] + salutations,
                        index=0,
                        key="opt_salutation_raw",
                    )
                    st.session_state.opt_salutation = (
                        choice if choice != "(none)" else ""
                    )

                    closings = st.session_state.sentiment["Closings"]
                    choice = st.selectbox(
                        "Closing:",
                        ["(none)"] + closings,
                        index=0,
                        key="opt_closing_raw",
                    )
                    st.session_state.opt_closing = choice if choice != "(none)" else ""
            else:
                st.info(
                    "Click **Generate suggestions** to see suggestions for improving your email."
                )

        # Body suggestions
        bodies = st.session_state.sentiment.get("Bodies", [])
        if bodies:
            display_bodies = bodies.copy()
            display_bodies.append(st.session_state.persistent_body)
            st.markdown("### Body Suggestions")
            # Create tabs
            tabs = st.tabs([f"Option {i+1}" for i in range(len(bodies))] + ["Original"])
            for i, (tab, text) in enumerate(zip(tabs, display_bodies)):
                with tab:
                    if i == len(bodies):  # Check against original bodies length
                        st.markdown("**Original Body**")
                    else:
                        st.markdown(f"**Option {i+1}**")

                    st.text_area(
                        "Suggestion text",
                        text,
                        height=150,
                        disabled=True,
                        key=f"preview_{i}",
                    )

                    # Use a more descriptive button label
                    button_label = f"Select Option {i+1}"

                    if st.button(button_label, key=f"select_{i}", type="secondary"):
                        st.session_state.opt_body = text

    with preview_tab:
        # Preview and Sentiment tabs
        st.markdown("### Email Preview")
        # Prepare preview content
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
        st.code(preview_text, language="text")
        st.download_button(
            "Download Email",
            preview_text,
            file_name="email_preview.txt",
            mime="text/plain",
        )

    with analysis_tab:
        st.markdown("### Email Analysis")

        if st.session_state.get("sentiment"):
            sentiment_data = analyze_sentiment(preview_text)
            sent_cat = sentiment_data["sentiment_category"]
            cat_colors = {"positive": "green", "negative": "red", "neutral": "gray"}

            # Overall sentiment with better visualization
            st.markdown("#### Overall Impression")

            sentiment_col, gauge_col = st.columns([1, 1])
            with sentiment_col:
                st.markdown(
                    f"<h3 style='color:{cat_colors.get(sent_cat, 'black')}'>{sent_cat.capitalize()}</h3>",
                    unsafe_allow_html=True,
                )

            # Sentiment scores with better visualization
            st.markdown("#### Sentiment Breakdown")

            pos = sentiment_data["sentiment_scores"]["pos"]
            neg = sentiment_data["sentiment_scores"]["neg"]
            neu = sentiment_data["sentiment_scores"]["neu"]
            comp = sentiment_data["sentiment_scores"]["compound"]

            score_cols = st.columns(4)
            score_cols[0].metric("Positive", f"{pos:.2f}")
            score_cols[1].metric("Negative", f"{neg:.2f}")
            score_cols[2].metric("Neutral", f"{neu:.2f}")
            score_cols[3].metric("Compound", f"{comp:.2f}")

            # Email characteristics with better visualization
            st.markdown("#### Email Characteristics")

            with st.container(border=True):
                attribute_cols = st.columns(3)

                with attribute_cols[0]:
                    intent = sentiment_data["intent"]
                    st.metric("Intent", intent.capitalize() if intent else "N/A")

                with attribute_cols[1]:
                    formality = sentiment_data["formality"]
                    st.metric(
                        "Formality", formality.capitalize() if formality else "N/A"
                    )

                with attribute_cols[2]:
                    audience = sentiment_data["audience"]
                    st.metric("Audience", audience.capitalize() if audience else "N/A")
        else:
            st.info("Generate suggestions first to see analysis of your email.")
