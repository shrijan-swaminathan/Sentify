import streamlit as st

st.set_page_config(page_title="Email Assistant", layout="wide")


# Cache resource loading
@st.cache_resource
def load_resources():
    from models.models import analyze, get_sentence_formality_match
    from nltk.tokenize import sent_tokenize
    from models.gpt import gpt_feedback, gpt_generate_and_analyze, gpt_edit_email

    return (
        analyze,
        get_sentence_formality_match,
        gpt_feedback,
        gpt_generate_and_analyze,
        gpt_edit_email,
        sent_tokenize,
    )


# Load all resources at once
(
    analyze,
    get_sentence_formality_match,
    gpt_feedback,
    gpt_generate_and_analyze,
    gpt_edit_email,
    sent_tokenize,
) = load_resources()


def chatbot_response(input_text):
    sentiment = analyze(input_text)
    return gpt_feedback(input_text, sentiment)


def generate_email_response(input_text):
    return gpt_generate_and_analyze(input_text, analyze)


def get_edits(input_text, mode="Auto", target=None):
    sentiment = analyze(input_text)
    if mode == "Auto":
        return gpt_edit_email(input_text, sentiment)
    return gpt_edit_email(input_text, sentiment, target)


def check_formality(input_text, target_formality):
    flagged_sentences = get_sentence_formality_match(input_text, target_formality)
    return flagged_sentences


tab_chat, tab_emailassistant, tab_formality = st.tabs(
    ["Chatbot & Feedback", "Email Assistant", "Formality Alignment Check"]
)


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
        "target_polarity": "",
        "chatbot_target_intent": "",
        "chatbot_target_formality": "",
        "chatbot_target_audience": "",
        "chatbot_target_polarity": "",
        "formality_target": "Neutral",
        "formality_analysis_result": {},
        "formality_email_text": {},
        "generated_emails": [],
        "messages": [],
    }

    for key, default_value in default_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# Initialize session state variables
initialize_session_state()

# === Tab 1: Chatbot & Feedback ===
with tab_chat:
    st.header("Chat-based Email Revision Assistant")

    # Confirmation Dialog to clear chat
    @st.dialog("Clear Chat")
    def confirm_clear_chat():
        st.warning(
            "Are you sure you want to clear the chat? This cannot be undone.", icon="⚠️"
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Cancel", key="cancel_clear"):
                st.session_state.show_clear_dialog = False
                st.rerun()
        with col2:
            if st.button("Yes, clear chat", key="confirm_clear"):
                st.session_state.messages = []
                st.session_state.latest_email = ""
                st.session_state.latest_analysis = {}
                st.session_state.show_clear_dialog = False
                st.rerun()

    # Radio button to choose mode
    mode = st.radio("Select Mode", ["Generate Email", "Feedback Only"], horizontal=True)
    if mode == "Generate Email":
        tone_mode = current_mode = st.radio(
            "Choose Tone Mode", ["Auto", "Guided"], index=0, horizontal=True
        )

    # Split screen: chat on left, preview/analysis on right
    col_chat, col_email = st.columns([2, 3], gap="large")

    with col_chat:
        with st.container(height=400, border=True):
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if mode == "Generate Email":
            user_input = st.chat_input("Type your prompt...")
            if tone_mode == "Guided":
                with st.expander("Tone Settings", expanded=True):
                    st.markdown("##### Set your desired email tone")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.selectbox(
                            "Intent",
                            ["follow-up", "inform", "request"],
                            index=0,
                            key="chatbot_target_intent",
                            help="What is the main purpose of your email?",
                        )
                    with col2:
                        st.selectbox(
                            "Formality",
                            ["formal", "neutral", "informal"],
                            key="chatbot_target_formality",
                            help="How formal should your email be?",
                        )
                    with col3:
                        st.selectbox(
                            "Audience",
                            ["professional", "personal", "general"],
                            key="chatbot_target_audience",
                            help="Who will be reading your email?",
                        )
                    with col4:
                        st.selectbox(
                            "Polarity",
                            ["positive", "negative", "neutral"],
                            key="chatbot_target_polarity",
                            help="How positive or negative do you want the response to be?",
                        )
        else:
            user_input = st.chat_input("Type your email content...")

        if user_input:
            with st.spinner("Thinking..."):
                st.session_state.messages.append(
                    {"role": "user", "content": user_input}
                )

                if mode == "Generate Email":
                    targets = {}
                    if tone_mode == "Guided":
                        targets = {
                            "intent": st.session_state.chatbot_target_intent,
                            "formality": st.session_state.chatbot_target_formality,
                            "audience": st.session_state.chatbot_target_audience,
                            "polarity": st.session_state.chatbot_target_polarity,
                        }

                    generated_email, sentiment_data = gpt_generate_and_analyze(
                        user_input, analyze, targets
                    )
                    st.session_state.latest_email = generated_email
                    st.session_state.latest_analysis = sentiment_data
                    st.session_state.messages.append(
                        {"role": "assistant", "content": generated_email}
                    )
                    st.session_state.generated_emails.append(generated_email)
                else:  # Feedback Only
                    sentiment_data = analyze(user_input)
                    st.session_state.latest_email = user_input
                    st.session_state.latest_analysis = sentiment_data
                    feedback = gpt_feedback(user_input, sentiment_data)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": feedback}
                    )
                st.rerun()

        if st.button("Clear Chat 🗑️", use_container_width=True):
            st.session_state.show_clear_dialog = True
        if st.session_state.get("show_clear_dialog", False):
            confirm_clear_chat()

    with col_email:
        st.subheader("Generated Email")
        email_tab, analysis_tab = st.tabs(["Email", "Analysis"])
        if st.session_state.get("latest_email", ""):
            with email_tab:
                if mode == "Feedback Only":
                    st.markdown("**Generated Email is Same as Input Email**")
                st.code(st.session_state.latest_email, language="text")
                with st.expander("Previously Generated Emails", expanded=False):
                    # Scrollable area
                    with st.container():
                        for idx, email in enumerate(st.session_state.generated_emails):
                            st.code(email, language="text")
            with analysis_tab:
                sentiment_data = st.session_state.latest_analysis
                sent_cat = sentiment_data["sentiment_category"]
                cat_colors = {"positive": "green", "negative": "red", "neutral": "gray"}

                st.markdown("#### Overall Impression")
                st.markdown(
                    f"<h3 style='color:{cat_colors.get(sent_cat, 'black')}'>{sent_cat.capitalize()}</h3>",
                    unsafe_allow_html=True,
                )

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

                st.markdown("#### Email Characteristics")
                attribute_cols = st.columns(3)
                attribute_cols[0].metric(
                    "Intent",
                    (
                        sentiment_data["intent"].capitalize()
                        if sentiment_data["intent"]
                        else "N/A"
                    ),
                )
                attribute_cols[0].markdown(
                    f"**Confidence:** {sentiment_data.get('intent_confidence', 0):.2%}"
                )
                attribute_cols[1].metric(
                    "Formality",
                    (
                        sentiment_data["formality"].capitalize()
                        if sentiment_data["formality"]
                        else "N/A"
                    ),
                )
                attribute_cols[2].metric(
                    "Audience",
                    (
                        sentiment_data["audience"].capitalize()
                        if sentiment_data["audience"]
                        else "N/A"
                    ),
                )
                attribute_cols[2].markdown(
                    f"**Confidence:** {sentiment_data.get('audience_confidence', 0):.2%}"
                )
        else:
            st.info("No email content yet. Use the chat to get started.")

# === Tab 2: Email Editor ===
with tab_emailassistant:
    compose_tab, preview_tab = st.tabs(["Compose", "Preview + Analysis"])
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
                    col1, col2, col3, col4 = st.columns(4)

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
                    with col4:
                        st.selectbox(
                            "Polarity",
                            ["positive", "negative", "neutral"],
                            key="target_polarity",
                            help="How positive or negative do you want the response to be?",
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
                                "polarity": st.session_state.target_polarity
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
                bodies = st.session_state.sentiment.get("Bodies", [])
                if bodies:
                    display_bodies = bodies.copy()
                    display_bodies.append(st.session_state.persistent_body)
                    st.markdown("### Body Suggestions")
                    # Create tabs
                    tabs = st.tabs(
                        [f"Option {i+1}" for i in range(len(bodies))] + ["Original"]
                    )
                    for i, (tab, text) in enumerate(zip(tabs, display_bodies)):
                        with tab:
                            if i == len(bodies):  # Check against original bodies length
                                st.markdown("**Original Body**")
                            else:
                                st.markdown(f"**Option {i+1}**")

                            st.text_area(
                                "Suggested Body",
                                text,
                                height=150,
                                disabled=True,
                                key=f"preview_{i}",
                            )
                            # Use a more descriptive button label
                            button_label = f"Select Option {i+1}"

                            if st.button(
                                button_label, key=f"select_{i}", type="secondary"
                            ):
                                st.session_state.opt_body = text
            else:
                st.info(
                    "Click **Generate suggestions** to see suggestions for improving your email."
                )
        # Body suggestions

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
        st.markdown("### Email Analysis")
        if st.session_state.get("sentiment"):
            sentiment_data = analyze(preview_text)
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
                    st.markdown(
                        f"**Confidence:** {sentiment_data.get('intent_confidence', 0):.2%}"
                    )
                with attribute_cols[1]:
                    formality = sentiment_data["formality"]
                    st.metric(
                        "Formality", formality.capitalize() if formality else "N/A"
                    )
                with attribute_cols[2]:
                    audience = sentiment_data["audience"]
                    st.metric("Audience", audience.capitalize() if audience else "N/A")
                    st.markdown(
                        f"**Confidence:** {sentiment_data.get('audience_confidence', 0):.2%}"
                    )
        else:
            st.info("Generate suggestions first to see analysis of your email.")

# line by line formality checker
with tab_formality:
    st.header("Formality Alignment Check")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Email")
        salutation = st.text_input("Salutation", key="salutation_input")
        email_input = st.text_area("Paste your email here:", height=200)
        closing = st.text_input("Closing", key="closing_input")
        st.selectbox(
            "Select Desired Formality",
            ["Formal", "Neutral", "Informal"],
            index=1,
            key="formality_target",
            help="Select the desired formality level for the email.",
        )
        if st.button("Check Formality", type="primary"):
            if email_input or salutation or closing:
                with st.spinner("Analyzing formality..."):
                    # Analyze each section separately
                    flagged_body = check_formality(
                        email_input, st.session_state.formality_target.lower()
                    )
                    flagged_salutation = check_formality(
                        salutation, st.session_state.formality_target.lower()
                    )
                    flagged_closing = check_formality(
                        closing, st.session_state.formality_target.lower()
                    )
                    st.session_state.formality_analysis_result = {
                        "salutation": flagged_salutation,
                        "body": flagged_body,
                        "closing": flagged_closing,
                    }
                    st.session_state.formality_email_text = {
                        "salutation": salutation,
                        "body": email_input,
                        "closing": closing,
                    }
            else:
                st.error(
                    "Please enter at least one field to check formality.", icon="🚨"
                )

    with col2:
        st.subheader("Formality Issues Underlined")
        if (
            "formality_analysis_result" in st.session_state
            and "formality_email_text" in st.session_state
        ):
            original_text = st.session_state.formality_email_text
            flagged_info = st.session_state.formality_analysis_result

            highlighted_text = ""

            # ---- Handle Salutation ----
            salutation_text = original_text.get("salutation", "").strip()
            flagged_salutation = flagged_info.get("salutation", [])

            if salutation_text:
                if flagged_salutation:
                    detected_formality = flagged_salutation[0]["detected_formality"]
                    highlighted_text += f"<u style='color: red'><span style='color:white' title='Detected: {detected_formality.capitalize()}'>{salutation_text}</span></u><br><br>"
                else:
                    highlighted_text += f"{salutation_text}<br><br>"

            # ---- Handle Body ----
            body_text = original_text.get("body", "").strip()
            flagged_body = flagged_info.get("body", [])

            paragraphs = body_text.split("\n\n")

            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                sentences = sent_tokenize(paragraph)

                for sent in sentences:
                    flagged = False
                    detected_formality = ""

                    for item in flagged_body:
                        if sent.strip() == item["sentence"].strip():
                            flagged = True
                            detected_formality = item["detected_formality"]
                            break

                    if flagged:
                        highlighted_text += f"<u style='color: red'><span style='color: white' title='Detected: {detected_formality.capitalize()}'>{sent}</span></u> "
                    else:
                        highlighted_text += f"{sent} "
                highlighted_text += "<br><br>"  # after paragraph
            closing_text = original_text.get("closing", "").strip()
            flagged_closing = flagged_info.get("closing", [])

            if closing_text:
                if flagged_closing:
                    detected_formality = flagged_closing[0]["detected_formality"]
                    highlighted_text += f"<u style='color: red'><span style='color: white' title='Detected: {detected_formality.capitalize()}'>{closing_text}</span></u>"
                else:
                    highlighted_text += f"{closing_text}"

            # Final render
            st.markdown(highlighted_text, unsafe_allow_html=True)

        else:
            st.info(
                "Paste your email and click 'Check Formality' to see highlighted results."
            )
