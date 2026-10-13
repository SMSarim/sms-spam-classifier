import string
import pickle

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

AUTHOR = "Syed Muhammad Sarim"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=f"Spam Inspector | {AUTHOR}",
    page_icon="📮",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens
#   paper  #EDF1F5  cool light page background
#   ink    #10262F  headings, sidebar
#   teal   #0F5C6E  primary action
#   alarm  #C23B2E  spam verdict
#   clear  #1F7A55  safe verdict
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Manrope:wght@400;500;600;700&display=swap');

:root {
    --paper: #EDF1F5;
    --card: #FFFFFF;
    --ink: #10262F;
    --muted: #5B6B75;
    --line: #D3DCE3;
    --teal: #0F5C6E;
    --teal-dark: #0B4655;
    --alarm: #C23B2E;
    --alarm-soft: #FBECEA;
    --clear: #1F7A55;
    --clear-soft: #E6F4EC;
}

html, body, .stApp {
    background-color: var(--paper);
    color: var(--ink);
    font-family: 'Manrope', sans-serif;
}

/* Keep the header (it holds the sidebar toggle) but make it blend in */
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { visibility: hidden; }

/* Sidebar reopen button, visible on the light page */
[data-testid="stExpandSidebarButton"],
[data-testid="collapsedControl"] {
    visibility: visible !important;
    color: var(--ink) !important;
}
[data-testid="stExpandSidebarButton"] *,
[data-testid="collapsedControl"] * {
    color: var(--ink) !important;
    fill: var(--ink) !important;
}

.block-container { padding-top: 2.5rem; max-width: 760px; }

/* ---------- Masthead ---------- */
.masthead { margin-bottom: 1.6rem; }
.masthead h1 {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 2.7rem;
    line-height: 1.1;
    letter-spacing: -0.01em;
    color: var(--ink);
    margin: 0 0 0.6rem 0;
}
.masthead p {
    color: var(--muted);
    font-size: 1.05rem;
    max-width: 52ch;
    margin: 0;
}

/* ---------- Input ---------- */
.stTextArea label p {
    font-weight: 600;
    color: var(--ink);
    font-size: 0.95rem;
}
.stTextArea textarea {
    background-color: var(--card) !important;
    color: var(--ink) !important;
    border: 1.5px solid var(--line) !important;
    border-radius: 10px !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.55 !important;
    padding: 0.9rem 1rem !important;
}
.stTextArea textarea:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px rgba(15, 92, 110, 0.18) !important;
}

/* ---------- Buttons ---------- */
.stButton button {
    font-family: 'Manrope', sans-serif;
    font-weight: 700;
    border-radius: 10px;
    padding: 0.65rem 1.2rem;
    width: 100%;
    transition: background-color 0.15s ease, border-color 0.15s ease;
}
.stButton button[kind="primary"] {
    background: var(--teal);
    border: 1px solid var(--teal);
    color: #FFFFFF;
}
.stButton button[kind="primary"]:hover {
    background: var(--teal-dark);
    border-color: var(--teal-dark);
    color: #FFFFFF;
}
.stButton button[kind="secondary"] {
    background: transparent;
    border: 1.5px solid var(--line);
    color: var(--ink);
    font-weight: 600;
    font-size: 0.88rem;
}
.stButton button[kind="secondary"]:hover {
    border-color: var(--teal);
    color: var(--teal);
    background: transparent;
}
.stButton button:focus-visible {
    outline: 3px solid rgba(15, 92, 110, 0.35);
    outline-offset: 2px;
}

/* ---------- Verdict ---------- */
.verdict {
    display: flex;
    gap: 1.6rem;
    align-items: center;
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-top: 1.6rem;
    border: 1.5px solid;
}
.verdict.spam  { background: var(--alarm-soft); border-color: var(--alarm); }
.verdict.clean { background: var(--clear-soft); border-color: var(--clear); }

.postmark {
    flex: 0 0 auto;
    width: 118px;
    height: 118px;
    border-radius: 50%;
    border: 3px double currentColor;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    transform: rotate(-9deg);
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 1.15rem;
    line-height: 1.15;
    padding: 0.4rem;
}
.spam  .postmark { color: var(--alarm); }
.clean .postmark { color: var(--clear); }

.verdict h3 {
    font-family: 'Fraunces', serif;
    font-size: 1.45rem;
    margin: 0 0 0.35rem 0;
    color: var(--ink);
}
.verdict p { margin: 0; color: #2B3F49; line-height: 1.55; }

/* ---------- Meter ---------- */
.meter-wrap { margin-top: 1rem; }
.meter-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--muted);
    margin-bottom: 0.35rem;
}
.meter {
    height: 8px;
    border-radius: 999px;
    background: var(--line);
    overflow: hidden;
}
.meter > span { display: block; height: 100%; border-radius: 999px; }
.spam  .meter > span { background: var(--alarm); }
.clean .meter > span { background: var(--clear); }

/* ---------- Stats ---------- */
.stats {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.stat {
    flex: 1 1 140px;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
.stat b { display: block; font-size: 1.35rem; color: var(--ink); }
.stat span { font-size: 0.85rem; color: var(--muted); }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background-color: var(--ink);
    border-right: none;
}
[data-testid="stSidebar"] * { color: #E4ECF0 !important; }
.sb-name {
    font-family: 'Fraunces', serif;
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1.2;
    margin: 0.2rem 0 0.2rem 0;
    color: #FFFFFF !important;
}
.sb-role { font-size: 0.9rem; opacity: 0.8; margin-bottom: 1rem; }
.sb-rule { border: none; border-top: 1px solid rgba(255,255,255,0.15); margin: 1rem 0; }
.sb-list { font-size: 0.9rem; line-height: 1.7; padding-left: 1.1rem; margin: 0; }

/* ---------- Signature ---------- */
.signature {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--line);
    color: var(--muted);
    font-size: 0.9rem;
    text-align: center;
}
.signature b { font-family: 'Fraunces', serif; color: var(--ink); font-size: 1rem; }

@media (max-width: 600px) {
    .masthead h1 { font-size: 2rem; }
    .verdict { flex-direction: column; text-align: center; }
}
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# NLP setup
# ---------------------------------------------------------------------------
@st.cache_resource
def setup_nltk():
    for pkg in ("punkt", "punkt_tab", "stopwords"):
        nltk.download(pkg, quiet=True)
    return set(stopwords.words("english"))


STOP_WORDS = setup_nltk()
ps = PorterStemmer()


def transform_text(text: str) -> str:
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in STOP_WORDS and t not in string.punctuation]
    return " ".join(ps.stem(t) for t in tokens)


@st.cache_resource
def load_models():
    # File name kept exactly as in your project folder.
    tfidf = pickle.load(open("vectoriizer.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
    return tfidf, model


tfidf, model = load_models()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"<div class='sb-name'>{AUTHOR}</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sb-role'>Creator of Spam Inspector</div>", unsafe_allow_html=True
    )
    st.markdown("<hr class='sb-rule'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='font-size:0.95rem; line-height:1.6;'>
        Paste an SMS or email and get a clear verdict on whether it looks like spam or phishing.
        </div>
        <hr class='sb-rule'>
        <div style='font-weight:600; margin-bottom:0.4rem;'>How it works</div>
        <ul class='sb-list'>
            <li>Cleans and stems your text</li>
            <li>Converts it with a TF-IDF vectorizer</li>
            <li>Classifies it with a trained Naive Bayes model</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
SAMPLE_SPAM = (
    "Congratulations! You have won a free ticket worth $1000. "
    "Click the link now to claim your prize before it expires."
)
SAMPLE_HAM = "Hey, are we still meeting at 6 for dinner? Let me know if you want me to book a table."

if "message" not in st.session_state:
    st.session_state.message = ""

# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class='masthead'>
        <h1>Is this message safe to trust?</h1>
        <p>Paste any SMS or email below. Spam Inspector checks it for the patterns scammers and advertisers rely on.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

s1, s2, _ = st.columns([1.2, 1.2, 2])
with s1:
    if st.button("Try a spam sample", key="s_spam"):
        st.session_state.message = SAMPLE_SPAM
with s2:
    if st.button("Try a normal sample", key="s_ham"):
        st.session_state.message = SAMPLE_HAM

input_sms = st.text_area(
    "Message to check",
    key="message",
    placeholder="Paste the text of an SMS or email here...",
    height=150,
)

predict_clicked = st.button("Check message", type="primary")

if predict_clicked:
    if not input_sms.strip():
        st.warning("Paste a message first, then select Check message.")
    else:
        with st.spinner("Checking message..."):
            transformed = transform_text(input_sms)
            vector = tfidf.transform([transformed])
            result = int(model.predict(vector)[0])
            try:
                spam_prob = float(model.predict_proba(vector)[0][1])
            except Exception:
                spam_prob = None

        is_spam = result == 1
        css = "spam" if is_spam else "clean"
        stamp = "SPAM<br>FLAGGED" if is_spam else "LOOKS<br>CLEAN"
        title = "This looks like spam or phishing" if is_spam else "This looks like a genuine message"
        body = (
            "It matches patterns common in unsolicited offers and scams. "
            "Don't click links or share personal details."
            if is_spam
            else "No spam or phishing patterns were found. As always, be careful with unexpected links."
        )

        meter_html = ""
        if spam_prob is not None:
            shown = spam_prob if is_spam else 1 - spam_prob
            label = "Confidence it is spam" if is_spam else "Confidence it is legitimate"
            meter_html = f"""
            <div class='meter-wrap'>
                <div class='meter-label'><span>{label}</span><span>{shown * 100:.0f}%</span></div>
                <div class='meter'><span style='width:{shown * 100:.0f}%'></span></div>
            </div>
            """

        st.markdown(
            f"""
            <div class='verdict {css}'>
                <div class='postmark'>{stamp}</div>
                <div style='flex:1'>
                    <h3>{title}</h3>
                    <p>{body}</p>
                    {meter_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        words = len(input_sms.split())
        chars = len(input_sms)
        links = sum(
            1 for w in input_sms.lower().split() if w.startswith(("http", "www.")) or ".com" in w
        )
        st.markdown(
            f"""
            <div class='stats'>
                <div class='stat'><b>{words}</b><span>Words</span></div>
                <div class='stat'><b>{chars}</b><span>Characters</span></div>
                <div class='stat'><b>{links}</b><span>Links found</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    f"<div class='signature'>Designed and built by <b>{AUTHOR}</b></div>",
    unsafe_allow_html=True,
)