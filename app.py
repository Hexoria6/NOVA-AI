
import streamlit as st
import requests
import pypdf
import io
import html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="NOVA AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL = "gemini-3.6-flash"
API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/"
    f"models/{MODEL}:generateContent"
)

# ==================================================
# SESSION STATE
# ==================================================

defaults = {
    "messages": [],
    "chunks": [],
    "vectorizer": None,
    "matrix": None,
    "document_name": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ==================================================
# PREMIUM NOVA DESIGN
# ==================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
      radial-gradient(ellipse at 5% 5%, #242052 0%, transparent 38%),
      radial-gradient(ellipse at 95% 85%, #123451 0%, transparent 35%),
      #080b16;
    color: #f5f6ff;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: #101427;
    border-right: 1px solid rgba(150,130,255,.18);
}

section[data-testid="stSidebar"] * {
    color: #e9eaff;
}

/* HERO */

.hero {
    padding: 38px 32px;
    border-radius: 25px;
    background: linear-gradient(
        120deg,
        rgba(111,76,255,.25),
        rgba(28,164,255,.12)
    );
    border: 1px solid rgba(165,145,255,.25);
    margin-bottom: 25px;
    box-shadow: 0 15px 60px rgba(0,0,0,.15);
}

.logo {
    color: #b8a9ff;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 5px;
}

.hero h1 {
    font-size: clamp(38px, 6vw, 60px);
    font-weight: 800;
    margin: 17px 0 10px;
    background: linear-gradient(90deg,#ffffff,#b8a5ff,#73d8ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #bdc4df;
    font-size: 15px;
    line-height: 1.8;
}

.status {
    display: inline-block;
    color: #7ff5c3;
    background: rgba(50,220,160,.10);
    border: 1px solid rgba(80,240,180,.25);
    border-radius: 30px;
    padding: 7px 13px;
    font-size: 12px;
}

/* DASHBOARD CARDS */

.panel {
    background: rgba(24,29,53,.85);
    border: 1px solid rgba(150,145,255,.18);
    border-radius: 19px;
    padding: 20px;
    margin-bottom: 16px;
    min-height: 115px;
}

.small-muted {
    color: #929bbd;
    font-size: 12px;
    letter-spacing: .5px;
}

/* USER QUESTION - WHITE BUBBLE */

.user-bubble {
    background: #ffffff;
    color: #17182a;
    padding: 17px 21px;
    border-radius: 19px 19px 4px 19px;
    margin: 18px 0 18px auto;
    max-width: 78%;
    width: fit-content;
    font-size: 15px;
    line-height: 1.7;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    box-shadow: 0 8px 25px rgba(0,0,0,.15);
}

.user-label {
    color: #7164d9;
    font-weight: 800;
    font-size: 11px;
    letter-spacing: 1px;
    margin-bottom: 7px;
}

/* ASSISTANT CHAT MESSAGE */

.stChatMessage {
    background: rgba(25,31,56,.92);
    border: 1px solid rgba(145,130,255,.20);
    border-radius: 18px;
    padding: 13px 16px;
    margin: 14px 0;
}

.stChatMessage p,
.stChatMessage li,
.stChatMessage h1,
.stChatMessage h2,
.stChatMessage h3 {
    color: #f0f1ff !important;
    line-height: 1.75;
}

/* INPUT */

[data-testid="stChatInput"] {
    background: #151a30;
    border: 1px solid #514b91;
    border-radius: 17px;
}

[data-testid="stChatInput"] textarea {
    color: #ffffff !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #929bbd !important;
}

/* BUTTONS */

div.stButton > button {
    width: 100%;
    border: 0;
    border-radius: 12px;
    padding: 12px 18px;
    font-weight: 700;
    color: white;
    background: linear-gradient(100deg,#7656ff,#368dff);
    transition: .2s ease;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(105,90,255,.35);
    color: white;
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {
    background: rgba(30,35,65,.5);
    border: 1px dashed #7770c9;
    border-radius: 15px;
    padding: 12px;
}

/* GENERAL */

h2, h3 {
    color: #e9eaff;
}

hr {
    border-color: rgba(150,145,255,.18);
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# DOCUMENT PROCESSING
# ==================================================

def extract_text(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        return uploaded_file.getvalue().decode(
            "utf-8", errors="ignore"
        )

    if filename.endswith(".pdf"):
        reader = pypdf.PdfReader(
            io.BytesIO(uploaded_file.getvalue())
        )
        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    return ""


def split_text(text, size=180):
    words = text.split()
    return [
        " ".join(words[i:i + size])
        for i in range(0, len(words), size)
        if words[i:i + size]
    ]


def index_document(text):
    chunks = split_text(text)

    if not chunks:
        return False

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=12000
    )

    matrix = vectorizer.fit_transform(chunks)

    st.session_state.chunks = chunks
    st.session_state.vectorizer = vectorizer
    st.session_state.matrix = matrix

    return True


def retrieve_context(question):
    if not st.session_state.chunks:
        return ""

    vectorizer = st.session_state.vectorizer
    matrix = st.session_state.matrix

    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, matrix)[0]

    best_indices = scores.argsort()[-4:][::-1]

    selected = [
        st.session_state.chunks[i]
        for i in best_indices
        if scores[i] > 0
    ]

    return "\n\n".join(selected)


# ==================================================
# GEMINI AI CONNECTION
# ==================================================

def ask_nova(question):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]

    except Exception:
        return (
            "Gemini API key is missing. "
            "Please add GEMINI_API_KEY in Streamlit Secrets."
        )

    context = retrieve_context(question)

    if context:
        prompt = f"""
DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Answer clearly using the document context when relevant.
If the document does not contain the answer, say so.
"""
    else:
        prompt = question

    contents = []

    # Include recent conversation history
    for message in st.session_state.messages[-10:]:
        role = (
            "user"
            if message["role"] == "user"
            else "model"
        )

        contents.append({
            "role": role,
            "parts": [{"text": message["content"]}]
        })

    # Add current question and document context
    contents.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })

    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "system_instruction": {
            "parts": [{
                "text": (
                    "You are NOVA AI, a friendly, intelligent, "
                    "professional assistant. Give accurate, "
                    "helpful and well-organized answers. "
                    "Use document context when it is provided. "
                    "Do not invent information from documents."
                )
            }]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.6,
            "maxOutputTokens": 1200
        }
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=90
        )

        if response.status_code != 200:
            try:
                error = response.json().get("error", {}).get(
                    "message", response.text
                )
            except Exception:
                error = response.text

            return (
                f"Gemini API error ({response.status_code}): "
                f"{error}"
            )

        data = response.json()

        return "".join(
            part.get("text", "")
            for part in data["candidates"][0]["content"]["parts"]
        ).strip()

    except requests.exceptions.Timeout:
        return "NOVA took too long to respond. Please try again."

    except Exception as e:
        return f"Connection error: {str(e)}"


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("""
    <div style="padding:12px 0 24px">
        <div style="font-size:31px;font-weight:800;color:#b7a5ff">
            ✦ NOVA
        </div>
        <div class="small-muted">
            YOUR INTELLIGENCE SPACE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<span class="status">● SYSTEM READY</span>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### ◈ Knowledge base")

    uploaded_file = st.file_uploader(
        "Upload your PDF or TXT",
        type=["pdf", "txt"]
    )

    if uploaded_file:
        if st.button("✦ Process document"):
            with st.spinner("Reading your document..."):
                try:
                    text = extract_text(uploaded_file)

                    if index_document(text):
                        st.session_state.document_name = (
                            uploaded_file.name
                        )
                        st.success("Document indexed successfully!")
                    else:
                        st.error("No readable text found.")

                except Exception as e:
                    st.error(f"Document error: {e}")

    if st.session_state.chunks:
        st.success(
            f"{len(st.session_state.chunks)} sections ready"
        )
        st.caption(
            f"File: {st.session_state.document_name}"
        )

    st.markdown("---")

    if st.button("＋ New conversation"):
        st.session_state.messages = []
        st.rerun()

    if st.button("Clear document"):
        st.session_state.chunks = []
        st.session_state.vectorizer = None
        st.session_state.matrix = None
        st.session_state.document_name = None
        st.rerun()

    st.markdown("---")

    st.markdown("""
    <div class="small-muted">
        ✦ Powered by Gemini<br><br>
        NOVA AI • YOUR IDEAS, AMPLIFIED
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# MAIN DASHBOARD
# ==================================================

st.markdown("""
<div class="hero">
    <div class="logo">✦ NOVA INTELLIGENCE</div>
    <h1>Think beyond limits.</h1>
    <p>
        Your personal AI workspace.<br>
        Ask questions, explore ideas and discover answers
        from your documents.
    </p>
    <span class="status">● READY TO ASSIST</span>
</div>
""", unsafe_allow_html=True)


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="panel">
        <div class="small-muted">AI ENGINE</div>
        <h3>✦ Gemini AI</h3>
        <div class="small-muted">Intelligent assistance</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="panel">
        <div class="small-muted">KNOWLEDGE BASE</div>
        <h3>▤ {len(st.session_state.chunks)} sections</h3>
        <div class="small-muted">Document search</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="panel">
        <div class="small-muted">CONVERSATION</div>
        <h3>◈ {len(st.session_state.messages)} messages</h3>
        <div class="small-muted">Your AI workspace</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("## ✦ Your conversation")
st.caption(
    "Ask NOVA anything or ask questions about your uploaded document."
)

# ==================================================
# CHAT HISTORY
# ==================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        safe_text = html.escape(message["content"])

        st.markdown(
            f"""
            <div class="user-bubble">
                <div class="user-label">YOU</div>
                {safe_text}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        with st.chat_message("assistant", avatar="✦"):
            st.markdown(message["content"])


# ==================================================
# CHAT INPUT
# ==================================================

question = st.chat_input("Message NOVA AI...")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.spinner("✦ NOVA is thinking..."):
        answer = ask_nova(question)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()


# ==================================================
# FOOTER
# ==================================================

st.markdown("""
<div style="
    text-align:center;
    padding:35px 0 5px;
    color:#7882a5;
    font-size:12px;
    letter-spacing:2px;">
    ✦ NOVA AI • YOUR IDEAS, AMPLIFIED
</div>
""", unsafe_allow_html=True)
