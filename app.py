
import streamlit as st
import requests
import html
import io
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="NOVA AI",
    page_icon="✦",
    layout="wide"
)

MODEL = "gemini-3.8-flash"
API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/"
    f"models/{MODEL}:generateContent"
)

# ---------------- DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #080b1b, #15102d, #090d20);
    color: white;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
}

h1, h2, h3, p, label {
    color: white;
}

.nova-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: white;
    margin-bottom: 5px;
}

.nova-subtitle {
    text-align: center;
    color: #b9b8d8;
    font-size: 16px;
    margin-bottom: 30px;
}

.welcome {
    text-align: center;
    padding: 28px 15px;
    border: 1px solid #51448b;
    border-radius: 20px;
    background: linear-gradient(135deg, #191735, #25204a);
    margin-bottom: 25px;
}

.welcome h2 {
    font-size: 30px;
    margin-bottom: 8px;
}

.welcome p {
    color: #c9c7e7;
}

div[data-testid="stChatMessage"] {
    border: 1px solid #454064;
    border-radius: 16px;
    padding: 14px;
    margin-bottom: 14px;
    background: #17152d;
}

div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-user"]
) {
    background: #ffffff;
    color: #17152d;
    border: 1px solid #dddded;
}

div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-user"]
) p {
    color: #17152d !important;
}

div[data-testid="stChatMessage"] p {
    color: white;
}

.stButton > button {
    width: 100%;
    min-height: 48px;
    border-radius: 12px;
    border: 1px solid #6555a8;
    background: #211d43;
    color: white;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #493b83;
    border-color: #a18bff;
    color: white;
}

.stChatInput textarea {
    color: white !important;
}

section[data-testid="stSidebar"] {
    background: #0d1023;
}

div[data-testid="stFileUploader"] {
    border: 1px dashed #7564b8;
    border-radius: 12px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ---------------- DOCUMENT FUNCTIONS ----------------

def extract_text(uploaded_file):
    name = uploaded_file.name.lower()

    try:
        if name.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
            return "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        if name.endswith(".txt"):
            return uploaded_file.getvalue().decode(
                "utf-8", errors="replace"
            )

    except Exception as error:
        st.warning(f"Could not read {uploaded_file.name}: {error}")

    return ""


def split_into_chunks(text, chunk_size=900, overlap=150):
    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def retrieve_context(question):
    chunks = st.session_state.documents

    if not chunks:
        return ""

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )

        matrix = vectorizer.fit_transform(
            chunks + [question]
        )

        scores = cosine_similarity(
            matrix[-1],
            matrix[:-1]
        ).flatten()

        best_indices = scores.argsort()[-4:][::-1]

        selected = [
            chunks[i]
            for i in best_indices
            if scores[i] > 0
        ]

        return "\n\n".join(selected)

    except Exception:
        return ""


# ---------------- GEMINI API ----------------

def ask_nova(question):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]

        context = retrieve_context(question)

        if context:
            prompt = f"""
Use the following document excerpts when they are relevant.
If the answer is not present in the documents, clearly say so
and then offer general help if appropriate.

DOCUMENT EXCERPTS:
{context}

USER QUESTION:
{question}
"""
        else:
            prompt = question

        system_instruction = """
You are NOVA, a helpful, friendly and intelligent AI assistant.
Give clear, accurate and useful answers.
Use simple explanations when appropriate.
When answering questions about uploaded documents,
prioritize the supplied document excerpts.
Do not claim to have read information that was not provided.
"""

        contents = []

        # Add earlier conversation, excluding the current question.
        for message in st.session_state.messages[-10:]:
            contents.append({
                "role": (
                    "user" if message["role"] == "user"
                    else "model"
                ),
                "parts": [{"text": message["content"]}]
            })

        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })

        payload = {
            "system_instruction": {
                "parts": [{"text": system_instruction}]
            },
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1500
            }
        }

        response = requests.post(
            API_URL,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key
            },
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            try:
                detail = response.json().get("error", {}).get(
                    "message", "Unknown API error"
                )
            except Exception:
                detail = "The API returned an unexpected response."

            return (
                f"NOVA could not get a response. "
                f"API error {response.status_code}: {detail}"
            )

        data = response.json()

        candidates = data.get("candidates", [])

        if not candidates:
            return "NOVA couldn't generate an answer. Please try again."

        parts = candidates[0].get("content", {}).get("parts", [])

        answer = "\n".join(
            part.get("text", "")
            for part in parts
            if part.get("text")
        ).strip()

        return answer or "NOVA returned an empty response. Please try again."

    except KeyError:
        return (
            "The GEMINI_API_KEY secret is missing. "
            "Please check your Streamlit app secrets."
        )

    except requests.exceptions.Timeout:
        return "The request timed out. Please try again."

    except requests.exceptions.RequestException:
        return "Could not connect to the AI service. Please try again."

    except Exception as error:
        return f"Something went wrong: {error}"


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("## ✦ NOVA AI")
    st.caption("Your intelligent assistant")

    st.divider()

    st.markdown("### 📁 Your documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process uploaded documents"):
            all_chunks = []

            for uploaded_file in uploaded_files:
                extracted = extract_text(uploaded_file)
                all_chunks.extend(split_into_chunks(extracted))

            st.session_state.documents = all_chunks

            st.success(
                f"Processed {len(uploaded_files)} file(s). "
                f"Created {len(all_chunks)} text chunks."
            )

    if st.session_state.documents:
        st.caption(
            f"{len(st.session_state.documents)} document chunks available"
        )

    st.divider()

    if st.button("＋ New conversation"):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()

    if st.button("Clear documents"):
        st.session_state.documents = []
        st.rerun()


# ---------------- HOME SCREEN ----------------

st.markdown(
    '<div class="nova-title">✦ NOVA AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="nova-subtitle">'
    'Your intelligent AI assistant'
    '</div>',
    unsafe_allow_html=True
)

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
        <h2>Hello Samiksha! ✨</h2>
        <p>Welcome to NOVA AI. What would you like to explore today?</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 💡 Suggested questions")

    suggestions = [
        "Explain a difficult topic in simple words",
        "Summarize my uploaded PDF",
        "Answer questions from my documents",
        "Help me brainstorm new ideas"
    ]

    columns = st.columns(2)

    for index, suggestion in enumerate(suggestions):
        with columns[index % 2]:
            if st.button(suggestion, key=f"suggestion_{index}"):
                st.session_state.pending_prompt = suggestion
                st.rerun()


# ---------------- CHAT HISTORY ----------------

for message in st.session_state.messages:
    # No custom avatar argument: avoids the avatar media-file error.
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------- CHAT INPUT ----------------

typed_question = st.chat_input("Ask NOVA anything...")

question = (
    st.session_state.pending_prompt
    or typed_question
)

if question:
    st.session_state.pending_prompt = None

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("NOVA is thinking..."):
            answer = ask_nova(question)

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
