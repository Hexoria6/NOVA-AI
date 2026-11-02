import io
import requests
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- SETTINGS ----------------
st.set_page_config(
    page_title="NOVA AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "openai/gpt-oss-120b:fastest"

# ---------------- DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #090d1c, #15132e, #21163d);
    color: #f5f3ff;
}
[data-testid="stSidebar"] {
    background: #101329;
    border-right: 1px solid #38305f;
}
.hero {
    padding: 35px;
    border-radius: 22px;
    background: linear-gradient(120deg, #34246d, #512b79, #263b70);
    border: 1px solid #7760b5;
    margin-bottom: 25px;
}
.hero h1 {
    font-size: 48px;
    color: white;
    margin-bottom: 5px;
}
.hero p {
    color: #e0d9ff;
    font-size: 17px;
}
.welcome {
    padding: 22px;
    border-radius: 16px;
    background: rgba(105, 83, 170, 0.16);
    border: 1px solid #514477;
    margin: 15px 0;
}
div[data-testid="stChatMessage"] {
    background: rgba(38, 36, 70, 0.8);
    border: 1px solid #514477;
    border-radius: 16px;
    padding: 12px;
}
.stButton button {
    border-radius: 12px;
    background: linear-gradient(90deg, #7654d6, #a34ed0);
    color: white;
    border: none;
    font-weight: bold;
}
.stButton button:hover {
    border: 1px solid #c4a8ff;
    color: white;
}
.small-note {
    color: #aaa6c9;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
defaults = {
    "chunks": [],
    "vectorizer": None,
    "matrix": None,
    "document_name": "",
    "messages": []
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------- DOCUMENT FUNCTIONS ----------------
def extract_text(file):
    if file.name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file.getvalue()))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if file.name.lower().endswith(".txt"):
        return file.getvalue().decode("utf-8", errors="ignore")

    return ""


def make_chunks(text, size=1000, overlap=150):
    text = " ".join(text.split())
    chunks = []

    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        part = text[start:end].strip()

        if part:
            chunks.append(part)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def process_document(file):
    raw_text = extract_text(file)
    chunks = make_chunks(raw_text)

    if not chunks:
        raise ValueError(
            "No readable text found. Scanned PDFs may need OCR."
        )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=12000
    )
    matrix = vectorizer.fit_transform(chunks)

    st.session_state.chunks = chunks
    st.session_state.vectorizer = vectorizer
    st.session_state.matrix = matrix
    st.session_state.document_name = file.name
    st.session_state.messages = []


def retrieve(question, top_k=4):
    vectorizer = st.session_state.vectorizer
    matrix = st.session_state.matrix
    chunks = st.session_state.chunks

    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, matrix).flatten()

    best_ids = scores.argsort()[::-1][:top_k]

    return [
        {"text": chunks[i], "score": float(scores[i])}
        for i in best_ids
    ]


# ---------------- ONLINE AI ----------------
def ask_nova(question, sources):
    try:
        token = st.secrets["HF_TOKEN"]
    except Exception:
        raise ValueError(
            "HF_TOKEN is missing. Add it in your hosting app's Secrets settings."
        )

    context = "\n\n".join(
        f"Source {i+1}: {item['text']}"
        for i, item in enumerate(sources)
    )

    system_prompt = """
You are NOVA AI, a helpful document assistant.

Answer questions using the supplied document context.
Do not invent information.
If the document does not contain the answer, say so clearly.
Give clear, well-organized answers.
"""

    response = requests.post(
        HF_API_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={
            "model": HF_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"DOCUMENT CONTEXT:\n{context}\n\n"
                        f"QUESTION:\n{question}"
                    )
                }
            ],
            "max_tokens": 900,
            "temperature": 0.4
        },
        timeout=120
    )

    if not response.ok:
        raise RuntimeError(
            f"Hugging Face API error ({response.status_code}): "
            f"{response.text[:500]}"
        )

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# ✨ NOVA AI")
    st.caption("YOUR PERSONAL DOCUMENT INTELLIGENCE")

    st.divider()

    st.markdown("### 📚 Knowledge Base")
    uploaded_file = st.file_uploader(
        "Upload your document",
        type=["pdf", "txt"],
        help="Choose a PDF or TXT file."
    )

    if st.button("⚡ Process Document", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please upload a document first.")
        else:
            try:
                with st.spinner("Reading and indexing document..."):
                    process_document(uploaded_file)
                st.success("Document processed successfully!")
            except Exception as error:
                st.error(str(error))

    st.divider()

    st.markdown("### 📊 Document Status")

    if st.session_state.document_name:
        st.success("Document ready")
        st.write("📄 " + st.session_state.document_name)
        st.write(
            f"🧩 {len(st.session_state.chunks)} passages indexed"
        )
    else:
        st.info("No document uploaded yet.")

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        '<p class="small-note">Powered by NOVA AI · RAG Technology</p>',
        unsafe_allow_html=True
    )


# ---------------- MAIN PAGE ----------------
st.markdown("""
<div class="hero">
    <h1>✨ NOVA AI</h1>
    <p>Your documents. Your questions. Intelligent answers.</p>
    <p>AI-powered Retrieval Augmented Generation Assistant</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="welcome">
    <h3>👋 Hello Samiksha!</h3>
    <p>Welcome to NOVA AI. Upload a document and let me help
    you understand it, find information and get answers.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- METRICS ----------------
if st.session_state.chunks:
    col1, col2, col3 = st.columns(3)

    col1.metric("📄 Document", "Ready")
    col2.metric("🧩 Passages", len(st.session_state.chunks))
    col3.metric("🔎 Search", "TF-IDF RAG")

    with st.expander("📖 Preview document passages"):
        for i, chunk in enumerate(st.session_state.chunks[:5]):
            st.markdown(f"**Passage {i+1}**")
            st.write(chunk[:600])

# ---------------- CHAT HISTORY ----------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("🔎 View document sources"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(
                        f"**Source {i+1} · Score {source['score']:.3f}**"
                    )
                    st.write(source["text"])

# ---------------- CHAT INPUT ----------------
st.markdown("### 💬 Chat with your document")

question = st.chat_input(
    "Ask NOVA AI anything about your uploaded document..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    if not st.session_state.chunks:
        answer = "Please upload and process a PDF or TXT document first."
        sources = []

    else:
        try:
            with st.spinner("✨ NOVA is thinking..."):
                sources = retrieve(question)
                answer = ask_nova(question, sources)

        except Exception as error:
            answer = f"Sorry, something went wrong: {error}"
            sources = []

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

    with st.chat_message("assistant"):
        st.markdown(answer)

        if sources:
            with st.expander("🔎 View document sources"):
                for i, source in enumerate(sources):
                    st.markdown(
                        f"**Source {i+1} · Score {source['score']:.3f}**"
                    )
                    st.write(source["text"])

# ---------------- EXAMPLE QUESTIONS ----------------
with st.expander("💡 Try these example questions"):
    st.markdown("""
    - What is this document about?
    - Summarize the main points.
    - Explain the important concepts.
    - What are the key findings?
    - What does the document say about a particular topic?
    """)

st.markdown("""
<p class="small-note" style="text-align:center; margin-top:35px;">
NOVA AI · Intelligent Document Assistant
</p>
""", unsafe_allow_html=True)
