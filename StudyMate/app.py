import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator   # pip install deep-translator
import speech_recognition as sr              # pip install SpeechRecognition pyaudio
from gtts import gTTS                        # pip install gTTS
import tempfile
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Make sure you have the required libraries installed:
# pip install transformers torch sentence-transformers faiss-cpu deep-translator PyMuPDF SpeechRecognition gTTS streamlit

# Set page configuration for a wide layout
st.set_page_config(page_title="StudyMate", layout="wide")

# -------------------
# Q&A Logger Class
# -------------------
class QALog:
    """Manages the history of questions, answers, and sources."""
    def __init__(self):
        self.items = []

    def add(self, question, answer, sources):
        self.items.append({"question": question, "answer": answer, "sources": sources})

    def to_text(self):
        """Formats the Q&A history into a plain text string."""
        return "\n\n".join(
            f"Q: {item['question']}\nA: {item['answer']}\nSources: {', '.join(item['sources'])}"
            for item in self.items
        )

# -------------------
# Core PDF Processing and Embedding Functions
# -------------------
def extract_text_from_pdf(pdf_file):
    """Extracts all text from a single PDF file."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + " "
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    """Splits text into overlapping chunks of a specified size."""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def build_faiss_index(chunks, model):
    """Encodes text chunks and builds a FAISS index for fast similarity search."""
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

def search_index(query, model, index, chunks, top_k=3):
    """Searches the FAISS index for the most relevant text chunks."""
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec).astype("float32"), top_k)
    results = []
    for idx in indices[0]:
        # Ensure the index is within the bounds of the chunks list
        if idx < len(chunks):
            results.append({"text": chunks[idx], "source": f"Chunk {idx}"})
    return results

# -------------------
# Hugging Face Model Integration
# -------------------
@st.cache_resource
def load_huggingface_model():
    """Loads a Hugging Face tokenizer and model and caches them."""
    # You can replace this with any other Hugging Face model
    # Note: Larger models may require more RAM/VRAM
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def generate_answer_with_huggingface(context_chunks, question):
    """
    Generates an answer using a Hugging Face model.
    """
    if not context_chunks:
        return "No relevant content found in the uploaded PDFs."
    
    try:
        tokenizer, model = load_huggingface_model()
        
        # Construct the RAG prompt with retrieved context
        prompt = (
            "You are a helpful assistant. Use the following documents to answer the question. "
            "If the answer is not in the documents, say 'I cannot answer this question based on the documents.'.\n\n"
            "Documents:\n" + "\n---\n".join(context_chunks) + "\n\n"
            "Question: " + question + "\n\n"
            "Answer:"
        )
        
        # Tokenize the prompt and generate the answer
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, max_length=500, do_sample=False)
        
        # Decode the generated tokens
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
        
    except Exception as e:
        st.error(f"❌ Hugging Face Model Error: {e}")
        return "I'm sorry, an error occurred while generating the answer. Please check your model and environment setup."

# -------------------
# Translation Helpers
# -------------------
def translate(text, source_lang, target_lang):
    """Translates text from one language to another using deep_translator."""
    return GoogleTranslator(source=source_lang, target=target_lang).translate(text)

# -------------------
# Speech Helpers
# -------------------
def speak_text_streamlit(text, language="en"):
    """Converts text to speech and plays it in the Streamlit app."""
    try:
        tts = gTTS(text=text, lang=language)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            st.audio(tmp_file.name, format="audio/mp3")
        os.remove(tmp_file.name) # Clean up the temporary file
    except Exception as e:
        st.error(f"❌ Text-to-Speech Error: {e}")

# -------------------
# UI Text Dictionary
# -------------------
UI_TEXT = {
    "English": {
        "title": "📚 StudyMate — AI-Powered PDF Q&A",
        "caption": "Upload PDFs, ask questions, and get grounded answers with sources.",
        "upload": "Upload one or more academic PDFs",
        "ask": "Ask a question about your PDFs",
        "ask_button": "Ask",
        "history": "Q&A History",
        "download": "Download Q&A History",
        "bookmarks": "📌 Bookmarks",
        "download_bookmarks": "Download Bookmarks",
    },
    "Tamil": {
        "title": "📚 ஸ்டடி மேட் — செயற்கை நுண்ணறிவு PDF கேள்வி & பதில்",
        "caption": "PDF-களை பதிவேற்றி, கேள்விகளை கேளுங்கள், மற்றும் ஆதாரங்களுடன் பதில்களைப் பெறுங்கள்.",
        "upload": "ஒரு அல்லது அதற்கு மேற்பட்ட கல்வி PDF-களை பதிவேற்றவும்",
        "ask": "உங்கள் PDF-களைப் பற்றி ஒரு கேள்வி கேளுங்கள்",
        "ask_button": "கேள்",
        "history": "கேள்வி & பதில் வரலாறு",
        "download": "கேள்வி & பதில் வரலாறை பதிவிறக்கவும்",
        "bookmarks": "📌 புத்தக குறிச்சொற்கள்",
        "download_bookmarks": "புத்தக குறிச்சொற்களை பதிவிறக்கவும்"
    }
}

# -------------------
# Streamlit App Logic
# -------------------

# Initialize session state variables if they don't exist
if "qa" not in st.session_state:
    st.session_state.qa = QALog()
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "bookmarks" not in st.session_state:
    st.session_state.bookmarks = []
if "model" not in st.session_state:
    @st.cache_resource
    def load_embedding_model():
        return SentenceTransformer("all-MiniLM-L6-v2")
    st.session_state.model = load_embedding_model()

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    language = st.selectbox("🌍 Choose Language", ["English", "Tamil"])
    theme = st.radio("🎨 Theme", ["Light", "Dark"])
    top_k = st.slider("🔎 Number of chunks to retrieve (k)", 1, 10, 3)
    chunk_size = st.slider("📏 Chunk size (words)", 100, 1000, 500, step=50)
    overlap = st.slider("🔗 Chunk overlap (words)", 0, 300, 100, step=10)

# Apply theme styling
if theme == "Dark":
    st.markdown("""<style>body, .stApp {background-color: #121212; color: #f5f5dc;}.stTextInput > div > div > input {color: #f5f5dc; background-color: #333333;}.stButton button {background-color: #444444; color: #f5f5dc;}</style>""", unsafe_allow_html=True,)
else:
    st.markdown("""<style>body, .stApp {background-color: #f9f9f9; color: #000000;}.stButton button {background-color: #007bff; color: #ffffff;}</style>""", unsafe_allow_html=True,)

# Main App Content
st.title(UI_TEXT[language]["title"])
st.caption(UI_TEXT[language]["caption"])

# PDF Upload
uploaded = st.file_uploader(UI_TEXT[language]["upload"], type=["pdf"], accept_multiple_files=True)

if uploaded:
    all_text = ""
    for file in uploaded:
        all_text += extract_text_from_pdf(file)
    chunks = chunk_text(all_text, chunk_size=chunk_size, overlap=overlap)
    index = build_faiss_index(chunks, st.session_state.model)
    st.session_state.index = index
    st.session_state.chunks = chunks
    st.success("✅ PDFs processed successfully!")

# User Input Section
question = st.text_input(UI_TEXT[language]["ask"])
ask = st.button(UI_TEXT[language]["ask_button"], type="primary", disabled=st.session_state.index is None or not question.strip())

if ask:
    if st.session_state.index is None:
        st.warning("⚠️ Please upload a PDF before asking questions.")
    else:
        question_en = translate(question, "auto", "en")
        with st.spinner("🔍 Retrieving relevant passages..."):
            results = search_index(
                question_en,
                st.session_state.model,
                st.session_state.index,
                st.session_state.chunks,
                top_k=top_k
            )
        context_chunks = [r["text"] for r in results]
        sources = [r["source"] for r in results]

        with st.spinner("🤖 Generating answer..."):
            # Call the new Hugging Face function
            answer_en = generate_answer_with_huggingface(context_chunks, question_en)

        final_answer = translate(answer_en, "en", language)

        # Display the result
        st.markdown(f"**Answer ({language})**\n\n{final_answer}")
        speak_text_streamlit(final_answer, language="ta" if language == "Tamil" else "en")

        with st.expander("📖 Referenced Paragraphs"):
            for i, r in enumerate(results, 1):
                st.markdown(f"**S{i}. {r['source']}**\n{r['text']}")
                if st.button(f"🔖 Bookmark Chunk {i}", key=f"bm_{i}"):
                    st.session_state.bookmarks.append(r)

        st.session_state.qa.add(question, final_answer, sources)

# Q&A History
st.subheader(UI_TEXT[language]["history"])
for item in st.session_state.qa.items:
    st.markdown(f"**Q:** {item['question']}")
    st.markdown(f"**A:** {item['answer']}")

st.download_button(
    UI_TEXT[language]["download"],
    data=st.session_state.qa.to_text().encode("utf-8"),
    file_name="studymate_qa_history.txt",
    mime="text/plain",
    disabled=not st.session_state.qa.items
)

# Bookmarks
st.subheader(UI_TEXT[language]["bookmarks"])
if st.session_state.bookmarks:
    bookmarks_text = "\n\n".join([f"{bm['source']} - {bm['text']}" for bm in st.session_state.bookmarks])
    for bm in st.session_state.bookmarks:
        st.markdown(f"**{bm['source']}**\n{bm['text']}")
    st.download_button(
        UI_TEXT[language]["download_bookmarks"],
        data=bookmarks_text.encode("utf-8"),
        file_name="studymate_bookmarks.txt",
        mime="text/plain"
    )

# Footer
st.markdown("---")
st.markdown("© 2025 StudyMate | The Learning League")
