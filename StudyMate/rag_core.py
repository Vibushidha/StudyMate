import io
import re
import fitz  # PyMuPDF
import faiss
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

# ---- PDF TEXT EXTRACTION ----
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from a single PDF (bytes) preserving basic layout."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        texts = []
        for page in doc:
            texts.append(page.get_text("text"))
    text = "\n".join(texts)
    # Basic cleanup: remove repeated whitespace, normalize headers/footers heuristically
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks based on word count."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks

# ---- EMBEDDINGS + INDEX ----
class EmbeddingIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # model_name accepts both "all-MiniLM-L6-v2" and full hub id
        if "/" not in model_name:
            model_name = f"sentence-transformers/{model_name}"
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.text_store: List[Dict] = []  # [{text, source, chunk_id}]
        self.dim = None

    def add_documents(self, docs: List[Dict[str, str]]):
        """
        docs: list of dicts with keys {text, source}
        """
        texts = [d["text"] for d in docs]
        embs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        if self.index is None:
            self.dim = embs.shape[1]
            self.index = faiss.IndexFlatIP(self.dim)  # cosine via normalized vectors
        self.index.add(embs)
        base_id = len(self.text_store)
        for i, d in enumerate(docs):
            self.text_store.append({
                "text": d["text"],
                "source": d.get("source", "unknown"),
                "chunk_id": base_id + i
            })

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        D, I = self.index.search(q.astype(np.float32), top_k)
        out = []
        if I.size > 0:
            idxs = I[0]
            scores = D[0]
            for rank, (ii, sc) in enumerate(zip(idxs, scores), 1):
                if ii == -1:
                    continue
                meta = self.text_store[ii]
                out.append({
                    "rank": rank,
                    "score": float(sc),
                    "text": meta["text"],
                    "source": meta["source"],
                    "chunk_id": meta["chunk_id"]
                })
        return out

# ---- HELPER TO BUILD CORPUS ----
def build_corpus_from_files(uploaded_files: List, chunk_size: int, overlap: int) -> List[Dict]:
    """
    uploaded_files: list of streamlit UploadedFile-like objects (must have .name and .read())
    Returns list of dicts: {text, source}
    """
    corpus = []
    for f in uploaded_files:
        try:
            pdf_bytes = f.read()
            text = extract_text_from_pdf_bytes(pdf_bytes)
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for i, ch in enumerate(chunks):
                corpus.append({"text": ch, "source": f"{f.name} | chunk {i+1}"})
        except Exception as e:
            # Skip corrupt files
            print(f"Failed to process {getattr(f, 'name', 'file')}: {e}")
    return corpus
