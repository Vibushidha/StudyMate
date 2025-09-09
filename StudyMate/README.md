# StudyMate: An AI‑Powered PDF Q&A (RAG) System

StudyMate lets students upload one or more PDFs and ask natural-language questions. It retrieves the most relevant text chunks and generates answers grounded in the source content.

## Features
- Multi‑PDF upload (PyMuPDF for robust text extraction)
- Overlapping chunking (default: 500 words, overlap: 100)
- Semantic embeddings with `sentence-transformers` (`all-MiniLM-L6-v2`)
- Fast similarity search using FAISS (cosine similarity via inner product)
- IBM watsonx.ai LLM (Mixtral‑8x7B‑Instruct) integration
- Session Q&A history with one‑click download
- Expandable “Referenced Paragraphs” to verify grounding
- Optional **local extractive fallback** (no API key required) for demo

## Quickstart
1. **Install Python 3.10+** (Windows 11 tested).
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in IBM credentials (or keep fallback mode):
   ```bash
   cp .env.example .env
   # then edit .env
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```

## .env
```env
USE_WATSONX=true
IBM_API_KEY=your_api_key
IBM_PROJECT_ID=your_project_id
IBM_URL=https://us-south.ml.cloud.ibm.com
MODEL_ID=mistralai/mixtral-8x7b-instruct-v01
MAX_NEW_TOKENS=300
TEMPERATURE=0.5
TOP_K=3
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

- Set `USE_WATSONX=false` to use a simple **local extractive answer** from retrieved chunks.

## Notes
- FAISS CPU wheels are used (`faiss-cpu`). If install issues on Windows, try:
  ```bash
  pip install --no-cache-dir faiss-cpu
  ```
- The app normalizes embeddings and uses `IndexFlatIP` (cosine).

## Project Structure
```
StudyMate/
├─ app.py
├─ rag_core.py
├─ watsonx_client.py
├─ qa_logger.py
├─ requirements.txt
├─ .env.example
└─ README.md
```

## License
MIT
