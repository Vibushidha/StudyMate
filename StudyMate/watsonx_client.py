import os
from typing import List
from dotenv import load_dotenv

# watsonx imports
try:
    from ibm_watsonx_ai.foundation_models import Model
except Exception:
    Model = None

load_dotenv()

def use_watsonx() -> bool:
    return os.getenv("USE_WATSONX", "true").lower() == "true"

def get_model_config():
    return {
        "model_id": os.getenv("MODEL_ID", "mistralai/mixtral-8x7b-instruct-v01"),
        "project_id": os.getenv("IBM_PROJECT_ID", ""),
        "url": os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com"),
        "api_key": os.getenv("IBM_API_KEY", ""),
        "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "300")),
        "temperature": float(os.getenv("TEMPERATURE", "0.5")),
    }

def build_prompt(context_chunks: List[str], question: str) -> str:
    # Keep concise and grounded
    ctx = "\n\n".join(f"- {c.strip()}" for c in context_chunks if c and c.strip())
    prompt = f"""You are StudyMate, an academic assistant.
Answer the user's question STRICTLY using the context below. If the answer is not in the context, say you don't know.

Context:
{ctx}

Question: {question}

Rules:
- Be concise and clear.
- Cite which chunk(s) you used by brief labels like [S1], [S2] if helpful.
- Do NOT invent facts not present in the context.
"""
    return prompt

def generate_answer_with_fallback(context_chunks: List[str], question: str) -> str:
    """
    If USE_WATSONX=true and credentials are present, call watsonx.
    Otherwise, fall back to a simple extractive answer (best paragraph). 
    """
    prompt = build_prompt(context_chunks, question)
    if use_watsonx() and Model is not None:
        cfg = get_model_config()
        if not (cfg["api_key"] and cfg["project_id"] and cfg["url"]):
            # Missing credentials -> fallback
            pass
        else:
            try:
                model = Model(
                    model_id=cfg["model_id"],
                    params={
                        "decoding_method": "greedy",
                        "max_new_tokens": cfg["max_new_tokens"],
                        "temperature": cfg["temperature"],
                    },
                    credentials={
                        "apikey": cfg["api_key"],
                        "url": cfg["url"],
                    },
                    project_id=cfg["project_id"],
                )
                resp = model.generate_text(prompt=prompt)
                if isinstance(resp, dict) and "results" in resp and resp["results"]:
                    return resp["results"][0].get("generated_text", "").strip() or "(empty response)"
                # Some SDKs return raw string
                if isinstance(resp, str):
                    return resp.strip()
            except Exception as e:
                return f"(Watsonx error, showing best-match context instead)\n\n{_extractive_fallback(context_chunks, question)}"
    # Fallback
    return _extractive_fallback(context_chunks, question)

def _extractive_fallback(context_chunks: List[str], question: str) -> str:
    # Very simple heuristic: return the longest chunk and remind it's from sources
    if not context_chunks:
        return "I couldn't find relevant content in the uploaded PDFs."
    longest = max(context_chunks, key=lambda s: len(s))
    return f"Based on the retrieved context, here's a relevant excerpt:\n\n{longest}"
