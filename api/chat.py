import os
from typing import Optional, List, Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from supabase import create_client
from google import genai
from google.genai import types

# Vercel Python runtime expects an "app" variable for ASGI frameworks. :contentReference[oaicite:3]{index=3}
app = FastAPI()

# ---- Config ----
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Gemini API key can be GEMINI_API_KEY or GOOGLE_API_KEY. :contentReference[oaicite:4]{index=4}
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.0-flash-001")

EMBED_DIMS = int(os.getenv("EMBED_DIMS", "768"))
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.78"))
MATCH_COUNT = int(os.getenv("MATCH_COUNT", "8"))

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    # We raise runtime errors inside handlers to avoid breaking imports on deploy previews.
    pass

# Clients (constructed lazily)
_supabase = None
_genai = None


def get_supabase():
    global _supabase
    if _supabase is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY.")
        _supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _supabase


def get_genai():
    global _genai
    if _genai is None:
        # Client picks up env vars too, but we support explicit key. :contentReference[oaicite:5]{index=5}
        _genai = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else genai.Client()
    return _genai


def embed_query(text: str) -> List[float]:
    client = get_genai()
    # Gemini embedding supports output_dimensionality; default is larger. :contentReference[oaicite:6]{index=6}
    cfg = types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=EMBED_DIMS,
    )
    res = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[text],
        config=cfg,
    )
    # python-genai returns embeddings with `.values` (common pattern used by integrations)
    return res.embeddings[0].values


def build_prompt(user_text: str, docs: List[Dict[str, Any]]) -> str:
    context_blocks = []
    for d in docs:
        src = d.get("source") or d.get("metadata", {}).get("source") or "unknown"
        chunk = d.get("chunk", "")
        context_blocks.append(f"SOURCE: {src}\n{chunk}")

    context = "\n\n---\n\n".join(context_blocks).strip()

    return f"""You are a helpful startup sales/business assistant.
Use the CONTEXT if it is relevant. If the context does not contain the answer, say so and give the best general guidance.

CONTEXT:
{context if context else "(no matching context found)"}

USER QUESTION:
{user_text}

Answer clearly and practically. If you used the context, end with a short 'Sources:' list (just the SOURCE names).
"""


def generate_answer(prompt: str) -> str:
    client = get_genai()
    resp = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
    )
    # python-genai exposes .text for convenience in docs/examples :contentReference[oaicite:7]{index=7}
    return (resp.text or "").strip()


class ChatRequest(BaseModel):
    text: str
    conversation_id: Optional[str] = None


@app.get("/")
def health():
    return {"ok": True}


@app.post("/")
def chat(req: ChatRequest):
    try:
        supabase = get_supabase()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text'.")

    # 1) Ensure conversation exists
    conversation_id = req.conversation_id
    if not conversation_id:
        inserted = supabase.table("conversations").insert({}).execute()
        conversation_id = inserted.data[0]["id"]

    # 2) Save user message
    supabase.table("messages").insert({
        "conversation_id": conversation_id,
        "role": "user",
        "content": text
    }).execute()

    # 3) Embed + retrieve
    try:
        q_emb = embed_query(text)
        # Supabase recommends rpc('match_documents', ...) :contentReference[oaicite:8]{index=8}
        matches = supabase.rpc("match_documents", {
            "query_embedding": q_emb,
            "match_threshold": MATCH_THRESHOLD,
            "match_count": MATCH_COUNT,
        }).execute()
        docs = matches.data or []
    except Exception as e:
        docs = []
        # Keep going even if retrieval fails

    # 4) Generate answer grounded in context
    prompt = build_prompt(text, docs)
    answer = generate_answer(prompt)

    # 5) Save assistant message
    supabase.table("messages").insert({
        "conversation_id": conversation_id,
        "role": "assistant",
        "content": answer
    }).execute()

    used_sources = []
    for d in docs:
        src = d.get("source") or d.get("metadata", {}).get("source")
        if src and src not in used_sources:
            used_sources.append(src)

    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "sources": used_sources[:6],
    }
