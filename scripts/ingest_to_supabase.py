import os
import glob
from typing import List

from supabase import create_client
from pypdf import PdfReader

from google import genai
from google.genai import types

SUPABASE_URL = os.environ["https://jcaitutfrqkelipgvfct.supabase.co"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["sb_publishable_p0C56Ce6NcIcAhLlpawHKw_659_gcy8"]
GEMINI_API_KEY = os.getenv("AIzaSyCMYEtjYcO-XCntKKyQsRzS-5cFI5EEbkI")

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIMS = int(os.getenv("EMBED_DIMS", "768"))

PDF_DIR = os.getenv("PDF_DIR", "./pdfs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = " ".join(text.split())
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        i += max(1, chunk_size - overlap)
    return chunks


def main():
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else genai.Client()

    cfg = types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=EMBED_DIMS,
    )

    pdfs = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {PDF_DIR}")

    for pdf_path in pdfs:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += (page.extract_text() or "") + "\n"

        chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"{os.path.basename(pdf_path)} -> {len(chunks)} chunks")

        # embed in batches (keep small to avoid request limits)
        batch_size = 32
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            res = client.models.embed_content(
                model=EMBED_MODEL,
                contents=batch,
                config=cfg,
            )
            embeddings = [e.values for e in res.embeddings]

            rows = []
            for chunk, emb in zip(batch, embeddings):
                rows.append({
                    "source": os.path.basename(pdf_path),
                    "chunk": chunk,
                    "embedding": emb,
                    "metadata": {"path": pdf_path},
                })

            supabase.table("documents").insert(rows).execute()

    print("Done.")


if __name__ == "__main__":
    main()
