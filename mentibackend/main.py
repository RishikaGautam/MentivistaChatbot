import os
import wave
import tempfile
import numpy as np
import datetime as dt

import sounddevice as sd
import simpleaudio as sa
from google.cloud import speach as gspeech
from google.cloud import texttospeech as gtts

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# 1) Load env (ensure ATT81274.env contains: GOOGLE_API_KEY=your-key)
load_dotenv(dotenv_path="ATT81274.env", override=False)
 
# (Optional) fail fast if key is missing
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY missing. Add it to ATT81274.env or your environment.")

if not os.getenv("GOOGLE_SPEACH_API_KEY"):
    raise RuntimeError("GOOGLE_SPEACH_API_KEY missing. Add it to ATT81274.env or your environment.")


# 2) LLM + embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# 3) Load PDFs
source_path = "/Users/rishikagautam/Desktop/Mentivista LLM/menti_backend/SellingTextbooks"
loader = PyPDFDirectoryLoader(source_path)
all_docs = loader.load()
if not all_docs:
    raise ValueError(f"No PDFs found at: {source_path}")

# 4) Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
splits = text_splitter.split_documents(all_docs)

# 5) Vector store + retriever
vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

#5.1) Record User Audio Input


# 6) Prompt (note: {today} is required → supply via .partial)
system_prompt = """
FORMATTING (Markdown — STRICT)
- Use **H1 section headers** when needed. 
- Under each header, use **bulleted lists** only (no long paragraphs).
- Bold all labels and increase size of labels
- Use **bold labels** inside bullets for scan-ability.
- Numbers that matter should be **bold**.
- If rules/limits are mentioned, lead with: **“As of {today} …”**.
- If inputs are missing, state **Assumptions** at the end as a short bulleted list.

You are a certified sales professional who provides personalized,
actionable guidance based on user inputs/questions.
When the user asks a sales question, analyze their full context and
deliver specific, tailored options. Adapt behavioral guidance to the user’s scenario.
Keep the tone educational and practical. Use concise bullets, avoid fluff,
and convert advice into simple dollar targets when possible.\n\n{context}


Your are a certified sales profession giving guidance based on:

Scenarios: - FOR THE VOICE OF THE USER:

"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
).partial(today=dt.date.today().isoformat())

# 7) Chains
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# 8) Run
question = "I own a massage center and I have recently made my own face mask, how do I effectively sell this product to my customers?"
result = rag_chain.invoke({"input": question})

print("\n=== Answer ===\n")
print(result.get("answer", result))

print("\n=== Sources (first few chunks) ===\n")
for i, doc in enumerate(result.get("context", [])[:5], start=1):
    meta = getattr(doc, "metadata", {}) or {}
    print(f"[{i}] {meta.get('source', 'unknown')} | p.{meta.get('page', '?')}")
