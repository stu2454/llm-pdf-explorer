from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

import chromadb
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pypdf import PdfReader

# ───────────────────────────────────────────────────────────────
# Utility helpers
# ───────────────────────────────────────────────────────────────

def clean_filename(name: str) -> str:
    return Path(name).stem.replace(" ", "_").lower()


def requires_project_id(api_key: str) -> bool:
    return api_key.startswith("sk-proj-")

# ───────────────────────────────────────────────────────────────
# PDF parsing & chunking
# ───────────────────────────────────────────────────────────────

def get_pdf_text(uploaded_file) -> List[Document]:
    reader = PdfReader(uploaded_file)
    pages = [Document(page_content=(p.extract_text() or ""), metadata={"page": i + 1})
             for i, p in enumerate(reader.pages)]

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    return splitter.split_documents(pages)

# ───────────────────────────────────────────────────────────────
# Embeddings factory
# ───────────────────────────────────────────────────────────────

def make_embeddings(api_key: str, project_id: str | None) -> OpenAIEmbeddings:
    client = OpenAI(api_key=api_key, project=project_id) if project_id else OpenAI(api_key=api_key)
    return OpenAIEmbeddings(client=client.embeddings, model="text-embedding-3-small")

# ───────────────────────────────────────────────────────────────
# Chroma helpers (PersistentClient API)
# ───────────────────────────────────────────────────────────────

_DB_ROOT = Path("db")
_client: chromadb.PersistentClient | None = None


def _reset_db():
    shutil.rmtree(_DB_ROOT, ignore_errors=True)
    _DB_ROOT.mkdir(parents=True, exist_ok=True)


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _DB_ROOT.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(_DB_ROOT))
    return _client


def create_vectorstore_from_texts(
    docs,
    api_key: str,
    file_name: str,
    project_id: str | None = None,
):
    """Create or fetch a Chroma collection and ensure embeddings exist."""
    embeddings = make_embeddings(api_key, project_id)
    name = clean_filename(file_name)
    client = _get_client()

    try:
        # Load existing collection (if any)
        vectorstore = Chroma(client=client, collection_name=name, embedding_function=embeddings)
    except Exception:
        # Corrupt DB → wipe and recreate client
        _reset_db()
        client = _get_client()
        vectorstore = Chroma(client=client, collection_name=name, embedding_function=embeddings)

    # If the collection is empty, add docs with embeddings now
    if vectorstore._collection.count() == 0:
        vectorstore.add_documents(docs)

    return vectorstore

# ───────────────────────────────────────────────────────────────
# Retrieval + answer synthesis
# ───────────────────────────────────────────────────────────────

def query_document(vectorstore, query: str, api_key: str, project_id: str | None = None):
    context_docs = vectorstore.similarity_search(query, k=4)
    context = "\n\n".join(d.page_content for d in context_docs)

    prompt = (
        "Answer the question using *only* the context below. "
        "If the context is irrelevant, say so politely.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )

    client = OpenAI(api_key=api_key, project=project_id) if project_id else OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    return response.choices[0].message.content

