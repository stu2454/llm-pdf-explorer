# 📰 LLM‑PDF‑Explorer

*Upload any PDF, ask questions, get answers – all in your browser.*

A Streamlit web‑app that turns PDFs into a searchable knowledge base with **OpenAI embeddings**, **GPT‑4o** and a local **Chroma** vector store.

---

## ✨ Features

| Capability                | Details                                                                                         |
| ------------------------- | ----------------------------------------------------------------------------------------------- |
| **One‑click upload**      | Drag‑and‑drop a PDF – pages are text‑extracted, chunked and embedded.                           |
| **Fast semantic search**  | Embeddings are cached in a local DuckDB/Parquet database via Chroma’s new **PersistentClient**. |
| **Natural‑language Q\&A** | Top‑k relevant chunks are fed to GPT‑4o to generate grounded answers.                           |
| **Per‑session API keys**  | Each visitor can paste their own OpenAI key (supports user or project‑scoped keys).             |
| **Self‑healing DB**       | Corrupt stores auto‑wipe and rebuild; zero manual migration pain.                               |
| **Docker‑ready**          | Single‑stage `Dockerfile` builds a lean image (\~200 MB) pinned to Python 3.11.                 |

---

## 🚀 Quick start

### 1  Clone & set up

```bash
git clone https://github.com/stu2454/llm-pdf-explorer.git
cd llm-pdf-explorer
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2  Add your OpenAI credentials

Create **`.env`** in the repo root:

```env
OPENAI_API_KEY=sk-…               # required
OPENAI_PROJECT_ID=proj_…          # only if using an sk-proj key
```

> `.env` is already in `.gitignore` – your secret stays local.

### 3  Run

```bash
streamlit run streamlit_app.py
# then visit http://localhost:8501
```

Upload a PDF, paste a key in **⚙️ Settings**, and start querying.

---

## 🐳 Docker

```bash
# build
docker build -t llm-pdf-explorer .

# run with env file
docker run -p 8501:8501 --env-file .env llm-pdf-explorer
```

Mount the `db/` folder as a volume if you want embeddings to persist:

```bash
docker run -p 8501:8501 -v $(pwd)/db:/app/db --env-file .env llm-pdf-explorer
```

---

## 🛠️ Architecture in 60 sec

```
PDF → PyPDF → LangChain chunk splitter
        ↓
  OpenAIEmbeddings (text-embedding‑3‑small) → Chroma PersistentClient (DuckDB)
        ↓                                             ↑
   GPT‑4o (chat completion) ← top‑k similarity search—┘
```

* **Embeddings dimension**: 1 536  *Collections are created with the correct dimensionality; the app refuses mismatched dims.*
* **Chunk size / overlap**: 512 / 64 tokens (tweak in `functions.py`).
* **Database path**: `./db` – delete it to start fresh.

---

## 📝 Roadmap / ideas

* 🔒 **Auth proxy** so each user requests with their own IP •
* 🔍 OCR pass for scanned PDFs •
* 🖼️ Inline image captioning for figures •
* 📊 Auto‑detect tables and render as DataFrames.

Contributions & ideas welcome – open an issue or PR!

---

## 🧑‍⚖️ Licence

MIT.  © 2025 Stu Smith.  Use freely, credit appreciated.

