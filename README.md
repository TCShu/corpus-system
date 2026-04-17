# ACAS — Agent-Driven Corpus Analysis System

ACAS is a Flask web app for corpus linguistics. Upload a text file, ask questions in plain language, and get structured linguistic analysis back — frequency tables, concordance lines, collocation scores, and keyword comparisons — powered by a local LLM via Ollama.

---

## Features

| Analysis Type | What it does |
|---|---|
| **Frequency** | Ranked word frequency table with relative percentages |
| **KWIC** | Keyword-in-Context concordance lines with configurable window |
| **N-gram / Collocation** | Bigram/trigram extraction with PMI association scores |
| **Keyword Comparison** | Keyness ratio between a target and reference corpus |
| **Corpus Q&A** | Open questions answered from corpus context (RAG-grounded) |

Results are validated before display. All analysis happens locally — no data leaves your machine.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | |
| [Ollama](https://ollama.com) | latest | Must be running before starting ACAS |
| A pulled model | `gemma3:4b` recommended | See step 3 below |

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd corpus-system-3
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and start Ollama

Download Ollama from [ollama.com](https://ollama.com) and install it.

Then pull a model (first run only):

```bash
ollama pull gemma3:4b
```

Verify Ollama is running:

```bash
ollama list
```

> **Default model:** `gemma3:4b`. Change this with the `OLLAMA_MODEL` environment variable (see Configuration below). Other supported models: `llama3`, `gemma3:12b`, `gemma3:27b`.

### 5. Run the app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## Configuration

Create a `.env` file in the project root to override defaults:

```env
# LLM model (must be pulled in Ollama)
OLLAMA_MODEL=gemma3:4b

# Database — SQLite by default, swap for PostgreSQL if needed
DATABASE_URL=sqlite:///./corpus_system.db
```

---

## Usage

### Upload a corpus

Click **Upload Corpus** in the sidebar and select a `.txt`, `.csv`, `.json`, or `.xml` file. The corpus appears in the library immediately.

### Run analysis

1. Select a corpus from the **target corpus** dropdown in the composer.
2. (Optional) Select a **reference corpus** for keyword comparison.
3. Type a natural-language request and press Enter or click the send button.

### Example prompts

```
Show the top 20 most frequent words excluding stopwords
Generate KWIC for "language" with a 10-word context window
Find bigram collocations with PMI scores
What topics appear most in this document?
Compare my corpus against the reference corpus
```

### Conversation history

Previous conversations are listed in the sidebar. Click any entry to replay the full thread. Click **+ New** to start a fresh conversation.

---

## Project Structure

```
corpus-system-3/
├── app.py                  # Entry point — run this
├── app/
│   ├── __init__.py         # Flask app factory
│   └── routes.py           # All API routes
├── agents/
│   ├── _shared.py          # Model selection shared across agents
│   ├── coordination_agent.py  # LangGraph router
│   ├── frequency_agent.py
│   ├── kwic_agent.py
│   ├── ngram_agent.py
│   ├── keyword_agent.py
│   ├── rag_agent.py
│   ├── data_access_agent.py
│   └── validation_agent.py
├── database/
│   ├── models.py           # SQLAlchemy ORM models
│   ├── db.py               # Engine + session factory
│   ├── db_manager.py       # Table creation
│   └── config.py           # DATABASE_URL setting
├── templates/
│   └── index.html          # Single-page app shell
├── static/
│   ├── css/style.css
│   └── js/main.js
├── data/                   # Uploaded corpus files (git-ignored)
├── chroma_db/              # Chroma vector store (git-ignored)
├── requirements.txt
└── .env                    # Local config (git-ignored)
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload corpus file (multipart `file` field) |
| `GET` | `/api/corpora` | List all uploaded corpora |
| `POST` | `/api/query` | Run analysis query |
| `GET` | `/api/conversations` | List conversation history |
| `GET` | `/api/conversations/<id>/messages` | Replay a conversation |
| `GET` | `/api/models` | List available Ollama models |
| `POST` | `/api/models/select` | Switch active model |
| `GET` | `/health` | Health check |

**Query request body:**
```json
{
  "question": "Show top 20 frequent words",
  "corpus_id": "myfile.txt",
  "reference_corpus_id": "reference.txt",
  "conversation_id": 3
}
```

---

## Troubleshooting

**"Coordination agent could not start"**
Ollama is not running. Start it with `ollama serve` (or open the Ollama desktop app) and reload the page.

**Upload succeeds but analysis fails**
The corpus is in the database but Ollama is unreachable. Confirm `ollama list` shows your model and try again.

**"No corpora uploaded yet" after upload**
Check the browser console. The `/api/upload` call may have returned an error — most likely the file format is unsupported (only `.txt`, `.csv`, `.json`, `.xml` are accepted).

**RAG warning message after upload**
The vector indexing step requires Ollama embeddings. Core frequency/KWIC/n-gram/keyword analysis still works without it — only the open-ended corpus Q&A route is affected.

---

## Running Tests

```bash
python -m pytest -q
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Flask |
| Agent orchestration | LangGraph |
| LLM | Ollama (local) |
| Vector store | Chroma |
| Database | SQLite (dev) / PostgreSQL (prod) |
| ORM | SQLAlchemy |
