# ACAS - Agent-Driven Corpus Analysis System

ACAS is a Flask-based corpus linguistics platform with a multi-agent backend.  
Users upload corpora and ask natural-language analysis questions through a web UI.

## Implemented Features

- Corpus upload and preprocessing (`TXT`, `CSV`, `JSON`, `XML`)
- Agent orchestration from natural-language prompts
- Specialized linguistic agents:
  - Frequency Analysis
  - KWIC Concordance
  - N-gram/Collocation (with PMI)
  - Keyword Comparison (target vs reference corpus)
- WatchDog validation before displaying results
- Safe dynamic fallback path:
  - Unsupported but linguistic prompts trigger generated Python
  - Code is statically validated
  - Code executes only in a restricted Docker container
  - If container execution is unavailable/unsafe, result is blocked
- Persistence for corpora, documents, queries, results, and execution logs
- Frontend dashboard aligned to project mockup style

## Architecture

1. Presentation Layer: Flask-rendered UI (`frontend/templates/index.html`)
2. Coordination Layer: `CoordinationAgent`
3. Agent Service Layer: `FrequencyAgent`, `KWICAgent`, `NgramAgent`, `KeywordAgent`, `ValidationAgent`
4. Safety Layer: `SafeCodeExecutionService` (restricted Docker runtime)
5. Data Layer:
   - SQLAlchemy schema (users, corpora, documents, queries, results, logs)
   - Chroma vector store (RAG path)

## Project Structure

```text
agents/      # All analysis + coordination + validation agents
app/         # Flask app factory, routes, main entrypoint
database/    # SQLAlchemy config/models/session helpers
services/    # CRUD services + safe code execution service
frontend/    # HTML template(s)
tests/       # Unit/integration API and DB tests
```

## Setup

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```powershell
python run.py
```

Open: `http://127.0.0.1:5000`

## Main API

- `POST /api/upload` (multipart form-data `file`)
- `GET /api/corpora`
- `POST /api/query`
- `GET /health`
- Legacy compatibility:
  - `POST /upload`
  - `POST /ask`

## Testing

Use the virtual environment interpreter:

```powershell
.\venv\Scripts\python.exe -m pytest -q
```
