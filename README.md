# ACAS – Agent-Based Corpus Analysis System

## Project Overview

The Agent-Based Corpus Analysis System (ACAS) is an intelligent backend system for corpus ingestion, semantic retrieval, and agent-based text analysis.

The system supports:

- Dynamic corpus upload
- Multi-corpus isolation
- Retrieval-Augmented Generation (RAG)
- Local LLM responses via Ollama
- Vector-based semantic search using Chroma
- Agent-based modular backend architecture

The backend is built using **Python + Flask** and organized using a modular agent-based design.

---

## Current Architecture

Client  
↓  
Flask API  
↓  
Coordinating Logic (in progress)  
↓  
Data Access Agent + RAG Agent  
↓  
Chroma Vector Store + (Upcoming) PostgreSQL  
↓  
Ollama (Local LLM)

---

## Tech Stack

| Layer | Technology |
|--------|------------|
| Backend | Python, Flask |
| LLM | Ollama (local models) |
| Vector Database | Chroma |
| Embeddings | langchain_ollama |
| Architecture | Agent-based modular services |
| Database (next phase) | PostgreSQL |
| Testing | Pytest |

---

## Repository Structure
```
corpus-system/
├── agents/ # Agent implementations (RAG, Data, etc.)
├── app/ # Flask app + routes
├── data/ # Uploaded corpora
├── chroma_db/ # Local vector index (DO NOT COMMIT)
├── tests/ # Unit + integration tests
├── run.py # App entry point
└── README.md
```

---

## Setup

### 1. Clone

git clone <repo-url>
cd corpus-system

### 2. Create virtual environment

PowerShell:
```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```
---

## Running the Application
```
python run.py
```

Server runs at:
http://127.0.0.1:5000

---

## API Endpoints

### Upload Corpus
```POST /upload```

Form-data:
- file: text file

Indexes document into Chroma with a corpus_id.

---

### Query Corpus
```POST /ask```


JSON body:

```json
{
  "question": "Your question here",
  "corpus_id": "sample.txt"
}
