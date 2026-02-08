# **ACAS Corpus Analysis System**
## Project Overview
The Agent-Based Corpus Analysis System (ACAS) is designed to facilitate automated linguistic analysis of text corpora through an intelligent, agent-based architecture.

The system provides:
- Frequency analysis
- Keyword extraction
- N-gram analysis
- KWIC (Key Word in Context)
- Coordinated agent responses

The backend is built with **Python + Flask** and organized using an **agent-based architecture**.
The frontend will communicate with the backend through HTTP API endpoints.

---

## Tech Stack
| Layer | Technology |
|-------|------|
| Backend | Python, Flask |
|-------|------|
| NLP | NLTK |
|-------|------|
| Architecture | Agent-based modular services |
| Database | PostgreSQL |
| Containerization | Docker |
| Frontend | HTML / CSS / JavaScript |
| Testing | Pytest |

---

## Repository Structure
corpus-system/
├── agents/ # Independent analysis agents 
├── app/ # Flask application 
├── frontend/ # HTML templates + static files 
├── database/ # Models and migrations (future sprint) 
├── tests/ # Unit and integration tests 
├── data/ # Sample corpora 
├── docker/ # Containerization (future sprint) 
└── README.md

---

## Prerequisites
Install the following before setup:
- Python 3.10+
- pip
- git

---

## Project Setup
### 1. Clone the repository
git clone <repo-url>
cd acas-corpus-analysis

### 2. Create a virtual environment

Windows:
    python -m venv venv
    venv\Scripts\activate

Mac/Linux:
    python3 -m venv venv
    source venv/bin/activate

You should now see:
    (venv)

### 3. Install dependencies
pip install -r requirements.txt

### 4. Download NLP resources (IMPORTANT)
We use NLTK tokenizers which must be installed once per environment.

Run:
    python -m nltk.downloader punkt

This only needs to be done once after creating the virtual environment.

---

## Running the Application
From the project root:
    python -m backend.app

Then open in browser:
    http://127.0.0.1:5000

---

## Development Workflow
### Branching Strategy

Each feature must be developed in its own branch:

    feature/data-access-agent
    feature/frequency-agent
    feature/flask-routes
    feature/frontend-upload

Never push directly to main.

---

## Agents Architecture

Each analysis module is an independent agent:

| Agent | Responsibility |
| Data Access Agent | Reads and validates corpus |
| Frequency Agent | Word frequency counts |
| Keyword Agent | Important term extraction |
| Ngram Agent | Phrase pattern analysis |
| KWIC Agent | Context search |
| Coordinating Agent | Combines agent outputs |

Agents must:

- Be independent
- Not depend on Flask
- Return pure Python data structures (dict/list)

---

## Testing
Tests will be written using Pytest:
    pytest


## NLTK Setup
After installing dependencies, run:

```bash 
python -m nltk.downloader punkt
