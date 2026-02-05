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


## NLTK Setup
After installing dependencies, run:

```bash 
python -m nltk.downloader punkt
