from flask import Flask
from agents.coordination_agent import CoordinationAgent
from agents.data_access_agent import DataAccessAgent
from agents.rag_agent import RAGAgent
from database import initialize_database


class _FallbackRAGAgent:
    def add_document(self, text, corpus_id):
        return None

    def query(self, question, corpus_id):
        return (
            "RAG agent is unavailable because Ollama/embedding services are not configured. "
            "Core corpus analysis agents remain available."
        )


def create_app():
    app = Flask(
        __name__,
        template_folder="../frontend/templates",
        static_folder="../frontend/static",
        static_url_path="/static",
    )
    app.config["UPLOAD_FOLDER"] = "data"

    initialize_database()
    data_agent = DataAccessAgent()
    try:
        rag_agent = RAGAgent()
    except Exception:
        rag_agent = _FallbackRAGAgent()
    coordination_agent = CoordinationAgent()

    # Save to app context
    app.data_agent = data_agent
    app.rag_agent = rag_agent
    app.coordination_agent = coordination_agent

    from .routes import main
    app.register_blueprint(main)

    return app
