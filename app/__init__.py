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
            "RAG agent is unavailable — Ollama/embedding services are not configured. "
            "Core corpus analysis agents remain available."
        )


class _FallbackCoordinationAgent:
    def execute(self, query, tokens, reference_tokens=None, corpus_text=""):
        return {
            "route": "out_of_scope",
            "safe": False,
            "result": None,
            "validation": {
                "safe": False,
                "issues": ["Coordination agent could not start. Check that Ollama is running."],
                "warnings": [],
            },
        }

    def route_query(self, query):
        return "out_of_scope"


def create_app():
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
        static_url_path="/static",
    )
    app.config["UPLOAD_FOLDER"] = "data"

    initialize_database()

    app.data_agent = DataAccessAgent()

    try:
        app.rag_agent = RAGAgent()
    except Exception:
        app.rag_agent = _FallbackRAGAgent()

    try:
        app.coordination_agent = CoordinationAgent()
    except Exception:
        app.coordination_agent = _FallbackCoordinationAgent()

    from .routes import main
    app.register_blueprint(main)

    return app
