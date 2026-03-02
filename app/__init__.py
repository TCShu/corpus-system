from flask import Flask
from agents.data_access_agent import DataAccessAgent
from agents.rag_agent import RAGAgent

def create_app():
    app = Flask(__name__)

    data_agent = DataAccessAgent()
    rag_agent = RAGAgent()

    # Save to app context
    app.data_agent = data_agent
    app.rag_agent = rag_agent

    from .routes import main
    app.register_blueprint(main)

    return app