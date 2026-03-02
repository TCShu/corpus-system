import os
from werkzeug.utils import secure_filename
from flask import Blueprint, request, jsonify, current_app

UPLOAD_FOLDER = "data"

main = Blueprint("main", __name__)

@main.route("/")
def home():
    return "ACAS RAG server is running."

@main.route("/ask", methods=["POST"])
def ask():
    question = request.json["question"]
    corpus_id = request.json["corpus_id"]

    answer = current_app.rag_agent.query(question, corpus_id)
    return jsonify({"answer": answer})

@main.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Process file
    clean_text = current_app.data_agent.get_clean_text_from_file(filepath)
    corpus_id = filename  # temporary ID based on filename
    current_app.rag_agent.add_document(clean_text, corpus_id)

    return jsonify({"message": "File uploaded and indexed successfully"})