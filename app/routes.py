import os
from datetime import datetime, timezone
from werkzeug.utils import secure_filename
from flask import Blueprint, current_app, jsonify, render_template, request
from sqlalchemy import select

from database.db import SessionLocal
from database.models import (
    AgentExecutionLog,
    AnalysisResult,
    Corpus,
    Document,
    FrequencyAnalysis,
    KWICEntry,
    Ngram,
    Query,
    User,
)

UPLOAD_FOLDER = "data"
CORPUS_CACHE: dict[str, str] = {}

main = Blueprint("main", __name__)

@main.route("/")
def home():
    return render_template("index.html")

@main.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@main.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "Invalid filename"}), 400

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    clean_text = current_app.data_agent.get_clean_text_from_file(filepath)
    corpus_id = filename
    CORPUS_CACHE[corpus_id] = clean_text
    rag_warning = None
    try:
        current_app.rag_agent.add_document(clean_text, corpus_id)
    except Exception as exc:
        rag_warning = f"RAG indexing skipped: {exc}"

    with SessionLocal() as db:
        owner = _get_or_create_default_user(db)
        corpus = _get_corpus_by_name(db, corpus_id)
        if not corpus:
            corpus = Corpus(
                user_id=owner.user_id,
                corpus_name=corpus_id,
                description=f"Uploaded file {filename}",
                language="en",
                source_type="file",
                file_path=filepath,
                status="preprocessed",
            )
            db.add(corpus)
            db.commit()
            db.refresh(corpus)

        document = Document(
            corpus_id=corpus.corpus_id,
            title=filename,
            text_content=clean_text,
            preprocessed_text=clean_text,
            word_count=len(current_app.data_agent.tokenize_text(clean_text)),
            doc_metadata={"uploaded_from": filepath},
        )
        db.add(document)
        corpus.total_documents = (corpus.total_documents or 0) + 1
        db.commit()

    return jsonify(
        {
            "message": "File uploaded and indexed successfully",
            "corpus_id": corpus_id,
            "rag_warning": rag_warning,
        }
    )

@main.route("/api/corpora", methods=["GET"])
def list_corpora():
    with SessionLocal() as db:
        corpora = db.execute(select(Corpus).order_by(Corpus.corpus_name.asc())).scalars().all()
    return jsonify({"corpora": [corpus.corpus_name for corpus in corpora]})

@main.route("/api/query", methods=["POST"])
def query():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "").strip()
    corpus_id = payload.get("corpus_id", "").strip()
    reference_corpus_id = payload.get("reference_corpus_id", "").strip()

    if not question:
        return jsonify({"error": "Question is required."}), 400
    if not corpus_id:
        return jsonify({"error": "corpus_id is required."}), 400
    with SessionLocal() as db:
        owner = _get_or_create_default_user(db)
        target_corpus = _get_corpus_by_name(db, corpus_id)
        if not target_corpus:
            return jsonify({"error": f"Unknown corpus_id '{corpus_id}'."}), 404

        corpus_text = CORPUS_CACHE.get(corpus_id, "")
        if not corpus_text:
            corpus_text = _load_latest_document_text(db, target_corpus.corpus_id)
            CORPUS_CACHE[corpus_id] = corpus_text
        tokens = current_app.data_agent.tokenize_text(corpus_text)

        reference_tokens = None
        if reference_corpus_id:
            reference_corpus = _get_corpus_by_name(db, reference_corpus_id)
            if not reference_corpus:
                return jsonify({"error": f"Unknown reference corpus '{reference_corpus_id}'."}), 404
            ref_text = CORPUS_CACHE.get(reference_corpus_id, "")
            if not ref_text:
                ref_text = _load_latest_document_text(db, reference_corpus.corpus_id)
                CORPUS_CACHE[reference_corpus_id] = ref_text
            reference_tokens = current_app.data_agent.tokenize_text(ref_text)

        query_row = Query(
            user_id=owner.user_id,
            corpus_id=target_corpus.corpus_id,
            query_text=question,
            query_type="linguistic_analysis",
            status="pending",
        )
        db.add(query_row)
        db.commit()
        db.refresh(query_row)

        execution_start = datetime.now(timezone.utc)
        orchestrated = current_app.coordination_agent.execute(
            query=question,
            tokens=tokens,
            reference_tokens=reference_tokens,
            corpus_text=corpus_text,
        )
        execution_end = datetime.now(timezone.utc)

        query_row.execution_time_ms = int((execution_end - execution_start).total_seconds() * 1000)
        query_row.status = "completed" if orchestrated["safe"] else "failed"

        result_payload = orchestrated.get("result")
        validation_payload = orchestrated.get("validation", {})
        analysis_result = AnalysisResult(
            query_id=query_row.query_id,
            agent_type=(result_payload or {}).get("analysis_type", "unknown"),
            result_data=result_payload,
            validated=validation_payload.get("safe", False),
            validation_report=validation_payload,
        )
        db.add(analysis_result)
        db.flush()
        if result_payload:
            _persist_analysis_specific_rows(db, analysis_result.result_id, result_payload)

        log_row = AgentExecutionLog(
            query_id=query_row.query_id,
            agent_name="coordinating_agent",
            execution_start=execution_start,
            execution_end=execution_end,
            success=orchestrated["safe"],
            error_message="; ".join(validation_payload.get("issues", []))
            if not orchestrated["safe"]
            else None,
            docker_container_id="n/a",
        )
        db.add(log_row)
        db.commit()

        if not orchestrated["safe"]:
            return jsonify(_build_client_error_response(orchestrated)), 400
        return jsonify(_build_client_success_response(orchestrated))

@main.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question")
    corpus_id = payload.get("corpus_id")
    if not question or not corpus_id:
        return jsonify({"error": "question and corpus_id are required"}), 400
    try:
        answer = current_app.rag_agent.query(question, corpus_id)
    except Exception as exc:
        answer = f"RAG query unavailable: {exc}"
    return jsonify({"answer": answer})

@main.route("/upload", methods=["POST"])
def upload_legacy():
    return upload()


def _get_or_create_default_user(db):
    user = db.execute(select(User).where(User.email == "local-user@acas.local")).scalar_one_or_none()
    if user:
        return user
    user = User(
        username="local_user",
        email="local-user@acas.local",
        password_hash="local-only-placeholder",
        role="student",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _get_corpus_by_name(db, corpus_name: str):
    return db.execute(select(Corpus).where(Corpus.corpus_name == corpus_name)).scalar_one_or_none()


def _load_latest_document_text(db, corpus_id: int) -> str:
    document = (
        db.execute(
            select(Document)
            .where(Document.corpus_id == corpus_id)
            .order_by(Document.document_id.desc())
        )
        .scalars()
        .first()
    )
    return document.preprocessed_text or document.text_content if document else ""


def _persist_analysis_specific_rows(db, result_id: int, result_payload: dict) -> None:
    analysis_type = result_payload.get("analysis_type")
    if analysis_type == "frequency":
        for row in result_payload.get("rows", []):
            db.add(
                FrequencyAnalysis(
                    result_id=result_id,
                    word=row.get("word"),
                    frequency=row.get("frequency"),
                    relative_frequency=row.get("relative_frequency"),
                    rank=row.get("rank"),
                )
            )
    if analysis_type == "kwic":
        for row in result_payload.get("matches", []):
            db.add(
                KWICEntry(
                    result_id=result_id,
                    document_id=None,
                    keyword=row.get("keyword"),
                    left_context=row.get("left_context"),
                    right_context=row.get("right_context"),
                    position_in_document=row.get("position"),
                )
            )
    if analysis_type == "ngram_collocation":
        for row in result_payload.get("rows", []):
            db.add(
                Ngram(
                    result_id=result_id,
                    ngram_text=row.get("ngram_text"),
                    ngram_size=row.get("ngram_size"),
                    frequency=row.get("frequency"),
                    pmi_score=row.get("pmi_score"),
                    dice_coefficient=None,
                )
            )


def _build_client_success_response(orchestrated: dict) -> dict:
    result = orchestrated.get("result") or {}
    return {
        "safe": True,
        "result": result,
    }


def _build_client_error_response(orchestrated: dict) -> dict:
    validation = orchestrated.get("validation", {})
    issues = validation.get("issues", [])
    message = (
        "This result was blocked because it could not be safely validated for display."
    )
    if issues and "out of acas scope" in issues[0].lower():
        message = "That request is outside the scope of corpus linguistic analysis supported by ACAS."
    return {
        "safe": False,
        "error": message,
    }
