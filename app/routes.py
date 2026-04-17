import os
from datetime import datetime, timezone
from werkzeug.utils import secure_filename
from flask import Blueprint, current_app, jsonify, render_template, request
from sqlalchemy import select

from database.db import SessionLocal
from database.models import (
    AgentExecutionLog,
    AnalysisResult,
    Conversation,
    ConversationMessage,
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


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@main.route("/")
def home():
    return render_template("index.html")


@main.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

@main.route("/api/models", methods=["GET"])
def list_models():
    from agents._shared import AVAILABLE_MODELS, get_model
    return jsonify({"models": AVAILABLE_MODELS, "current": get_model()})


@main.route("/api/models/select", methods=["POST"])
def select_model():
    from agents._shared import AVAILABLE_MODELS, set_model
    from agents.coordination_agent import CoordinationAgent
    from agents.rag_agent import RAGAgent
    payload = request.get_json(silent=True) or {}
    model = payload.get("model", "").strip()
    if not model or model not in AVAILABLE_MODELS:
        return jsonify({"error": f"Invalid model. Choose from: {AVAILABLE_MODELS}"}), 400
    set_model(model)
    current_app.rag_agent = RAGAgent(model=model)
    current_app.coordination_agent = CoordinationAgent()
    return jsonify({"message": f"Model switched to {model}. All agents reloaded."})


# ---------------------------------------------------------------------------
# Corpus management
# ---------------------------------------------------------------------------

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
        corpora = (
            db.execute(select(Corpus).order_by(Corpus.corpus_name.asc()))
            .scalars()
            .all()
        )
        result = [
            {
                "name": c.corpus_name,
                "document_count": c.total_documents or 0,
                "upload_date": c.upload_date.isoformat() if c.upload_date else None,
            }
            for c in corpora
        ]
    return jsonify({"corpora": result})


# ---------------------------------------------------------------------------
# Conversation management
# ---------------------------------------------------------------------------

@main.route("/api/conversations", methods=["GET"])
def list_conversations():
    with SessionLocal() as db:
        owner = _get_or_create_default_user(db)
        convs = (
            db.execute(
                select(Conversation)
                .where(Conversation.user_id == owner.user_id)
                .order_by(Conversation.updated_at.desc())
                .limit(50)
            )
            .scalars()
            .all()
        )
        result = [
            {
                "conversation_id": c.conversation_id,
                "title": c.title,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "updated_at": c.updated_at.isoformat() if c.updated_at else None,
            }
            for c in convs
        ]
    return jsonify({"conversations": result})


@main.route("/api/conversations/<int:conv_id>/messages", methods=["GET"])
def get_conversation_messages(conv_id: int):
    with SessionLocal() as db:
        conv = db.get(Conversation, conv_id)
        if not conv:
            return jsonify({"error": "Conversation not found."}), 404
        msgs = [
            {
                "message_id": m.message_id,
                "role": m.role,
                "content": m.content,
                "analysis_type": m.analysis_type,
                "result_data": m.result_data,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in conv.messages
        ]
        return jsonify(
            {
                "conversation_id": conv.conversation_id,
                "title": conv.title,
                "messages": msgs,
            }
        )


# ---------------------------------------------------------------------------
# Main query endpoint
# ---------------------------------------------------------------------------

@main.route("/api/query", methods=["POST"])
def query():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "").strip()
    corpus_id = (payload.get("corpus_id") or "").strip()
    reference_corpus_id = (payload.get("reference_corpus_id") or "").strip()
    conversation_id = payload.get("conversation_id")

    if not question:
        return jsonify({"error": "Question is required."}), 400

    corpus_text = ""
    tokens: list[str] = []
    reference_tokens = None
    corpus_db_id = None

    with SessionLocal() as db:
        owner = _get_or_create_default_user(db)

        # --- Load target corpus (optional — conversational queries may skip this) ---
        if corpus_id:
            target_corpus = _get_corpus_by_name(db, corpus_id)
            if not target_corpus:
                return jsonify({"error": f"Unknown corpus '{corpus_id}'."}), 404
            corpus_db_id = target_corpus.corpus_id

            corpus_text = CORPUS_CACHE.get(corpus_id, "")
            if not corpus_text:
                corpus_text = _load_latest_document_text(db, target_corpus.corpus_id)
                CORPUS_CACHE[corpus_id] = corpus_text
            tokens = current_app.data_agent.tokenize_text(corpus_text)

        # --- Load reference corpus (optional) ---
        if reference_corpus_id:
            reference_corpus = _get_corpus_by_name(db, reference_corpus_id)
            if not reference_corpus:
                return jsonify({"error": f"Unknown reference corpus '{reference_corpus_id}'."}), 404
            ref_text = CORPUS_CACHE.get(reference_corpus_id, "")
            if not ref_text:
                ref_text = _load_latest_document_text(db, reference_corpus.corpus_id)
                CORPUS_CACHE[reference_corpus_id] = ref_text
            reference_tokens = current_app.data_agent.tokenize_text(ref_text)

        # --- Create or retrieve conversation ---
        conv: Conversation
        if conversation_id:
            conv = db.get(Conversation, int(conversation_id))
            if not conv:
                # Stale id — create fresh
                conv = Conversation(
                    user_id=owner.user_id,
                    title=question[:200],
                )
                db.add(conv)
                db.commit()
                db.refresh(conv)
        else:
            conv = Conversation(
                user_id=owner.user_id,
                title=question[:200],
            )
            db.add(conv)
            db.commit()
            db.refresh(conv)

        conv_id = conv.conversation_id

        # Save user message
        db.add(ConversationMessage(
            conversation_id=conv_id,
            role="user",
            content=question,
        ))

        # Create query log row
        query_row = Query(
            user_id=owner.user_id,
            corpus_id=corpus_db_id,
            query_text=question,
            query_type="linguistic_analysis",
            status="pending",
        )
        db.add(query_row)
        db.commit()
        db.refresh(query_row)

        # --- Run the agent pipeline ---
        execution_start = datetime.now(timezone.utc)
        orchestrated = current_app.coordination_agent.execute(
            query=question,
            tokens=tokens,
            reference_tokens=reference_tokens,
            corpus_text=corpus_text,
        )
        execution_end = datetime.now(timezone.utc)

        # --- Persist results ---
        query_row.execution_time_ms = int(
            (execution_end - execution_start).total_seconds() * 1000
        )
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

        db.add(AgentExecutionLog(
            query_id=query_row.query_id,
            agent_name="coordinating_agent",
            execution_start=execution_start,
            execution_end=execution_end,
            success=orchestrated["safe"],
            error_message=(
                "; ".join(validation_payload.get("issues", []))
                if not orchestrated["safe"] else None
            ),
            docker_container_id="n/a",
        ))

        # Save assistant message
        if orchestrated["safe"] and result_payload:
            analysis_type = result_payload.get("analysis_type", "unknown")
            if analysis_type == "conversational":
                assistant_content = result_payload.get("reply", "")
            else:
                assistant_content = f"{analysis_type} analysis complete."
            db.add(ConversationMessage(
                conversation_id=conv_id,
                role="assistant",
                content=assistant_content,
                analysis_type=analysis_type,
                result_data=result_payload,
            ))

        # Touch conversation timestamp
        conv.updated_at = datetime.now(timezone.utc)
        db.commit()

        if not orchestrated["safe"]:
            resp = _build_client_error_response(orchestrated)
            resp["conversation_id"] = conv_id
            return jsonify(resp), 400

        resp = _build_client_success_response(orchestrated)
        resp["conversation_id"] = conv_id
        return jsonify(resp)


# ---------------------------------------------------------------------------
# RAG endpoint (direct, bypasses coordination agent)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_or_create_default_user(db):
    user = db.execute(
        select(User).where(User.email == "local-user@acas.local")
    ).scalar_one_or_none()
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
    return db.execute(
        select(Corpus).where(Corpus.corpus_name == corpus_name)
    ).scalar_one_or_none()


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
    return (document.preprocessed_text or document.text_content) if document else ""


def _persist_analysis_specific_rows(db, result_id: int, result_payload: dict) -> None:
    analysis_type = result_payload.get("analysis_type")
    if analysis_type == "frequency":
        for row in result_payload.get("rows", []):
            db.add(FrequencyAnalysis(
                result_id=result_id,
                word=row.get("word"),
                frequency=row.get("frequency"),
                relative_frequency=row.get("relative_frequency"),
                rank=row.get("rank"),
            ))
    if analysis_type == "kwic":
        for row in result_payload.get("matches", []):
            db.add(KWICEntry(
                result_id=result_id,
                document_id=None,
                keyword=row.get("keyword"),
                left_context=row.get("left_context"),
                right_context=row.get("right_context"),
                position_in_document=row.get("position"),
            ))
    if analysis_type == "ngram_collocation":
        for row in result_payload.get("rows", []):
            db.add(Ngram(
                result_id=result_id,
                ngram_text=row.get("ngram_text"),
                ngram_size=row.get("ngram_size"),
                frequency=row.get("frequency"),
                pmi_score=row.get("pmi_score"),
                dice_coefficient=None,
            ))


def _build_client_success_response(orchestrated: dict) -> dict:
    result = orchestrated.get("result") or {}
    if result.get("analysis_type") == "conversational":
        return {
            "safe": True,
            "conversational": True,
            "reply": result.get("reply", ""),
            "result": result,
        }
    return {"safe": True, "result": result}


def _build_client_error_response(orchestrated: dict) -> dict:
    validation = orchestrated.get("validation", {})
    issues = validation.get("issues", [])
    if issues:
        return {"safe": False, "error": issues[0]}
    return {"safe": False, "error": "This analysis could not be safely validated for display."}
