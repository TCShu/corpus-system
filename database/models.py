from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, JSON, Float
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()

# -----------------------
# Users
# -----------------------
class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True)
    username = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="student")
    created_at = Column(DateTime, default=datetime.utcnow)

    corpora = relationship("Corpus", back_populates="owner")


# -----------------------
# Corpora
# -----------------------
class Corpus(Base):
    __tablename__ = "corpora"

    corpus_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    corpus_name = Column(String(255), nullable=False)
    description = Column(Text)
    language = Column(String(50))
    source_type = Column(String(50))
    file_path = Column(Text)
    upload_date = Column(DateTime, default=datetime.utcnow)
    total_documents = Column(Integer, default=0)
    status = Column(String(50), default="raw")

    owner = relationship("User", back_populates="corpora")
    documents = relationship("Document", back_populates="corpus")


# -----------------------
# Documents
# -----------------------
class Document(Base):
    __tablename__ = "documents"

    document_id = Column(Integer, primary_key=True)
    corpus_id = Column(Integer, ForeignKey("corpora.corpus_id"))

    title = Column(String(255))
    author = Column(String(255))
    date_published = Column(DateTime)

    text_content = Column(Text)
    preprocessed_text = Column(Text)

    word_count = Column(Integer)
    metadata = Column(JSON)

    corpus = relationship("Corpus", back_populates="documents")


# -----------------------
# Queries
# -----------------------
class Query(Base):
    __tablename__ = "queries"

    query_id = Column(Integer, primary_key=True)

    user_id = Column(Integer, ForeignKey("users.user_id"))
    corpus_id = Column(Integer, ForeignKey("corpora.corpus_id"))

    query_text = Column(Text)
    query_type = Column(String(100))

    execution_time_ms = Column(Integer)
    status = Column(String(50), default="pending")

    timestamp = Column(DateTime, default=datetime.utcnow)


# -----------------------
# Analysis Results
# -----------------------
class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    result_id = Column(Integer, primary_key=True)

    query_id = Column(Integer, ForeignKey("queries.query_id"))
    agent_type = Column(String(100))

    result_data = Column(JSON)

    validated = Column(Boolean, default=False)
    validation_report = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)


# -----------------------
# Agent Execution Log
# -----------------------
class AgentExecutionLog(Base):
    __tablename__ = "agent_execution_logs"

    log_id = Column(Integer, primary_key=True)

    query_id = Column(Integer, ForeignKey("queries.query_id"))
    agent_name = Column(String(100))

    execution_start = Column(DateTime)
    execution_end = Column(DateTime)

    success = Column(Boolean)
    error_message = Column(Text)

    docker_container_id = Column(String(255))


# -----------------------
# Frequency Analysis
# -----------------------
class FrequencyAnalysis(Base):
    __tablename__ = "frequency_analysis"

    frequency_id = Column(Integer, primary_key=True)
    result_id = Column(Integer, ForeignKey("analysis_results.result_id"))

    word = Column(String(255))
    frequency = Column(Integer)
    relative_frequency = Column(Float)
    rank = Column(Integer)


# -----------------------
# KWIC Entries
# -----------------------
class KWICEntry(Base):
    __tablename__ = "kwic_entries"

    kwic_id = Column(Integer, primary_key=True)
    result_id = Column(Integer, ForeignKey("analysis_results.result_id"))
    document_id = Column(Integer, ForeignKey("documents.document_id"))

    keyword = Column(String(255))
    left_context = Column(Text)
    right_context = Column(Text)

    position_in_document = Column(Integer)


# -----------------------
# N-grams
# -----------------------
class Ngram(Base):
    __tablename__ = "ngrams"

    ngram_id = Column(Integer, primary_key=True)
    result_id = Column(Integer, ForeignKey("analysis_results.result_id"))

    ngram_text = Column(String(255))
    ngram_size = Column(Integer)

    frequency = Column(Integer)

    pmi_score = Column(Float)
    dice_coefficient = Column(Float)


# -----------------------
# Vector Embeddings
# -----------------------
class VectorEmbedding(Base):
    __tablename__ = "vector_embeddings"

    embedding_id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.document_id"))

    embedding_vector = Column(JSON)
    model_used = Column(String(100))

    created_at = Column(DateTime, default=datetime.utcnow)