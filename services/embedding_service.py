from sqlalchemy.orm import Session
from database.models import VectorEmbedding


def create_embedding(
    db: Session,
    chunk_id: int,
    embedding_vector: list[float],
    model_used: str,
) -> VectorEmbedding:
    embedding = VectorEmbedding(
        chunk_id=chunk_id,
        embedding_vector=embedding_vector,
        model_used=model_used,
    )
    db.add(embedding)
    db.commit()
    db.refresh(embedding)
    return embedding


def get_embedding_by_id(db: Session, embedding_id: int) -> VectorEmbedding | None:
    return (
        db.query(VectorEmbedding)
        .filter(VectorEmbedding.embedding_id == embedding_id)
        .first()
    )


def list_embeddings_by_chunk(db: Session, chunk_id: int) -> list[VectorEmbedding]:
    return (
        db.query(VectorEmbedding)
        .filter(VectorEmbedding.chunk_id == chunk_id)
        .all()
    )


def delete_embedding(db: Session, embedding_id: int) -> bool:
    embedding = get_embedding_by_id(db, embedding_id)
    if not embedding:
        return False
    db.delete(embedding)
    db.commit()
    return True
