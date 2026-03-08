from sqlalchemy.orm import Session
from database.models import DocumentChunk


def create_chunk(
    db: Session,
    document_id: int,
    chunk_text: str,
    chunk_index: int,
    token_count: int = None,
) -> DocumentChunk:
    chunk = DocumentChunk(
        document_id=document_id,
        chunk_text=chunk_text,
        chunk_index=chunk_index,
        token_count=token_count,
    )
    db.add(chunk)
    db.commit()
    db.refresh(chunk)
    return chunk


def bulk_create_chunks(
    db: Session, document_id: int, entries: list[dict]
) -> list[DocumentChunk]:
    records = [DocumentChunk(document_id=document_id, **e) for e in entries]
    db.add_all(records)
    db.commit()
    return records


def get_chunk_by_id(db: Session, chunk_id: int) -> DocumentChunk | None:
    return db.query(DocumentChunk).filter(DocumentChunk.chunk_id == chunk_id).first()


def list_chunks_by_document(db: Session, document_id: int) -> list[DocumentChunk]:
    return (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
        .all()
    )


def delete_chunk(db: Session, chunk_id: int) -> bool:
    chunk = get_chunk_by_id(db, chunk_id)
    if not chunk:
        return False
    db.delete(chunk)
    db.commit()
    return True
