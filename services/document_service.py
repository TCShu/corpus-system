from sqlalchemy.orm import Session
from database.models import Document


def create_document(
    db: Session,
    corpus_id: int,
    text_content: str,
    title: str = None,
    author: str = None,
    date_published=None,
    preprocessed_text: str = None,
    word_count: int = None,
    doc_metadata: dict = None,
) -> Document:
    doc = Document(
        corpus_id=corpus_id,
        title=title,
        author=author,
        date_published=date_published,
        text_content=text_content,
        preprocessed_text=preprocessed_text,
        word_count=word_count,
        doc_metadata=doc_metadata,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def get_document_by_id(db: Session, document_id: int) -> Document | None:
    return db.query(Document).filter(Document.document_id == document_id).first()


def list_documents_by_corpus(db: Session, corpus_id: int) -> list[Document]:
    return db.query(Document).filter(Document.corpus_id == corpus_id).all()


def update_document(db: Session, document_id: int, **kwargs) -> Document | None:
    doc = get_document_by_id(db, document_id)
    if not doc:
        return None
    for key, value in kwargs.items():
        setattr(doc, key, value)
    db.commit()
    db.refresh(doc)
    return doc


def delete_document(db: Session, document_id: int) -> bool:
    doc = get_document_by_id(db, document_id)
    if not doc:
        return False
    db.delete(doc)
    db.commit()
    return True
