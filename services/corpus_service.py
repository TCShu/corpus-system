from sqlalchemy.orm import Session
from database.models import Corpus


def create_corpus(
    db: Session,
    user_id: int,
    corpus_name: str,
    description: str = None,
    language: str = None,
    source_type: str = None,
    file_path: str = None,
) -> Corpus:
    corpus = Corpus(
        user_id=user_id,
        corpus_name=corpus_name,
        description=description,
        language=language,
        source_type=source_type,
        file_path=file_path,
    )
    db.add(corpus)
    db.commit()
    db.refresh(corpus)
    return corpus


def get_corpus_by_id(db: Session, corpus_id: int) -> Corpus | None:
    return db.query(Corpus).filter(Corpus.corpus_id == corpus_id).first()


def list_corpora_by_user(db: Session, user_id: int) -> list[Corpus]:
    return db.query(Corpus).filter(Corpus.user_id == user_id).all()


def update_corpus(db: Session, corpus_id: int, **kwargs) -> Corpus | None:
    corpus = get_corpus_by_id(db, corpus_id)
    if not corpus:
        return None
    for key, value in kwargs.items():
        setattr(corpus, key, value)
    db.commit()
    db.refresh(corpus)
    return corpus


def delete_corpus(db: Session, corpus_id: int) -> bool:
    corpus = get_corpus_by_id(db, corpus_id)
    if not corpus:
        return False
    db.delete(corpus)
    db.commit()
    return True
