from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base

class DatabaseManager:

    def __init__(self, db_url="sqlite:///corpus_system.db"):
        self.engine = create_engine(db_url, echo=False)
        self.Session = sessionmaker(bind=self.engine)

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def get_session(self):
        return self.Session()

    def close_session(self, session):
        session.close()