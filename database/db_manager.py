from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base

class DatabaseManager:
    def __init__(self, db_url):
        engine_kwargs = {}
        if db_url.startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}

        self.engine = create_engine(db_url, **engine_kwargs)
        self.Session = sessionmaker(bind=self.engine)

    def get_session(self):
        return self.Session()

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def close_session(self, session):
        session.close()
