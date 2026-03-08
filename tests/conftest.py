import os
import pytest
from dotenv import load_dotenv

load_dotenv()
os.environ["DATABASE_URL"] = "sqlite:///./tests/test.db"

from database.db_manager import DatabaseManager
from database.config import settings


@pytest.fixture(scope="session")
def db_session():
    manager = DatabaseManager(db_url=settings.DATABASE_URL)
    manager.create_tables()
    session = manager.get_session()
    yield session
    manager.close_session(session)
