from .db_manager import DatabaseManager
from .config import settings


def initialize_database():
    db = DatabaseManager(db_url=settings.DATABASE_URL)
    db.create_tables()
    print("Database tables created successfully.")


if __name__ == "__main__":
    initialize_database()
