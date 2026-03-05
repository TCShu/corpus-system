from db_manager import DatabaseManager

def initialize_database():
    db = DatabaseManager()
    db.create_tables()
    print("Database tables created successfully.")

if __name__ == "__main__":
    initialize_database()