import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# Gunakan SQLite lokal (bisa diubah ke PostgreSQL di config nanti)
SQLALCHEMY_DATABASE_URL = "sqlite:///./data/trading.db"

# Pastikan folder "data" ada sebelum membuat database
os.makedirs(os.path.dirname(SQLALCHEMY_DATABASE_URL.replace("sqlite:///", "")), exist_ok=True)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()