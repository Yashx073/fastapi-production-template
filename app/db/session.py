from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://fraud:fraud@localhost:5432/frauddb"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit = False, autoflush = False, bind = engine)