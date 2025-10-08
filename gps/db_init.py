# db_init.py
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime
import os

Base = declarative_base()

class Coord(Base):
    __tablename__ = "coords"
    id = Column(Integer, primary_key=True, autoincrement=True)
    lat = Column(Float, nullable=False)
    lng = Column(Float, nullable=False)
    meta = Column(String, default="")
    ts = Column(DateTime, default=datetime.utcnow)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    engine = create_engine("sqlite:///data/coords.db", connect_args={"check_same_thread": False}, echo=False)
    Base.metadata.create_all(engine)
    print("Database created (or already exists) at data/coords.db")
