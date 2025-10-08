# app.py — Stage 2 with API key check
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()
app = Flask(__name__)

MAPS_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
POST_API_KEY = os.getenv("POST_API_KEY", None)  # our shared secret for POST

# --- Database setup ---
DB_URL = "sqlite:///data/coords.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False}, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Coord(Base):
    __tablename__ = "coords"
    id = Column(Integer, primary_key=True, autoincrement=True)
    lat = Column(Float, nullable=False)
    lng = Column(Float, nullable=False)
    meta = Column(String, default="")
    ts = Column(DateTime, default=datetime.utcnow)

# Ensure tables exist
Base.metadata.create_all(bind=engine)

def get_session():
    return SessionLocal()

@app.route("/")
def index():
    return render_template("index.html", maps_api_key=MAPS_KEY)

@app.route("/api/coords", methods=["GET"])
def get_coords():
    after_id = request.args.get("after_id", default=0, type=int)
    db = get_session()
    try:
        rows = db.query(Coord).filter(Coord.id > after_id).order_by(Coord.id.asc()).all()
        result = []
        for r in rows:
            result.append({
                "id": r.id,
                "lat": r.lat,
                "lng": r.lng,
                "meta": r.meta or "",
                # ISO-like string in UTC (no microseconds)
                "ts": r.ts.strftime("%Y-%m-%dT%H:%M:%SZ") if r.ts else None
            })
        return jsonify(result)
    finally:
        db.close()

@app.route("/api/coords", methods=["POST"])
def post_coords():
    # API key check:
    if POST_API_KEY:
        header_key = request.headers.get("X-API-KEY")
        if not header_key or header_key != POST_API_KEY:
            return jsonify({"error": "Unauthorized - missing or invalid X-API-KEY"}), 401

    data = request.get_json(force=True)
    lat = data.get("lat")
    lng = data.get("lng")
    meta = data.get("meta", "")

    # Basic validation
    try:
        lat = float(lat)
        lng = float(lng)
    except (TypeError, ValueError):
        return jsonify({"error": "lat and lng must be numbers"}), 400
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        return jsonify({"error": "lat or lng out of range"}), 400

    db = get_session()
    try:
        coord = Coord(lat=lat, lng=lng, meta=str(meta))
        db.add(coord)
        db.commit()
        db.refresh(coord)  # ensure id is populated
        return jsonify({"status": "ok", "id": coord.id})
    finally:
        db.close()

if __name__ == "__main__":
    app.run(debug=True)
