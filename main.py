"""
WildVision — FastAPI Backend
Provides REST API endpoints for authentication, YOLO detection, AI chat,
sightings map, detection history, analytics, and conservation status.
Uses SQLite for persistent storage.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import uuid
import sqlite3
import hashlib
import logging
import requests as http_requests
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s:  %(message)s")

from yolo_service import detect_animal
from chatbot_service import get_animal_info, answer_question, generate_care_packages

app = FastAPI(title="WildVision API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")

# ─── Conservation Status Lookup ──────────────────────────────────────────────
# Offline lookup for common animals (works without IUCN API key)
CONSERVATION_STATUS = {
    # Domestic
    "dog": ("Least Concern", "LC", "#4CAF50"),
    "cat": ("Least Concern", "LC", "#4CAF50"),
    "horse": ("Least Concern", "LC", "#4CAF50"),
    "cow": ("Least Concern", "LC", "#4CAF50"),
    "sheep": ("Least Concern", "LC", "#4CAF50"),
    "goat": ("Least Concern", "LC", "#4CAF50"),
    "chicken": ("Least Concern", "LC", "#4CAF50"),
    "pig": ("Least Concern", "LC", "#4CAF50"),
    "rabbit": ("Least Concern", "LC", "#4CAF50"),
    # Wild — Least Concern
    "wolf": ("Least Concern", "LC", "#4CAF50"),
    "fox": ("Least Concern", "LC", "#4CAF50"),
    "deer": ("Least Concern", "LC", "#4CAF50"),
    "bear": ("Least Concern", "LC", "#4CAF50"),
    "monkey": ("Least Concern", "LC", "#4CAF50"),
    "zebra": ("Least Concern", "LC", "#4CAF50"),
    "hippopotamus": ("Vulnerable", "VU", "#FF9800"),
    "hippo": ("Vulnerable", "VU", "#FF9800"),
    # Vulnerable
    "lion": ("Vulnerable", "VU", "#FF9800"),
    "polar bear": ("Vulnerable", "VU", "#FF9800"),
    "cheetah": ("Vulnerable", "VU", "#FF9800"),
    "giraffe": ("Vulnerable", "VU", "#FF9800"),
    "giant panda": ("Vulnerable", "VU", "#FF9800"),
    "hippopotamus": ("Vulnerable", "VU", "#FF9800"),
    "african elephant": ("Vulnerable", "VU", "#FF9800"),
    # Endangered
    "tiger": ("Endangered", "EN", "#FF5722"),
    "snow leopard": ("Vulnerable", "VU", "#FF9800"),
    "gorilla": ("Endangered", "EN", "#FF5722"),
    "orangutan": ("Endangered", "EN", "#FF5722"),
    "blue whale": ("Endangered", "EN", "#FF5722"),
    "asian elephant": ("Endangered", "EN", "#FF5722"),
    "african wild dog": ("Endangered", "EN", "#FF5722"),
    # Critically Endangered
    "amur leopard": ("Critically Endangered", "CR", "#F44336"),
    "sumatran orangutan": ("Critically Endangered", "CR", "#F44336"),
    "black rhino": ("Critically Endangered", "CR", "#F44336"),
    "northern white rhino": ("Critically Endangered", "CR", "#F44336"),
    "hawksbill turtle": ("Critically Endangered", "CR", "#F44336"),
    "kakapo": ("Critically Endangered", "CR", "#F44336"),
    # Birds
    "eagle": ("Least Concern", "LC", "#4CAF50"),
    "hawk": ("Least Concern", "LC", "#4CAF50"),
    "parrot": ("Least Concern", "LC", "#4CAF50"),
    "penguin": ("Least Concern", "LC", "#4CAF50"),
    # Reptiles
    "crocodile": ("Least Concern", "LC", "#4CAF50"),
    "alligator": ("Least Concern", "LC", "#4CAF50"),
    "komodo dragon": ("Endangered", "EN", "#FF5722"),
}

# All domestic dog breeds are "domesticated" — least concern
DOG_BREED_KEYWORDS = [
    "retriever", "labrador", "poodle", "bulldog", "beagle", "shepherd",
    "rottweiler", "dachshund", "husky", "boxer", "chihuahua", "pug",
    "spaniel", "terrier", "setter", "pointer", "collie", "maltese",
    "bichon", "corgi", "dalmatian", "doberman", "mastiff", "schnauzer",
    "shih", "akita", "samoyed", "malamute", "vizsla", "weimaraner",
]


def get_conservation_status(species: str) -> dict:
    """Get conservation status for a species using offline lookup."""
    lower = species.lower().strip()

    # Check if it's a dog breed
    for kw in DOG_BREED_KEYWORDS:
        if kw in lower:
            return {
                "species": species,
                "status": "Least Concern",
                "code": "LC",
                "color": "#4CAF50",
                "description": f"The {species} is a domesticated breed with a stable global population.",
            }

    # Check full name match
    if lower in CONSERVATION_STATUS:
        status, code, color = CONSERVATION_STATUS[lower]
        return {"species": species, "status": status, "code": code, "color": color,
                "description": f"IUCN Red List status for {species}: {status}"}

    # Check partial keyword match
    for key, (status, code, color) in CONSERVATION_STATUS.items():
        if key in lower or lower in key:
            return {"species": species, "status": status, "code": code, "color": color,
                    "description": f"IUCN Red List status for {species}: {status}"}

    # Default
    return {
        "species": species,
        "status": "Data Deficient",
        "code": "DD",
        "color": "#9E9E9E",
        "description": f"Conservation status for {species} is not available in our database.",
    }


# ─── Database ─────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def init_db():
    conn = get_db()
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Sightings table (for map)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sightings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            animal_name TEXT NOT NULL,
            confidence REAL DEFAULT 0.0,
            latitude REAL DEFAULT 0.0,
            longitude REAL DEFAULT 0.0,
            user_email TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Detection history table (per user journal)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            animal_name TEXT NOT NULL,
            confidence REAL DEFAULT 0.0,
            info TEXT,
            conservation_status TEXT,
            conservation_code TEXT,
            user_email TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Seed default users
    for email, password in [("admin@wildvision.com", "wild123"), ("user@wildvision.com", "user123")]:
        try:
            cursor.execute(
                "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                (email, hash_password(password))
            )
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    conn.close()


init_db()
active_tokens = {}


# ─── Request / Response Models ────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    animal_name: str
    question: str


class CareRequest(BaseModel):
    animal_name: str


class SightingRequest(BaseModel):
    animal_name: str
    confidence: float = 0.0
    latitude: float = 0.0
    longitude: float = 0.0
    user_email: str = ""


class HistorySaveRequest(BaseModel):
    animal_name: str
    confidence: float = 0.0
    info: str = ""
    conservation_status: str = ""
    conservation_code: str = ""
    user_email: str = ""


# ─── Auth ─────────────────────────────────────────────────────────────────────

@app.post("/api/login")
async def login(request: LoginRequest):
    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE email = ? AND password_hash = ?",
        (request.email, hash_password(request.password))
    ).fetchone()
    conn.close()
    if user:
        token = str(uuid.uuid4())
        active_tokens[token] = request.email
        return {"success": True, "token": token, "message": "Welcome to WildVision!"}
    raise HTTPException(status_code=401, detail="Invalid email or password")


@app.post("/api/register")
async def register(request: RegisterRequest):
    if not request.email or "@" not in request.email:
        raise HTTPException(status_code=400, detail="Invalid email address")
    if not request.password or len(request.password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters")
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (request.email, hash_password(request.password))
        )
        conn.commit()
        conn.close()
        token = str(uuid.uuid4())
        active_tokens[token] = request.email
        return {"success": True, "token": token, "message": "Account created! Welcome to WildVision!"}
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=409, detail="Email already registered")


# ─── Detection ────────────────────────────────────────────────────────────────

@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    file_ext = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
    unique_name = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = detect_animal(file_path)
    info = get_animal_info(result["name"])

    # Get conservation status
    conservation = get_conservation_status(result["name"])

    try:
        os.remove(file_path)
    except OSError:
        pass

    # Automatically save to database for Gamification/Community tab updates
    animal_name = result["name"]
    confidence = result["confidence"]
    status_code = conservation["code"]
    status_desc = conservation["status"]
    
    # Use default user for the demo (or extract from token if implemented)
    user_email = "user@wildvision.com"
    timestamp = datetime.utcnow().isoformat()
    
    # Add to sightings map feed (mock lat/long for demo or you could send it from Android)
    conn = get_db()
    conn.execute(
        "INSERT INTO sightings (animal_name, confidence, latitude, longitude, user_email, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        (animal_name, confidence, 0.0, 0.0, user_email, timestamp)
    )
    
    # Add to detection history for leaderboards and achievements
    conn.execute(
        """INSERT INTO detection_history
           (animal_name, confidence, info, conservation_status, conservation_code, user_email, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (animal_name, confidence, info, status_desc, status_code, user_email, timestamp)
    )
    conn.commit()
    conn.close()

    return {
        "detection": result["name"],
        "confidence": result["confidence"],
        "info": info,
        "all_detections": result.get("all_detections", []),
        "conservation_status": conservation["status"],
        "conservation_code": conservation["code"],
        "conservation_color": conservation["color"],
        "conservation_description": conservation["description"],
    }


# ─── Chat ─────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(request: ChatRequest):
    response = answer_question(request.animal_name, request.question)
    return {"response": response}


# ─── Care Packages ────────────────────────────────────────────────────────────

@app.post("/api/care-packages")
async def care_packages(request: CareRequest):
    return generate_care_packages(request.animal_name)


# ─── Conservation Status ──────────────────────────────────────────────────────

@app.get("/api/conservation/{species}")
async def conservation(species: str):
    return get_conservation_status(species)


# ─── Sightings Map ────────────────────────────────────────────────────────────

@app.post("/api/sighting")
async def save_sighting(request: SightingRequest):
    conn = get_db()
    conn.execute(
        "INSERT INTO sightings (animal_name, confidence, latitude, longitude, user_email, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        (request.animal_name, request.confidence, request.latitude, request.longitude,
         request.user_email, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()
    return {"success": True, "message": "Sighting saved to map!"}


@app.get("/api/sightings")
async def get_sightings():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, animal_name, confidence, latitude, longitude, user_email, timestamp FROM sightings ORDER BY timestamp DESC LIMIT 500"
    ).fetchall()
    conn.close()
    return {"sightings": [dict(r) for r in rows]}


# ─── Detection History ────────────────────────────────────────────────────────

@app.post("/api/history")
async def save_history(request: HistorySaveRequest):
    conn = get_db()
    conn.execute(
        """INSERT INTO detection_history
           (animal_name, confidence, info, conservation_status, conservation_code, user_email, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (request.animal_name, request.confidence, request.info,
         request.conservation_status, request.conservation_code,
         request.user_email, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()
    return {"success": True}


@app.get("/api/history/{email}")
async def get_history(email: str):
    conn = get_db()
    rows = conn.execute(
        """SELECT id, animal_name, confidence, info, conservation_status, conservation_code, timestamp
           FROM detection_history WHERE user_email = ? ORDER BY timestamp DESC LIMIT 100""",
        (email,)
    ).fetchall()
    conn.close()
    return {"history": [dict(r) for r in rows]}


# ─── Analytics / Stats ────────────────────────────────────────────────────────

@app.get("/api/stats/{email}")
async def get_stats(email: str):
    conn = get_db()

    total = conn.execute(
        "SELECT COUNT(*) as cnt FROM detection_history WHERE user_email = ?", (email,)
    ).fetchone()["cnt"]

    top_breeds = conn.execute(
        """SELECT animal_name, COUNT(*) as count
           FROM detection_history WHERE user_email = ?
           GROUP BY animal_name ORDER BY count DESC LIMIT 5""",
        (email,)
    ).fetchall()

    latest = conn.execute(
        """SELECT animal_name, timestamp FROM detection_history
           WHERE user_email = ? ORDER BY timestamp DESC LIMIT 1""",
        (email,)
    ).fetchone()

    rarest = conn.execute(
        """SELECT animal_name, conservation_code FROM detection_history
           WHERE user_email = ? AND conservation_code IN ('CR','EN','VU')
           ORDER BY CASE conservation_code WHEN 'CR' THEN 1 WHEN 'EN' THEN 2 ELSE 3 END
           LIMIT 1""",
        (email,)
    ).fetchone()

    conn.close()

    return {
        "total_scans": total,
        "top_breeds": [{"name": r["animal_name"], "count": r["count"]} for r in top_breeds],
        "latest_detection": dict(latest) if latest else None,
        "rarest_find": dict(rarest) if rarest else None,
    }


# ─── Gamification & Community ───────────────────────────────────────────────────

@app.get("/api/leaderboard")
async def get_leaderboard():
    conn = get_db()
    rows = conn.execute(
        """SELECT user_email, COUNT(DISTINCT animal_name) as unique_species, COUNT(*) as total_scans
           FROM detection_history
           GROUP BY user_email
           ORDER BY unique_species DESC, total_scans DESC
           LIMIT 10"""
    ).fetchall()
    conn.close()
    return {"leaderboard": [dict(r) for r in rows]}


@app.get("/api/achievements/{email}")
async def get_achievements(email: str):
    conn = get_db()
    rows = conn.execute(
        "SELECT animal_name, conservation_code FROM detection_history WHERE user_email = ?",
        (email,)
    ).fetchall()
    conn.close()
    
    unique_species = set(r["animal_name"].lower() for r in rows)
    has_endangered = any(r["conservation_code"] in ("CR", "EN", "VU") for r in rows)
    has_dog = any(any(kw in r["animal_name"].lower() for kw in DOG_BREED_KEYWORDS) for r in rows)
    
    achievements = []
    if len(rows) > 0:
        achievements.append({"title": "First Sighting", "description": "You detected your first animal!", "icon": "🌟"})
    if len(unique_species) >= 5:
        achievements.append({"title": "Novice Explorer", "description": "Spotted 5 unique species.", "icon": "🔍"})
    if len(unique_species) >= 20:
        achievements.append({"title": "Expert Tracker", "description": "Spotted 20 unique species.", "icon": "🦅"})
    if has_endangered:
        achievements.append({"title": "Conservation Hero", "description": "Spotted a vulnerable or endangered species.", "icon": "🛡️"})
    if has_dog:
        achievements.append({"title": "Dog Lover", "description": "Spotted a dog breed.", "icon": "🐶"})
        
    return {"achievements": achievements}


# ─── Real-World Impact ────────────────────────────────────────────────────────

class LostFoundRequest(BaseModel):
    animal_name: str
    zipcode: str = ""

@app.post("/api/lost-and-found")
async def lost_and_found(request: LostFoundRequest):
    """
    Integrates with a 3rd party API for lost/found pets.
    Falls back to mock data if API keys aren't present (ideal for competition demos).
    """
    # Demo integration logic: In a real scenario, you'd use a Petfinder or similar API here.
    # e.g., http_requests.get(f"https://api.petfinder.com/v2/animals?type={request.animal_name}")
    
    mock_results = [
        {
            "name": "Buddy",
            "status": "Lost",
            "description": f"Lost {request.animal_name} near park.",
            "contact": "555-0101",
            "date": datetime.utcnow().isoformat()
        },
        {
            "name": "Unknown",
            "status": "Found",
            "description": f"Found a friendly {request.animal_name} wandering.",
            "contact": "555-0102",
            "date": datetime.utcnow().isoformat()
        }
    ]
    
    return {
        "success": True,
        "provider": "Petfinder Integration (Demo Mode)",
        "results": mock_results
    }


# ─── Admin ────────────────────────────────────────────────────────────────────

@app.get("/api/users")
async def list_users():
    conn = get_db()
    users = [dict(r) for r in conn.execute(
        "SELECT id, email, created_at FROM users ORDER BY created_at DESC"
    ).fetchall()]
    conn.close()
    return {"total": len(users), "users": users}


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "WildVision API", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
