"""
WildVision — FastAPI Backend
Provides REST API endpoints for authentication, YOLO detection, and AI chat.
Uses SQLite for persistent user storage.
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

# Configure logging so provider selection is visible
logging.basicConfig(level=logging.INFO, format="%(levelname)s:  %(message)s")

from yolo_service import detect_animal
from chatbot_service import get_animal_info, answer_question

app = FastAPI(title="WildVision API", version="1.0.0")

# Enable CORS for all origins (needed for Android app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── SQLite Database ─────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")


def get_db():
    """Get a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str) -> str:
    """Hash a password with SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def init_db():
    """Initialize the database and seed default users."""
    conn = get_db()
    cursor = conn.cursor()

    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Seed default users (only if they don't exist)
    default_users = [
        ("admin@wildvision.com", "wild123"),
        ("user@wildvision.com", "user123"),
    ]
    for email, password in default_users:
        try:
            cursor.execute(
                "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                (email, hash_password(password))
            )
        except sqlite3.IntegrityError:
            pass  # User already exists

    conn.commit()
    conn.close()


# Initialize DB on startup
init_db()

# In-memory session tokens
active_tokens = {}


# ─── Request/Response Models ─────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    animal_name: str
    question: str


# ─── Auth Endpoints ──────────────────────────────────────────────────────────

@app.post("/api/login")
async def login(request: LoginRequest):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM users WHERE email = ? AND password_hash = ?",
        (request.email, hash_password(request.password))
    )
    user = cursor.fetchone()
    conn.close()

    if user:
        token = str(uuid.uuid4())
        active_tokens[token] = request.email
        return {
            "success": True,
            "token": token,
            "message": "Welcome to WildVision!"
        }
    raise HTTPException(status_code=401, detail="Invalid email or password")


@app.post("/api/register")
async def register(request: RegisterRequest):
    # Validate email format (basic check)
    if not request.email or "@" not in request.email:
        raise HTTPException(status_code=400, detail="Invalid email address")

    # Validate password length
    if not request.password or len(request.password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters")

    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (request.email, hash_password(request.password))
        )
        conn.commit()
        conn.close()

        # Auto-login after registration
        token = str(uuid.uuid4())
        active_tokens[token] = request.email
        return {
            "success": True,
            "token": token,
            "message": "Account created! Welcome to WildVision!"
        }
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=409, detail="Email already registered")


# ─── Detection ───────────────────────────────────────────────────────────────

@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    # Save uploaded file
    file_ext = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
    unique_name = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run YOLO detection
    result = detect_animal(file_path)

    # Get AI info about the detected animal
    info = get_animal_info(result["name"])

    # Cleanup uploaded file
    try:
        os.remove(file_path)
    except OSError:
        pass

    return {
        "detection": result["name"],
        "confidence": result["confidence"],
        "info": info,
        "all_detections": result.get("all_detections", [])
    }


# ─── Chat ────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(request: ChatRequest):
    response = answer_question(request.animal_name, request.question)
    return {
        "response": response
    }


# ─── View Users (Admin) ──────────────────────────────────────────────────────

@app.get("/api/users")
async def list_users():
    """View all registered users (no passwords shown)."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, created_at FROM users ORDER BY created_at DESC")
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return {"total": len(users), "users": users}


# ─── Health Check ────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "WildVision API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
