"""
WildVision — FastAPI Backend
Provides REST API endpoints for authentication, YOLO detection, and AI chat.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import uuid

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

# ─── Simple Auth ─────────────────────────────────────────────────────────────

VALID_USERS = {
    "admin@wildvision.com": "wild123",
    "user@wildvision.com": "user123",
}

# In-memory session tokens
active_tokens = {}


class LoginRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    animal_name: str
    question: str


@app.post("/api/login")
async def login(request: LoginRequest):
    if request.email in VALID_USERS and VALID_USERS[request.email] == request.password:
        token = str(uuid.uuid4())
        active_tokens[token] = request.email
        return {
            "success": True,
            "token": token,
            "message": "Welcome to WildVision!"
        }
    raise HTTPException(status_code=401, detail="Invalid email or password")


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


# ─── Health Check ────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "WildVision API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
