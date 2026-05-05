"""
WildVision — Chatbot Service (Dual Provider)
Uses Ollama locally for development, falls back to Google Gemini for cloud deployment.
Provider is selected via CHAT_PROVIDER env var or auto-detected.
"""

import os
import json
import requests
import logging

logger = logging.getLogger(__name__)

# ─── Provider Configuration ──────────────────────────────────────────────────

CHAT_PROVIDER = os.environ.get("CHAT_PROVIDER", "auto")  # "ollama", "gemini", or "auto"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBsXc5R-Z-oIsxqyi2pkAUIhPkasvYqe6s")


# ─── Ollama Provider ─────────────────────────────────────────────────────────

class OllamaProvider:
    """Chat provider using a local Ollama instance."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is accessible."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def generate(self, prompt: str) -> str:
        """Generate a response from Ollama."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 300,  # Keep responses concise
            },
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,  # Local models can be slow on first load
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "No response generated.")
        except requests.Timeout:
            return "The local AI model timed out. Please try again."
        except requests.ConnectionError:
            return "Cannot connect to Ollama. Make sure it is running (ollama serve)."
        except Exception as e:
            return f"Ollama error: {str(e)}"


# ─── Gemini Provider ─────────────────────────────────────────────────────────

class GeminiProvider:
    """Chat provider using Google Gemini API (cloud)."""

    def __init__(self, api_key: str = GEMINI_API_KEY):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-lite")

    def is_available(self) -> bool:
        """Gemini is available if we have an API key."""
        return bool(GEMINI_API_KEY)

    def generate(self, prompt: str) -> str:
        """Generate a response from Gemini."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Gemini error: {str(e)}"


# ─── Provider Selection ──────────────────────────────────────────────────────

_provider = None


def _get_provider():
    """Select and cache the AI provider based on config / availability."""
    global _provider
    if _provider is not None:
        return _provider

    if CHAT_PROVIDER == "ollama":
        logger.info("🦙 CHAT_PROVIDER=ollama → Using Ollama")
        _provider = OllamaProvider()

    elif CHAT_PROVIDER == "gemini":
        logger.info("✨ CHAT_PROVIDER=gemini → Using Gemini")
        _provider = GeminiProvider()

    else:  # "auto" — try Ollama first, fallback to Gemini
        ollama = OllamaProvider()
        if ollama.is_available():
            logger.info("🦙 Auto-detected Ollama running locally → Using Ollama")
            _provider = ollama
        else:
            logger.info("✨ Ollama not available → Falling back to Gemini")
            _provider = GeminiProvider()

    return _provider


# ─── Public API (same interface as before) ────────────────────────────────────

def get_animal_info(animal_name: str) -> str:
    """
    Generate an informative description about the detected animal.
    Works identically whether using Ollama or Gemini.
    """
    if animal_name in ("Unknown", "Error"):
        return "I couldn't identify the animal in the image. Please try again with a clearer picture."

    prompt = (
        f"You are WildVision AI, an expert wildlife assistant. "
        f"The user just captured a photo identified as a '{animal_name}'. "
        f"Provide a brief, fascinating description including:\n"
        f"- Common name and scientific name\n"
        f"- Key physical traits\n"
        f"- Habitat and behavior\n"
        f"- 2 interesting facts\n"
        f"Keep it concise and engaging, under 200 words."
    )

    provider = _get_provider()
    return provider.generate(prompt)


def answer_question(animal_name: str, question: str) -> str:
    """
    Answer a follow-up question about the detected animal.
    Works identically whether using Ollama or Gemini.
    """
    if not question or not question.strip():
        return "Please ask a question about the animal."

    prompt = (
        f"You are WildVision AI, an expert wildlife assistant. "
        f"The user is asking about a '{animal_name}' they photographed. "
        f"Their question is: '{question}'\n\n"
        f"Provide a helpful, accurate, and concise answer (under 150 words). "
        f"If the question is unrelated to the animal, politely redirect."
    )

    provider = _get_provider()
    return provider.generate(prompt)
