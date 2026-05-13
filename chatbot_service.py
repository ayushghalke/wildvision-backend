"""
WildVision — Chatbot Service (Triple Provider)
Provider priority: Ollama (local dev) → Groq (fast cloud, free) → Gemini (ultimate fallback)
Provider is selected via CHAT_PROVIDER env var or auto-detected.
"""

import os
import json
import requests
import logging

# Load .env file if present (local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ─── Provider Configuration ──────────────────────────────────────────────────

CHAT_PROVIDER   = os.environ.get("CHAT_PROVIDER", "auto")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "llama3.2")
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
GROQ_MODEL      = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


# ─── Ollama Provider ─────────────────────────────────────────────────────────

class OllamaProvider:
    """Chat provider using a local Ollama instance."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 300},
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate", json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "No response generated.")
        except requests.Timeout:
            return "The local AI model timed out. Please try again."
        except requests.ConnectionError:
            return "Cannot connect to Ollama. Make sure it is running (ollama serve)."
        except Exception as e:
            return f"Ollama error: {str(e)}"


# ─── Groq Provider ───────────────────────────────────────────────────────────

class GroqProvider:
    """Chat provider using Groq cloud API (LLaMA 3.3 70B — fast & free)."""

    def __init__(self, api_key: str = GROQ_API_KEY):
        self.api_key = api_key
        self.client = None
        if api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=api_key)
            except ImportError:
                logger.warning("groq package not installed. Run: pip install groq")

    def is_available(self) -> bool:
        return bool(self.api_key and self.client is not None)

    def generate(self, prompt: str) -> str:
        if not self.client:
            return "Groq client not available. Install groq package and set GROQ_API_KEY."
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq error: {e}")
            return f"Groq error: {str(e)}"


# ─── Gemini Provider ─────────────────────────────────────────────────────────

class GeminiProvider:
    """Chat provider using Google Gemini API (cloud fallback)."""

    def __init__(self, api_key: str = GEMINI_API_KEY):
        if api_key:
            from google import genai
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None

    def is_available(self) -> bool:
        return bool(GEMINI_API_KEY)

    def generate(self, prompt: str) -> str:
        if not GEMINI_API_KEY or self.client is None:
            return "Gemini API key is not configured."
        try:
            from google.genai import types
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=400,
                ),
            )
            return response.text
        except Exception as e:
            return f"Gemini error: {str(e)}"


# ─── Provider Selection ──────────────────────────────────────────────────────

_provider = None


def _get_provider():
    """Select and cache the AI provider: Ollama → Groq → Gemini."""
    global _provider
    if _provider is not None:
        return _provider

    if CHAT_PROVIDER == "ollama":
        logger.info("🦙 CHAT_PROVIDER=ollama → Using Ollama")
        _provider = OllamaProvider()

    elif CHAT_PROVIDER == "groq":
        logger.info("⚡ CHAT_PROVIDER=groq → Using Groq")
        _provider = GroqProvider()

    elif CHAT_PROVIDER == "gemini":
        logger.info("✨ CHAT_PROVIDER=gemini → Using Gemini")
        _provider = GeminiProvider()

    else:  # "auto" — try Ollama → Groq → Gemini
        ollama = OllamaProvider()
        if ollama.is_available():
            logger.info("🦙 Auto: Ollama running locally → Using Ollama")
            _provider = ollama
        else:
            groq_p = GroqProvider()
            if groq_p.is_available():
                logger.info("⚡ Auto: Ollama not found → Using Groq (LLaMA 3.3)")
                _provider = groq_p
            else:
                logger.info("✨ Auto: Groq not available → Falling back to Gemini")
                _provider = GeminiProvider()

    return _provider


# ─── Public API ───────────────────────────────────────────────────────────────

def get_animal_info(animal_name: str) -> str:
    """Generate an informative description about the detected animal."""
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
    return _get_provider().generate(prompt)


def answer_question(animal_name: str, question: str) -> str:
    """Answer a follow-up question about the detected animal."""
    if not question or not question.strip():
        return "Please ask a question about the animal."

    prompt = (
        f"You are WildVision AI, an expert wildlife assistant. "
        f"The user is asking about a '{animal_name}' they photographed. "
        f"Their question is: '{question}'\n\n"
        f"Provide a helpful, accurate, and concise answer (under 150 words). "
        f"If the question is unrelated to the animal, politely redirect."
    )
    return _get_provider().generate(prompt)


def generate_care_packages(animal_name: str) -> dict:
    """Generate 3 tiered care packages (Basic, Standard, Premium) for the animal in INR."""
    prompt = (
        f"You are a veterinary and pet store assistant. The user has a '{animal_name}'.\n"
        f"Generate 3 distinct care packages (Basic/Low-Budget, Standard/Mid-Range, Premium/High-End) "
        f"with clear price variations in Indian Rupees (₹). Include food, vet needs, "
        f"and medical necessities (antibiotics, vaccines).\n\n"
        f"You MUST reply ONLY with a valid JSON object matching this exact structure, nothing else:\n"
        f"{{\n"
        f"  \"packages\": [\n"
        f"    {{\n"
        f"      \"tier\": \"Basic\",\n"
        f"      \"description\": \"Essential care items on a budget.\",\n"
        f"      \"total_price\": \"₹4000\",\n"
        f"      \"items\": [\n"
        f"        {{\"name\": \"Standard Kibble\", \"price\": \"₹1500\"}},\n"
        f"        {{\"name\": \"Basic Vet Checkup\", \"price\": \"₹2500\"}}\n"
        f"      ]\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n\n"
        f"Output ONLY valid JSON. No conversational text before or after."
    )

    response_text = _get_provider().generate(prompt)

    # Robust JSON extraction to prevent parsing errors when LLM includes markdown or conversational text
    import re
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if match:
        json_str = match.group(0)
    else:
        json_str = response_text

    try:
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Failed to parse care packages JSON: {e}\nResponse was: {response_text}")
        return {
            "packages": [
                {
                    "tier": "Error",
                    "description": "Failed to generate care packages. Please try again.",
                    "total_price": "₹0",
                    "items": [{"name": "Error fetching data", "price": "₹0"}],
                }
            ]
        }
