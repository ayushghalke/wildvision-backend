"""
WildVision — Chatbot Service
Uses Google Gemini AI to provide information about detected animals.
"""

import google.generativeai as genai
import os

# Configure Gemini API
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCQPii0sw55ZKhpNxdOG5WCvA-RTe5oMkA")
genai.configure(api_key=API_KEY)

# Use Gemini 2.0 Flash for fast responses
model = genai.GenerativeModel("gemini-2.0-flash")


def get_animal_info(animal_name: str) -> str:
    """
    Generate an informative description about the detected animal.
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

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not retrieve information. Error: {str(e)}"


def answer_question(animal_name: str, question: str) -> str:
    """
    Answer a follow-up question about the detected animal.
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

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, I couldn't process your question. Error: {str(e)}"
