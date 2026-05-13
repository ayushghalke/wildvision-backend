import os
from dotenv import load_dotenv
from chatbot_service import GroqProvider

load_dotenv()

provider = GroqProvider()
print("Available:", provider.is_available())
result = provider.generate("Hello, who are you?")
print("Result:", result)
