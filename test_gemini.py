import os
from dotenv import load_dotenv
from chatbot_service import GeminiProvider

load_dotenv()

provider = GeminiProvider()
print("Available:", provider.is_available())
result = provider.generate("Hello, who are you?")
print("Result:", result)
