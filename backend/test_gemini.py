import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Directly set your API key here (or load from .env if preferred)
api_key = "AIzaSyDVNEXEdnM04gWGjK2gykccLSmrjjZUlbU"

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key  # Force usage of API key instead of service account
)

response = llm.invoke("Hello! Can you confirm you are working?")
print(response)
