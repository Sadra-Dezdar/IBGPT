"""Test Ollama connection."""

import requests

try:
    response = requests.get("http://localhost:11434/api/version")
    print(f"Ollama is running! Version: {response.json()}")
    
    # List models
    models_response = requests.get("http://localhost:11434/api/tags")
    models = models_response.json()
    print("\nAvailable models:")
    for model in models.get("models", []):
        print(f"- {model['name']}")
except Exception as e:
    print(f"Error: {e}")
    print("Make sure Ollama is running with: ollama serve")