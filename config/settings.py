"""
Configuration settings for the IB Student Assistant.

This module centralizes all configuration parameters for the application,
including model settings, database paths, and system parameters.

Environment Variables:
- OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)

Key Components:
- Model configurations for different agents
- Database and embedding settings  
- Retrieval and UI parameters

Author: IB Student Assistant Team
Version: 2.0.0
"""

import os


# Ollama connection settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


# Model configurations - Optimized for IB assistant tasks
MODELS = {
    "fast_agent": {
        "name": "qwen3:latest",
        "description": "Fast query routing and classification"
    },
    "rag_agent": {
        "name": "deepseek-r1:14b-qwen-distill-q4_K_M", 
        "description": "Deep reasoning and knowledge retrieval"
    },
    "consensus_agent": {
        "name": "deepseek-r1:14b-qwen-distill-q4_K_M",
        "description": "Response review and refinement"
    }
}


# Database settings
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# Retrieval settings
DEFAULT_N_RESULTS = 6  # Number of documents to retrieve for context
MAX_CHUNK_SIZE = 1000  # Maximum size of document chunks


# UI settings  
DEFAULT_SHOW_THINKING = True  # Show AI reasoning process by default
