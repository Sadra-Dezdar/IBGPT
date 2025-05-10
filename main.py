"""Main entry point for the IB Student Assistant."""

import asyncio
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def check_environment():
    """Check if all required environment variables are set."""
    import os
    
    required_vars = [
        "FAST_AGENT_MODEL",
        "RAG_AGENT_MODEL",
        "CONSENSUS_AGENT_MODEL"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.info("Using default model configurations")
    
    # Check if Ollama is running
    try:
        import requests
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        response = requests.get(f"{ollama_host}/api/version", timeout=5)
        if response.status_code != 200:
            logger.warning("Ollama server might not be running properly")
    except Exception as e:
        logger.error(f"Cannot connect to Ollama: {e}")
        logger.error("Please make sure Ollama is running (ollama server)")
        sys.exit(1)

def main():
    """Main function to run the application."""
    
    # Check environment setup
    check_environment()
    
    # Check if running Streamlit UI or CLI
    if len(sys.argv) > 1:
        if sys.argv[1] == "setup":
            # Run setup script
            import subprocess
            logger.info("Running setup script...")
            subprocess.run([sys.executable, "scripts/setup_collections.py"])
        elif sys.argv[1] == "ingest":
            # Show ingest help
            logger.info("To ingest documents, use:")
            logger.info("  python scripts/ingest_documents.py <file> --doc-type <type> --subject <subject>")
            logger.info("  python scripts/batch_ingest.py <directory>")
        elif sys.argv[1] == "cli":
            # Run CLI version (if implemented)
            logger.error("CLI version not implemented yet")
        else:
            logger.error(f"Unknown command: {sys.argv[1]}")
            logger.info("Available commands: setup, ingest")
    else:
        # Run Streamlit UI
        import subprocess
        logger.info("Starting IB Student Assistant UI...")
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", "interface/streamlit_app.py"])
        except KeyboardInterrupt:
            logger.info("Application stopped by user")
        except Exception as e:
            logger.error(f"Error running Streamlit: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()