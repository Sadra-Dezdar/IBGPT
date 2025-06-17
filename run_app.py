#!/usr/bin/env python
"""
IB Student Assistant - Application Launcher

This is the primary startup script for the IB Student Assistant web application.
It handles PyTorch/Streamlit compatibility issues and ensures proper environment setup.

Features:
- Automatic environment configuration
- PyTorch-Streamlit conflict resolution
- Clean application startup
- Error handling and user feedback

Usage:
    python run_app.py

The application will be available at: http://localhost:8501

Author: IB Student Assistant Team
Version: 2.0.0
"""

import os
import sys
import warnings

# Suppress PyTorch warnings that interfere with Streamlit
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Set environment variables to prevent PyTorch-Streamlit conflicts
os.environ["TORCH_DISABLE_AUTOGRAD"] = "1"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"

def main():
    """Run the Streamlit app with proper environment setup."""
    
    # Check if collections are set up
    from pathlib import Path
    chroma_path = Path("./chroma_db")
    
    if not chroma_path.exists() or not any(chroma_path.iterdir()):
        print("ChromaDB collections not found. Setting up...")
        import subprocess
        subprocess.run([sys.executable, "scripts/setup_collections.py"])
    
    # Start Streamlit
    print("🎓 Starting IB Student Assistant with DeepSeek-R1...")
    print("📱 Will be available at: http://localhost:8501")
    import subprocess
    
    # Run streamlit with specific configuration to avoid PyTorch conflicts
    env = os.environ.copy()
    env["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    
    try:
        subprocess.run([
            sys.executable, 
            "-m", "streamlit", "run", 
            "interface/streamlit_app.py",
            "--server.port", "8501",
            "--server.fileWatcherType", "none",
            "--server.runOnSave", "false"
        ], env=env)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    main()
