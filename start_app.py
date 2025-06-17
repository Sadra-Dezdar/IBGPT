#!/usr/bin/env python3
"""
Quick status checker and app launcher for IBGPT.
"""

import os
import sys
import subprocess
import requests
import time

def check_streamlit_running():
    """Check if Streamlit is already running."""
    try:
        # Check if port 8501 is responding
        response = requests.get("http://localhost:8501", timeout=2)
        return True
    except:
        return False

def check_streamlit_processes():
    """Check for running Streamlit processes."""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        streamlit_processes = [line for line in result.stdout.split('\n') if 'streamlit' in line and 'grep' not in line]
        return streamlit_processes
    except:
        return []

def kill_streamlit():
    """Kill all Streamlit processes."""
    try:
        subprocess.run(['pkill', '-f', 'streamlit'], capture_output=True)
        print("🔄 Killed existing Streamlit processes")
        time.sleep(2)
        return True
    except:
        return False

def start_app():
    """Start the IBGPT app."""
    print("🚀 Starting IBGPT...")
    try:
        # Use the existing run_app.py script
        subprocess.run([sys.executable, 'run_app.py'])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error starting app: {e}")

def main():
    """Main function."""
    print("🔍 IBGPT Status Checker")
    print("=" * 30)
    
    # Check if app is already running
    if check_streamlit_running():
        print("✅ IBGPT is already running at http://localhost:8501")
        print("📱 Open this URL in your browser to use the app")
        return
    
    # Check for zombie processes
    processes = check_streamlit_processes()
    if processes:
        print("⚠️  Found existing Streamlit processes:")
        for proc in processes:
            print(f"   {proc}")
        
        response = input("🔄 Kill existing processes and restart? (y/n): ")
        if response.lower() in ['y', 'yes']:
            kill_streamlit()
        else:
            print("❌ Cannot start new instance with existing processes running")
            return
    
    print("🎓 No existing app found. Starting IBGPT...")
    print("📱 App will be available at: http://localhost:8501")
    print("💡 Press Ctrl+C to stop the app")
    print()
    
    start_app()

if __name__ == "__main__":
    main()
