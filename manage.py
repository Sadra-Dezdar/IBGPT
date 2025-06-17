#!/usr/bin/env python
"""IB Student Assistant - Management Script"""

import sys
import subprocess
import os
from pathlib import Path

def print_header():
    print("🎓 IB Student Assistant - Management Console")
    print("=" * 50)

def show_help():
    print("""
Available commands:

🚀 RUNNING
  start          Start the web application 
  demo           Run interactive demo
  demo-pdf       Demo PDF upload and IA assessment
  
📊 TESTING  
  test           Run comprehensive system test
  test-pdf       Test PDF upload functionality
  status         Check system status
  test-ollama    Test Ollama connection
  test-chats     Test chat management system
  
📚 DATA MANAGEMENT
  setup          Set up ChromaDB collections
  ingest         Show document ingestion help
  quick-ingest   Ingest sample documents
  
💬 CHAT MANAGEMENT
  setup-chats    Initialize persistent chat system
  chat-stats     Show chat statistics
  backup-chats   Backup all chat data
  
🔧 MAINTENANCE
  requirements   Install Python dependencies
  models         Show required Ollama models
  clean          Clean up cache files
  
❓ HELP
  help           Show this help message
    """)

def run_command(cmd, description, background=False):
    print(f"🔄 {description}...")
    try:
        if background:
            subprocess.Popen(cmd, shell=True)
            print(f"✅ Started in background")
        else:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {description} completed")
                if result.stdout:
                    print(result.stdout)
            else:
                print(f"❌ {description} failed")
                if result.stderr:
                    print(result.stderr)
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print_header()
    
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "start":
        run_command("python run_app.py", "Starting IB Student Assistant", background=True)
        print("📱 Access at: http://localhost:8501")
        
    elif command == "demo":
        run_command("python demo.py", "Running interactive demo")
        
    elif command == "demo-pdf":
        run_command("python demo_pdf_assessment.py", "Running PDF upload and IA assessment demo")
        
    elif command == "test":
        run_command("python final_test.py", "Running comprehensive test")
        
    elif command == "test-pdf":
        run_command("python test_pdf_upload.py", "Testing PDF upload functionality")
        
    elif command == "status":
        run_command("python status_check.py", "Checking system status")
        
    elif command == "test-ollama":
        run_command("python test_ollama_simple.py", "Testing Ollama connection")
        
    elif command == "test-chats":
        run_command("python test_chat_manager.py", "Testing chat management system")
        
    elif command == "setup":
        run_command("python scripts/setup_collections.py", "Setting up ChromaDB collections")
        
    elif command == "setup-chats":
        run_command("python setup_persistent_chats.py", "Setting up persistent chat system")
        
    elif command == "chat-stats":
        print("📊 Chat Statistics")
        print("=" * 20)
        try:
            from utils.chat_manager import ChatManager
            chat_manager = ChatManager("./chats")
            stats = chat_manager.get_chat_stats()
            
            print(f"💬 Total Chats: {stats['total_chats']}")
            print(f"📝 Total Messages: {stats['total_messages']}")
            print(f"📊 Average Messages per Chat: {stats['average_messages_per_chat']:.1f}")
            
            if stats['oldest_chat']:
                print(f"📅 Oldest Chat: {stats['oldest_chat']['title']}")
            if stats['newest_chat']:
                print(f"🆕 Newest Chat: {stats['newest_chat']['title']}")
                
        except Exception as e:
            print(f"❌ Error getting chat stats: {e}")
    
    elif command == "backup-chats":
        print("💾 Backing up chat data...")
        try:
            from utils.chat_manager import ChatManager
            chat_manager = ChatManager("./chats")
            if chat_manager.backup_all_chats():
                print("✅ Chat backup completed successfully")
            else:
                print("❌ Chat backup failed")
        except Exception as e:
            print(f"❌ Error backing up chats: {e}")
        
    elif command == "ingest":
        print("""
📚 Document Ingestion Help

Single file:
  python scripts/ingest_documents.py [FILE] --doc-type [TYPE] --subject [SUBJECT]

Example:
  python scripts/ingest_documents.py ./data/math_guide.pdf --doc-type syllabus --subject "Mathematics AA" --level HL

Batch processing:
  python quick_ingest.py
        """)
        
    elif command == "quick-ingest":
        run_command("python quick_ingest.py", "Ingesting sample documents")
        
    elif command == "requirements":
        run_command("pip install -r requirements.txt", "Installing Python dependencies")
        
    elif command == "models":
        print("""
🤖 Required Ollama Models

Install with:
  ollama pull qwen3:latest
  ollama pull deepseek-r1:14b-qwen-distill-q4_K_M

The system uses:
- Qwen3 for fast query classification
- DeepSeek-R1 for deep reasoning and IB expertise
        """)
        
    elif command == "clean":
        run_command("find . -name '__pycache__' -type d -exec rm -rf {} +", "Cleaning cache files")
        
    elif command == "help":
        show_help()
        
    else:
        print(f"❌ Unknown command: {command}")
        show_help()

if __name__ == "__main__":
    main()
