"""Chat session management utilities for persistent conversations."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import uuid
import shutil

class ChatManager:
    """Manages persistent chat sessions and conversation history."""
    
    def __init__(self, chats_dir: str = "./chats"):
        """Initialize chat manager with storage directory."""
        self.chats_dir = Path(chats_dir)
        self.chats_dir.mkdir(exist_ok=True)
        
        # Metadata file to track all chats
        self.metadata_file = self.chats_dir / "chat_metadata.json"
        self._ensure_metadata_file()
    
    def _ensure_metadata_file(self):
        """Ensure metadata file exists."""
        if not self.metadata_file.exists():
            self._save_metadata({})
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load chat metadata."""
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save chat metadata."""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def create_new_chat(self, title: str = None) -> str:
        """Create a new chat session and return its ID."""
        chat_id = str(uuid.uuid4())[:8]  # Short UUID for readability
        timestamp = datetime.now().isoformat()
        
        # Generate title if not provided
        if not title:
            title = f"Chat {timestamp[:10]}"  # Use date as default title
        
        # Create chat file
        chat_data = {
            "id": chat_id,
            "title": title,
            "created_at": timestamp,
            "updated_at": timestamp,
            "messages": []
        }
        
        self._save_chat(chat_id, chat_data)
        
        # Update metadata
        metadata = self._load_metadata()
        metadata[chat_id] = {
            "title": title,
            "created_at": timestamp,
            "updated_at": timestamp,
            "message_count": 0
        }
        self._save_metadata(metadata)
        
        return chat_id
    
    def _save_chat(self, chat_id: str, chat_data: Dict[str, Any]):
        """Save chat data to file."""
        chat_file = self.chats_dir / f"{chat_id}.json"
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)
    
    def load_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Load a specific chat by ID."""
        chat_file = self.chats_dir / f"{chat_id}.json"
        
        if not chat_file.exists():
            return None
        
        try:
            with open(chat_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    def save_message(self, chat_id: str, role: str, content: str, thinking: str = ""):
        """Add a message to a chat session."""
        chat_data = self.load_chat(chat_id)
        if not chat_data:
            return False
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if thinking:
            message["thinking"] = thinking
        
        chat_data["messages"].append(message)
        chat_data["updated_at"] = datetime.now().isoformat()
        
        self._save_chat(chat_id, chat_data)
        
        # Update metadata
        metadata = self._load_metadata()
        if chat_id in metadata:
            metadata[chat_id]["updated_at"] = chat_data["updated_at"]
            metadata[chat_id]["message_count"] = len(chat_data["messages"])
            self._save_metadata(metadata)
        
        return True
    
    def get_chat_list(self) -> List[Dict[str, Any]]:
        """Get list of all chats, sorted by most recent."""
        metadata = self._load_metadata()
        
        chats = []
        for chat_id, info in metadata.items():
            chats.append({
                "id": chat_id,
                "title": info["title"],
                "created_at": info["created_at"],
                "updated_at": info["updated_at"],
                "message_count": info.get("message_count", 0)
            })
        
        # Sort by updated_at (most recent first)
        chats.sort(key=lambda x: x["updated_at"], reverse=True)
        return chats
    
    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat session."""
        chat_file = self.chats_dir / f"{chat_id}.json"
        
        if chat_file.exists():
            chat_file.unlink()
        
        # Update metadata
        metadata = self._load_metadata()
        if chat_id in metadata:
            del metadata[chat_id]
            self._save_metadata(metadata)
            return True
        
        return False
    
    def update_chat_title(self, chat_id: str, new_title: str) -> bool:
        """Update the title of a chat session."""
        chat_data = self.load_chat(chat_id)
        if not chat_data:
            return False
        
        chat_data["title"] = new_title
        chat_data["updated_at"] = datetime.now().isoformat()
        self._save_chat(chat_id, chat_data)
        
        # Update metadata
        metadata = self._load_metadata()
        if chat_id in metadata:
            metadata[chat_id]["title"] = new_title
            metadata[chat_id]["updated_at"] = chat_data["updated_at"]
            self._save_metadata(metadata)
        
        return True
    
    def get_recent_chats(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent chats up to the specified limit."""
        all_chats = self.get_chat_list()
        return all_chats[:limit]
    
    def search_chats(self, query: str) -> List[Dict[str, Any]]:
        """Search chats by title or content."""
        matching_chats = []
        
        for chat_info in self.get_chat_list():
            chat_id = chat_info["id"]
            
            # Check title
            if query.lower() in chat_info["title"].lower():
                matching_chats.append(chat_info)
                continue
            
            # Check message content
            chat_data = self.load_chat(chat_id)
            if chat_data:
                for message in chat_data["messages"]:
                    if query.lower() in message["content"].lower():
                        matching_chats.append(chat_info)
                        break
        
        return matching_chats
    
    def get_chat_preview(self, chat_id: str, max_messages: int = 3) -> str:
        """Get a preview of the chat (first few messages)."""
        chat_data = self.load_chat(chat_id)
        if not chat_data or not chat_data["messages"]:
            return "Empty chat"
        
        preview_messages = chat_data["messages"][:max_messages]
        
        preview = ""
        for msg in preview_messages:
            role_emoji = "👤" if msg["role"] == "user" else "🤖"
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            preview += f"{role_emoji} {content}\n"
        
        if len(chat_data["messages"]) > max_messages:
            preview += f"... (+{len(chat_data['messages']) - max_messages} more messages)"
        
        return preview.strip()
    
    def export_chat(self, chat_id: str, format: str = "json") -> Optional[str]:
        """Export a chat to different formats."""
        chat_data = self.load_chat(chat_id)
        if not chat_data:
            return None
        
        if format == "json":
            return json.dumps(chat_data, indent=2, ensure_ascii=False)
        elif format == "txt":
            content = f"# {chat_data['title']}\n"
            content += f"Created: {chat_data['created_at']}\n"
            content += f"Updated: {chat_data['updated_at']}\n\n"
            
            for msg in chat_data["messages"]:
                role = "User" if msg["role"] == "user" else "Assistant"
                timestamp = msg.get("timestamp", "")
                content += f"## {role} ({timestamp})\n"
                content += f"{msg['content']}\n\n"
                
                if msg.get("thinking"):
                    content += f"### Thinking Process\n{msg['thinking']}\n\n"
            
            return content
        
        return None
    
    def backup_all_chats(self, backup_dir: str = "./chat_backups") -> bool:
        """Create a backup of all chats."""
        try:
            backup_path = Path(backup_dir)
            backup_path.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_path / f"chats_backup_{timestamp}.zip"
            
            # Create zip backup
            shutil.make_archive(str(backup_file).replace('.zip', ''), 'zip', self.chats_dir)
            
            return True
        except Exception:
            return False
    
    def get_chat_stats(self) -> Dict[str, Any]:
        """Get statistics about all chats."""
        all_chats = self.get_chat_list()
        
        if not all_chats:
            return {
                "total_chats": 0,
                "total_messages": 0,
                "oldest_chat": None,
                "newest_chat": None,
                "average_messages_per_chat": 0
            }
        
        total_messages = sum(chat.get("message_count", 0) for chat in all_chats)
        oldest_chat = min(all_chats, key=lambda x: x["created_at"])
        newest_chat = max(all_chats, key=lambda x: x["created_at"])
        
        return {
            "total_chats": len(all_chats),
            "total_messages": total_messages,
            "oldest_chat": oldest_chat,
            "newest_chat": newest_chat,
            "average_messages_per_chat": total_messages / len(all_chats) if all_chats else 0
        }
