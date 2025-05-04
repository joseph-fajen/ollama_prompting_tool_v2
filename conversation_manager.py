import json
import os
import time
from datetime import datetime
import uuid
from rich.console import Console
from typing import List, Dict, Optional, Any, Union

class Message:
    """Represents a single message in a conversation"""
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        self.timestamp = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create a Message from a dictionary"""
        message = cls(data["role"], data["content"])
        if "timestamp" in data:
            message.timestamp = data["timestamp"]
        return message
        
    def to_openai_format(self) -> Dict[str, str]:
        """Convert to OpenAI-compatible message format"""
        return {
            "role": self.role,
            "content": self.content
        }

class Conversation:
    """Manages a conversation with message history"""
    def __init__(self, id: Optional[str] = None, title: Optional[str] = None):
        self.id = id or str(uuid.uuid4())
        self.title = title or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.messages: List[Message] = []
        self.created_at = time.time()
        self.updated_at = time.time()
        self.metadata: Dict[str, Any] = {}
        
    def add_message(self, role: str, content: str) -> Message:
        """Add a new message to the conversation"""
        message = Message(role, content)
        self.messages.append(message)
        self.updated_at = time.time()
        return message
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary for serialization"""
        return {
            "id": self.id,
            "title": self.title,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create a Conversation from a dictionary"""
        conversation = cls(data.get("id"), data.get("title"))
        conversation.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        conversation.created_at = data.get("created_at", time.time())
        conversation.updated_at = data.get("updated_at", time.time())
        conversation.metadata = data.get("metadata", {})
        return conversation
        
    def to_openai_format(self) -> List[Dict[str, str]]:
        """Convert conversation to OpenAI-compatible messages format"""
        return [m.to_openai_format() for m in self.messages]
        
    def clear(self) -> None:
        """Clear all messages from the conversation"""
        self.messages = []
        self.updated_at = time.time()

class ConversationManager:
    """Manages multiple conversations and persistence"""
    def __init__(self, save_directory: str = "ollama_conversations"):
        self.conversations: Dict[str, Conversation] = {}
        self.current_conversation_id: Optional[str] = None
        self.save_directory = save_directory
        self.console = Console()
        
        # Create the conversations directory if it doesn't exist
        os.makedirs(self.save_directory, exist_ok=True)
        
    def create_conversation(self, title: Optional[str] = None) -> Conversation:
        """Create a new conversation and set it as current"""
        conversation = Conversation(title=title)
        self.conversations[conversation.id] = conversation
        self.current_conversation_id = conversation.id
        return conversation
        
    def get_current_conversation(self) -> Optional[Conversation]:
        """Get the current active conversation"""
        if self.current_conversation_id is None:
            return None
        return self.conversations.get(self.current_conversation_id)
        
    def set_current_conversation(self, conversation_id: str) -> bool:
        """Set the current active conversation by ID"""
        if conversation_id in self.conversations:
            self.current_conversation_id = conversation_id
            return True
        return False
        
    def add_message_to_current(self, role: str, content: str) -> Optional[Message]:
        """Add a message to the current conversation"""
        conversation = self.get_current_conversation()
        if conversation is None:
            conversation = self.create_conversation()
        return conversation.add_message(role, content)
        
    def save_conversation(self, conversation_id: Optional[str] = None) -> bool:
        """Save a specific conversation to disk"""
        if conversation_id is None and self.current_conversation_id is None:
            return False
            
        conv_id = conversation_id or self.current_conversation_id
        conversation = self.conversations.get(conv_id)
        if conversation is None:
            return False
            
        filename = f"{self.save_directory}/{conversation.id}.json"
        try:
            with open(filename, "w") as f:
                json.dump(conversation.to_dict(), f, indent=2)
            return True
        except Exception as e:
            self.console.print(f"[bold red]Error saving conversation:[/bold red] {str(e)}")
            return False
            
    def load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Load a conversation from disk by ID"""
        filename = f"{self.save_directory}/{conversation_id}.json"
        if not os.path.exists(filename):
            return None
            
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            conversation = Conversation.from_dict(data)
            self.conversations[conversation.id] = conversation
            return conversation
        except Exception as e:
            self.console.print(f"[bold red]Error loading conversation:[/bold red] {str(e)}")
            return None
            
    def list_saved_conversations(self) -> List[Dict[str, Any]]:
        """List all saved conversations"""
        conversations = []
        for filename in os.listdir(self.save_directory):
            if filename.endswith(".json"):
                try:
                    with open(f"{self.save_directory}/{filename}", "r") as f:
                        data = json.load(f)
                    conversations.append({
                        "id": data.get("id"),
                        "title": data.get("title"),
                        "message_count": len(data.get("messages", [])),
                        "updated_at": data.get("updated_at")
                    })
                except:
                    pass
        return sorted(conversations, key=lambda c: c.get("updated_at", 0), reverse=True)
        
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation from memory and disk"""
        filename = f"{self.save_directory}/{conversation_id}.json"
        
        # Remove from memory
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            
        # If this was the current conversation, unset it
        if self.current_conversation_id == conversation_id:
            self.current_conversation_id = None
            
        # Remove from disk if it exists
        if os.path.exists(filename):
            try:
                os.remove(filename)
                return True
            except Exception as e:
                self.console.print(f"[bold red]Error deleting conversation:[/bold red] {str(e)}")
                return False
        return True