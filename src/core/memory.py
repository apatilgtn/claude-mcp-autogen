"""
Memory management for the MCP system.
This module provides functionality to store and retrieve agent memory.
"""

import time
import json
from typing import Dict, List, Any, Optional, Union
from collections import deque

from loguru import logger
from pydantic import BaseModel, Field

from src.core.config import settings


class MemoryEntry(BaseModel):
    """Base memory entry model."""
    id: str
    created_at: float = Field(default_factory=lambda: time.time())
    updated_at: Optional[float] = None
    content: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationMessage(BaseModel):
    """Model for a conversation message."""
    id: str
    type: str  # "user" or "assistant"
    content: str
    timestamp: float = Field(default_factory=lambda: time.time())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationMemory:
    """Memory for storing conversation history."""
    
    def __init__(self, capacity: int = 20):
        """
        Initialize conversation memory.
        
        Args:
            capacity: Maximum number of messages to store
        """
        self.messages: deque = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a user message to memory.
        
        Args:
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Message ID
        """
        import uuid
        
        message_id = str(uuid.uuid4())
        message = ConversationMessage(
            id=message_id,
            type="user",
            content=content,
            metadata=metadata or {}
        )
        
        self.messages.append(message)
        return message_id
    
    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add an assistant message to memory.
        
        Args:
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Message ID
        """
        import uuid
        
        message_id = str(uuid.uuid4())
        message = ConversationMessage(
            id=message_id,
            type="assistant",
            content=content,
            metadata=metadata or {}
        )
        
        self.messages.append(message)
        return message_id
    
    def add_message(self, message_type: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a message to memory.
        
        Args:
            message_type: Message type ("user" or "assistant")
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Message ID
        """
        if message_type == "user":
            return self.add_user_message(content, metadata)
        elif message_type == "assistant":
            return self.add_assistant_message(content, metadata)
        else:
            raise ValueError(f"Invalid message type: {message_type}")
    
    def get_message(self, message_id: str) -> Optional[ConversationMessage]:
        """
        Get a message by ID.
        
        Args:
            message_id: Message ID
            
        Returns:
            The message if found, None otherwise
        """
        for message in self.messages:
            if message.id == message_id:
                return message
        return None
    
    def get_messages(self, count: Optional[int] = None) -> List[ConversationMessage]:
        """
        Get recent messages.
        
        Args:
            count: Number of messages to retrieve
            
        Returns:
            List of messages
        """
        messages = list(self.messages)
        if count is not None:
            messages = messages[-count:]
        return messages
    
    def get_recent_messages(self, count: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get recent messages in a format suitable for LLM context.
        
        Args:
            count: Number of messages to retrieve
            
        Returns:
            List of messages as dictionaries
        """
        messages = self.get_messages(count)
        return [{"type": m.type, "content": m.content} for m in messages]
    
    def clear(self):
        """Clear all messages from memory."""
        self.messages.clear()


class SemanticMemory:
    """Memory for storing and retrieving semantic information."""
    
    def __init__(self, vector_dimension: int = 1536, similarity_threshold: float = 0.75):
        """
        Initialize semantic memory.
        
        Args:
            vector_dimension: Dimension of the embedding vectors
            similarity_threshold: Threshold for similarity matching
        """
        self.entries: List[MemoryEntry] = []
        self.vectors: Dict[str, List[float]] = {}
        self.vector_dimension = vector_dimension
        self.similarity_threshold = similarity_threshold
    
    async def add_entry(self, content: Any, metadata: Optional[Dict[str, Any]] = None, embedding: Optional[List[float]] = None) -> str:
        """
        Add an entry to semantic memory.
        
        Args:
            content: Entry content
            metadata: Optional metadata
            embedding: Optional pre-computed embedding
            
        Returns:
            Entry ID
        """
        import uuid
        from src.core.llm_provider import embed_text
        
        entry_id = str(uuid.uuid4())
        
        # Create memory entry
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            metadata=metadata or {}
        )
        
        # Get vector embedding if not provided
        if not embedding:
            if isinstance(content, str):
                embedding = await embed_text(content)
            else:
                content_str = json.dumps(content) if not isinstance(content, str) else content
                embedding = await embed_text(content_str)
        
        # Store entry and vector
        self.entries.append(entry)
        self.vectors[entry_id] = embedding
        
        return entry_id
    
    async def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query semantic memory for relevant entries.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of relevant entries with similarity scores
        """
        from src.core.llm_provider import embed_text
        import numpy as np
        
        # Get query embedding
        query_embedding = await embed_text(query)
        
        # Find similar entries
        results = []
        for entry in self.entries:
            vector = self.vectors.get(entry.id)
            if vector:
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, vector)
                
                if similarity >= self.similarity_threshold:
                    results.append({
                        "entry": entry,
                        "similarity": similarity
                    })
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Get an entry by ID.
        
        Args:
            entry_id: Entry ID
            
        Returns:
            The entry if found, None otherwise
        """
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None
    
    def update_entry(self, entry_id: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing entry.
        
        Args:
            entry_id: Entry ID
            content: New content
            metadata: New metadata
            
        Returns:
            True if updated, False otherwise
        """
        entry = self.get_entry(entry_id)
        if not entry:
            return False
        
        entry.content = content
        if metadata:
            entry.metadata.update(metadata)
        entry.updated_at = time.time()
        
        return True
    
    def delete_entry(self, entry_id: str) -> bool:
        """
        Delete an entry.
        
        Args:
            entry_id: Entry ID
            
        Returns:
            True if deleted, False otherwise
        """
        entry = self.get_entry(entry_id)
        if not entry:
            return False
        
        self.entries = [e for e in self.entries if e.id != entry_id]
        if entry_id in self.vectors:
            del self.vectors[entry_id]
        
        return True
    
    def clear(self):
        """Clear all entries from memory."""
        self.entries.clear()
        self.vectors.clear()
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity
        """
        import numpy as np
        
        # Convert to numpy arrays
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Calculate cosine similarity
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class EpisodicMemory:
    """Memory for storing sequential events or episodes."""
    
    def __init__(self, max_episodes: int = 100):
        """
        Initialize episodic memory.
        
        Args:
            max_episodes: Maximum number of episodes to store
        """
        self.episodes: Dict[str, List[MemoryEntry]] = {}
        self.max_episodes = max_episodes
    
    def create_episode(self, episode_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a new episode.
        
        Args:
            episode_id: Episode ID
            metadata: Optional metadata
            
        Returns:
            True if created, False if already exists
        """
        if episode_id in self.episodes:
            return False
        
        self.episodes[episode_id] = []
        
        # Prune episodes if limit reached
        if len(self.episodes) > self.max_episodes:
            oldest_episode = min(self.episodes.keys(), key=lambda k: 
                               min([e.created_at for e in self.episodes[k]]) if self.episodes[k] else float('inf'))
            del self.episodes[oldest_episode]
        
        return True
    
    def add_memory(self, episode_id: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Add a memory to an episode.
        
        Args:
            episode_id: Episode ID
            content: Memory content
            metadata: Optional metadata
            
        Returns:
            Memory ID if added, None if episode doesn't exist
        """
        import uuid
        
        if episode_id not in self.episodes:
            return None
        
        memory_id = str(uuid.uuid4())
        memory = MemoryEntry(
            id=memory_id,
            content=content,
            metadata=metadata or {}
        )
        
        self.episodes[episode_id].append(memory)
        return memory_id
    
    def get_episode(self, episode_id: str) -> List[MemoryEntry]:
        """
        Get all memories in an episode.
        
        Args:
            episode_id: Episode ID
            
        Returns:
            List of memories in the episode
        """
        return self.episodes.get(episode_id, [])
    
    def get_memory(self, episode_id: str, memory_id: str) -> Optional[MemoryEntry]:
        """
        Get a specific memory from an episode.
        
        Args:
            episode_id: Episode ID
            memory_id: Memory ID
            
        Returns:
            The memory if found, None otherwise
        """
        if episode_id not in self.episodes:
            return None
        
        for memory in self.episodes[episode_id]:
            if memory.id == memory_id:
                return memory
        
        return None
    
    def delete_episode(self, episode_id: str) -> bool:
        """
        Delete an episode.
        
        Args:
            episode_id: Episode ID
            
        Returns:
            True if deleted, False if not found
        """
        if episode_id not in self.episodes:
            return False
        
        del self.episodes[episode_id]
        return True
    
    def delete_memory(self, episode_id: str, memory_id: str) -> bool:
        """
        Delete a memory from an episode.
        
        Args:
            episode_id: Episode ID
            memory_id: Memory ID
            
        Returns:
            True if deleted, False if not found
        """
        if episode_id not in self.episodes:
            return False
        
        original_length = len(self.episodes[episode_id])
        self.episodes[episode_id] = [m for m in self.episodes[episode_id] if m.id != memory_id]
        
        return len(self.episodes[episode_id]) < original_length
    
    def clear(self):
        """Clear all episodes from memory."""
        self.episodes.clear()
