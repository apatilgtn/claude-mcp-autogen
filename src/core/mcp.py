"""
Multi-Client Protocol (MCP) implementation for agent communication.
This module provides the core MCP functionality inspired by Claude's architecture.
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from loguru import logger
from pydantic import BaseModel, Field

class MCPMessage(BaseModel):
    """Message format for MCP communications."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    receiver: str
    content: Dict[str, Any]
    timestamp: float = Field(default_factory=lambda: __import__('time').time())
    message_type: str = "standard"
    trace_id: Optional[str] = None
    reply_to: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MCPChannel(BaseModel):
    """Channel for MCP communications between agents."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    participants: List[str] = Field(default_factory=list)
    is_active: bool = True
    created_at: float = Field(default_factory=lambda: __import__('time').time())
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MCPBus:
    """Message bus for MCP communications."""
    
    def __init__(self):
        self.channels: Dict[str, MCPChannel] = {}
        self.message_handlers: Dict[str, List[Callable[[MCPMessage], None]]] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task = None
    
    async def start(self):
        """Start the message bus."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._process_messages())
        logger.info("MCP Bus started")
    
    async def stop(self):
        """Stop the message bus."""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("MCP Bus stopped")
    
    async def _process_messages(self):
        """Process messages from the queue."""
        while self._running:
            try:
                message = await self.message_queue.get()
                await self._dispatch_message(message)
                self.message_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def _dispatch_message(self, message: MCPMessage):
        """Dispatch a message to registered handlers."""
        channel_id = message.metadata.get("channel_id")
        if not channel_id:
            logger.warning(f"Message {message.id} has no channel_id")
            return
        
        if channel_id not in self.message_handlers:
            logger.warning(f"No handlers registered for channel {channel_id}")
            return
        
        handlers = self.message_handlers.get(channel_id, [])
        for handler in handlers:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Error in message handler: {e}")
    
    async def send_message(self, message: MCPMessage):
        """Send a message to the bus."""
        await self.message_queue.put(message)
        logger.debug(f"Message {message.id} queued from {message.sender} to {message.receiver}")
    
    def create_channel(self, name: str, description: Optional[str] = None, 
                       participants: Optional[List[str]] = None) -> MCPChannel:
        """Create a new communication channel."""
        channel = MCPChannel(
            name=name,
            description=description,
            participants=participants or []
        )
        self.channels[channel.id] = channel
        self.message_handlers[channel.id] = []
        logger.info(f"Channel {channel.id} created: {name}")
        return channel
    
    def register_handler(self, channel_id: str, handler: Callable[[MCPMessage], None]):
        """Register a message handler for a channel."""
        if channel_id not in self.channels:
            raise ValueError(f"Channel {channel_id} does not exist")
        
        if channel_id not in self.message_handlers:
            self.message_handlers[channel_id] = []
        
        self.message_handlers[channel_id].append(handler)
        logger.debug(f"Handler registered for channel {channel_id}")
    
    def get_channel(self, channel_id: str) -> Optional[MCPChannel]:
        """Get a channel by ID."""
        return self.channels.get(channel_id)
    
    def add_participant(self, channel_id: str, participant_id: str):
        """Add a participant to a channel."""
        if channel_id not in self.channels:
            raise ValueError(f"Channel {channel_id} does not exist")
        
        channel = self.channels[channel_id]
        if participant_id not in channel.participants:
            channel.participants.append(participant_id)
            logger.debug(f"Added {participant_id} to channel {channel_id}")
    
    def remove_participant(self, channel_id: str, participant_id: str):
        """Remove a participant from a channel."""
        if channel_id not in self.channels:
            raise ValueError(f"Channel {channel_id} does not exist")
        
        channel = self.channels[channel_id]
        if participant_id in channel.participants:
            channel.participants.remove(participant_id)
            logger.debug(f"Removed {participant_id} from channel {channel_id}")

# Global MCP bus instance
mcp_bus = MCPBus()
