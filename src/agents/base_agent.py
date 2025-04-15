"""
Base agent implementation for the MCP system.
This provides the foundation for all agent types in the system.
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from loguru import logger
from pydantic import BaseModel, Field

from src.core.mcp import MCPMessage, mcp_bus


class AgentCapability(BaseModel):
    """Capability that an agent can have."""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    examples: List[Dict[str, Any]] = Field(default_factory=list)


class AgentMemory(BaseModel):
    """Memory model for agents."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    key: str
    value: Any
    created_at: float = Field(default_factory=lambda: __import__('time').time())
    updated_at: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseAgent:
    """Base agent class for all agent implementations."""
    
    def __init__(self, 
                 agent_id: str,
                 name: str,
                 description: Optional[str] = None,
                 capabilities: Optional[List[AgentCapability]] = None,
                 system_message: Optional[str] = None):
        """
        Initialize a base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            description: Description of the agent's purpose and functionality
            capabilities: List of capabilities the agent possesses
            system_message: System message to guide the agent's behavior
        """
        self.agent_id = agent_id
        self.name = name
        self.description = description or f"Agent {name}"
        self.capabilities = capabilities or []
        self.system_message = system_message or f"You are {name}, an AI assistant."
        self.memory: Dict[str, AgentMemory] = {}
        self.is_active = False
        self.message_handlers: List[Callable[[MCPMessage], None]] = []
        self._task = None
    
    async def start(self):
        """Start the agent."""
        if self.is_active:
            return
        
        self.is_active = True
        await self._register_with_mcp()
        logger.info(f"Agent {self.name} ({self.agent_id}) started")
    
    async def stop(self):
        """Stop the agent."""
        if not self.is_active:
            return
        
        self.is_active = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"Agent {self.name} ({self.agent_id}) stopped")
    
    async def _register_with_mcp(self):
        """Register the agent with the MCP bus."""
        # This would typically involve setting up channels and handlers
        pass
    
    async def process_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """
        Process an incoming message and generate a response.
        This should be implemented by subclasses.
        
        Args:
            message: The incoming message to process
            
        Returns:
            An optional response message
        """
        raise NotImplementedError("Subclasses must implement process_message")
    
    async def send_message(self, receiver_id: str, content: Dict[str, Any], 
                          **kwargs) -> MCPMessage:
        """
        Send a message to another agent.
        
        Args:
            receiver_id: ID of the receiving agent
            content: Message content
            **kwargs: Additional message parameters
            
        Returns:
            The sent message
        """
        message = MCPMessage(
            sender=self.agent_id,
            receiver=receiver_id,
            content=content,
            **kwargs
        )
        
        await mcp_bus.send_message(message)
        return message
    
    def remember(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> AgentMemory:
        """
        Store a memory item.
        
        Args:
            key: Memory key
            value: Memory value
            metadata: Optional metadata
            
        Returns:
            The created memory item
        """
        memory = AgentMemory(
            agent_id=self.agent_id,
            key=key,
            value=value,
            metadata=metadata or {}
        )
        
        self.memory[key] = memory
        return memory
    
    def recall(self, key: str) -> Optional[Any]:
        """
        Retrieve a memory value by key.
        
        Args:
            key: Memory key
            
        Returns:
            The memory value if found, None otherwise
        """
        memory = self.memory.get(key)
        return memory.value if memory else None
    
    def forget(self, key: str) -> bool:
        """
        Remove a memory item.
        
        Args:
            key: Memory key
            
        Returns:
            True if the memory was removed, False otherwise
        """
        if key in self.memory:
            del self.memory[key]
            return True
        return False
    
    def add_capability(self, capability: AgentCapability):
        """
        Add a capability to the agent.
        
        Args:
            capability: The capability to add
        """
        self.capabilities.append(capability)
    
    def has_capability(self, capability_name: str) -> bool:
        """
        Check if the agent has a specific capability.
        
        Args:
            capability_name: Name of the capability
            
        Returns:
            True if the agent has the capability, False otherwise
        """
        return any(c.name == capability_name for c in self.capabilities)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the agent to a dictionary.
        
        Returns:
            Dictionary representation of the agent
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": [c.dict() for c in self.capabilities],
            "is_active": self.is_active
        }
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name} ({self.agent_id})>"