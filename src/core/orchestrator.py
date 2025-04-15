"""
Orchestrator for agent interactions using AutoGen.
This module provides the core orchestration logic for the MCP system.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import autogen
from autogen import Agent, AssistantAgent, UserProxyAgent, config_list_from_json
from loguru import logger
from pydantic import BaseModel, Field

from src.core.mcp import MCPMessage, mcp_bus
from src.core.config import settings

class AgentConfig(BaseModel):
    """Configuration for an agent in the orchestrator."""
    agent_id: str
    name: str
    system_message: str
    llm_config: Dict[str, Any]
    human_input_mode: str = "NEVER"
    max_consecutive_auto_reply: Optional[int] = 10
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    is_termination_agent: bool = False
    is_initiator: bool = False

class Orchestrator:
    """
    Orchestrator for agent interactions using AutoGen.
    
    This class manages multi-agent conversations and coordinates
    the execution of tasks across different agents.
    """
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self._running = False
        self._task = None
    
    async def start(self):
        """Start the orchestrator."""
        if self._running:
            return
        
        self._running = True
        await mcp_bus.start()
        logger.info("Orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator."""
        if not self._running:
            return
        
        self._running = False
        await mcp_bus.stop()
        logger.info("Orchestrator stopped")
    
    def register_agent_config(self, config: AgentConfig):
        """Register an agent configuration with the orchestrator."""
        self.agent_configs[config.agent_id] = config
        logger.info(f"Registered agent config: {config.name} ({config.agent_id})")
    
    def _create_agent(self, config: AgentConfig) -> Agent:
        """Create an AutoGen agent from configuration."""
        if config.agent_id in self.agents:
            return self.agents[config.agent_id]
        
        # Configure the LLM
        llm_config = config.llm_config
        
        # Add tools if provided
        tools = None
        if config.tools:
            tools = config.tools
        
        # Create the agent based on type
        if config.is_termination_agent:
            agent = UserProxyAgent(
                name=config.name,
                system_message=config.system_message,
                human_input_mode=config.human_input_mode,
                max_consecutive_auto_reply=config.max_consecutive_auto_reply,
                code_execution_config={"use_docker": False} if tools else None,
            )
        else:
            agent = AssistantAgent(
                name=config.name,
                system_message=config.system_message,
                llm_config=llm_config,
                tools=tools,
            )
        
        self.agents[config.agent_id] = agent
        logger.info(f"Created agent: {config.name} ({config.agent_id})")
        return agent
    
    def _ensure_agents_created(self, agent_ids: List[str]) -> List[Agent]:
        """Ensure all specified agents are created."""
        agents = []
        for agent_id in agent_ids:
            if agent_id not in self.agent_configs:
                raise ValueError(f"No configuration found for agent {agent_id}")
            
            if agent_id not in self.agents:
                self._create_agent(self.agent_configs[agent_id])
            
            agents.append(self.agents[agent_id])
        
        return agents
    
    async def create_conversation(self, agent_ids: List[str], 
                                 task_description: str, 
                                 max_rounds: int = 10) -> str:
        """
        Create a new conversation between multiple agents.
        
        Args:
            agent_ids: List of agent IDs to include in the conversation
            task_description: Description of the task for the conversation
            max_rounds: Maximum number of conversation rounds
            
        Returns:
            The conversation ID
        """
        conv_id = f"conv-{len(self.conversations) + 1}"
        
        # Ensure all agents exist
        agents = self._ensure_agents_created(agent_ids)
        
        # Find initiator agent
        initiator = None
        for agent_id in agent_ids:
            if self.agent_configs[agent_id].is_initiator:
                initiator = self.agents[agent_id]
                break
        
        if not initiator:
            # Use first agent as initiator if none specified
            initiator = agents[0]
        
        # Create an MCP channel for the conversation
        channel = mcp_bus.create_channel(
            name=f"Conversation {conv_id}",
            description=task_description,
            participants=agent_ids
        )
        
        # Store conversation metadata
        self.conversations[conv_id] = {
            "id": conv_id,
            "agents": agent_ids,
            "task": task_description,
            "channel_id": channel.id,
            "max_rounds": max_rounds,
            "current_round": 0,
            "status": "created",
            "messages": []
        }
        
        # Register handlers for agent messages
        for agent in agents:
            agent_id = next(aid for aid, a in self.agents.items() if a == agent)
            self._register_agent_handler(agent_id, channel.id)
        
        logger.info(f"Created conversation {conv_id} with {len(agents)} agents")
        return conv_id
    
    def _register_agent_handler(self, agent_id: str, channel_id: str):
        """Register a handler for agent messages in a channel."""
        async def handler(message: MCPMessage):
            if message.receiver != agent_id and message.sender != agent_id:
                return
            
            # Process the message using the agent
            agent = self.agents[agent_id]
            
            # Handle the message based on agent type
            if isinstance(agent, AssistantAgent):
                # For assistant agents, process and respond
                if message.receiver == agent_id:
                    response_content = await self._process_with_llm(agent, message.content)
                    response = MCPMessage(
                        sender=agent_id,
                        receiver=message.sender,
                        content=response_content,
                        reply_to=message.id,
                        trace_id=message.trace_id,
                        metadata={"channel_id": channel_id}
                    )
                    await mcp_bus.send_message(response)
            
            elif isinstance(agent, UserProxyAgent):
                # For user proxy agents, process and potentially terminate
                if message.receiver == agent_id:
                    # Check for termination conditions or provide response
                    is_termination = await self._check_termination(agent, message.content)
                    if is_termination:
                        # Signal conversation termination
                        conv_id = next(
                            cid for cid, conv in self.conversations.items() 
                            if conv["channel_id"] == channel_id
                        )
                        self.conversations[conv_id]["status"] = "completed"
                        logger.info(f"Conversation {conv_id} terminated")
                    else:
                        # Generate response
                        response_content = await self._process_with_proxy(agent, message.content)
                        response = MCPMessage(
                            sender=agent_id,
                            receiver=message.sender,
                            content=response_content,
                            reply_to=message.id,
                            trace_id=message.trace_id,
                            metadata={"channel_id": channel_id}
                        )
                        await mcp_bus.send_message(response)
        
        mcp_bus.register_handler(channel_id, handler)
    
    async def _process_with_llm(self, agent: AssistantAgent, content: Dict[str, Any]) -> Dict[str, Any]:
        """Process a message with an LLM-based agent."""
        # For actual implementation, this would integrate with AutoGen's methods
        # This is a simplified placeholder
        message = content.get("message", "")
        
        # Simulate LLM processing
        response = await asyncio.to_thread(
            lambda: agent.generate_reply(sender=agent, messages=[{"content": message}])
        )
        
        return {"message": response}
    
    async def _process_with_proxy(self, agent: UserProxyAgent, content: Dict[str, Any]) -> Dict[str, Any]:
        """Process a message with a proxy agent."""
        # For actual implementation, this would integrate with AutoGen's methods
        # This is a simplified placeholder
        message = content.get("message", "")
        
        # Simulate proxy processing
        response = await asyncio.to_thread(
            lambda: agent.generate_reply(sender=agent, messages=[{"content": message}])
        )
        
        return {"message": response}
    
    async def _check_termination(self, agent: UserProxyAgent, content: Dict[str, Any]) -> bool:
        """Check if a conversation should be terminated."""
        # For actual implementation, this would integrate with AutoGen's methods
        # This is a simplified placeholder
        message = content.get("message", "")
        
        # Example basic termination check
        if "TERMINATE" in message.upper() or "COMPLETE" in message.upper():
            return True
        
        return False
    
    async def start_conversation(self, conversation_id: str, initial_message: str) -> None:
        """Start a conversation with an initial message."""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} does not exist")
        
        conv = self.conversations[conversation_id]
        if conv["status"] != "created":
            raise ValueError(f"Conversation {conversation_id} is already {conv['status']}")
        
        # Update status
        conv["status"] = "active"
        
        # Determine initial sender/receiver
        initiator_id = None
        receiver_id = None
        
        for agent_id in conv["agents"]:
            if self.agent_configs[agent_id].is_initiator:
                initiator_id = agent_id
                break
        
        if not initiator_id:
            # Use first agent as initiator if none specified
            initiator_id = conv["agents"][0]
        
        # Find a receiver that isn't the initiator
        for agent_id in conv["agents"]:
            if agent_id != initiator_id:
                receiver_id = agent_id
                break
        
        if not receiver_id:
            raise ValueError(f"Need at least two agents for conversation {conversation_id}")
        
        # Create and send initial message
        message = MCPMessage(
            sender=initiator_id,
            receiver=receiver_id,
            content={"message": initial_message},
            trace_id=f"trace-{conversation_id}",
            metadata={"channel_id": conv["channel_id"]}
        )
        
        await mcp_bus.send_message(message)
        
        # Record the message
        conv["messages"].append({
            "id": message.id,
            "sender": message.sender,
            "receiver": message.receiver,
            "content": message.content,
            "timestamp": message.timestamp
        })
        
        # Increment the round counter
        conv["current_round"] = 1
        
        logger.info(f"Started conversation {conversation_id}")
    
    def get_conversation_status(self, conversation_id: str) -> Dict[str, Any]:
        """Get the status of a conversation."""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} does not exist")
        
        return {
            "id": conversation_id,
            "status": self.conversations[conversation_id]["status"],
            "current_round": self.conversations[conversation_id]["current_round"],
            "max_rounds": self.conversations[conversation_id]["max_rounds"],
            "message_count": len(self.conversations[conversation_id]["messages"])
        }
    
    def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages from a conversation."""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} does not exist")
        
        return self.conversations[conversation_id]["messages"]

# Global orchestrator instance
orchestrator = Orchestrator()
