"""
Agent management routes for the FastAPI application.
This module provides endpoints for managing MCP agents.
"""

from typing import Dict, List, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, Body, Query, Path
from pydantic import BaseModel, Field

from src.api.dependencies import get_current_user, verify_admin, User
from src.core.orchestrator import orchestrator
from src.agents.base_agent import AgentCapability
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.research_agent import ResearchAgent
from src.agents.coding_agent import CodingAgent
from src.agents.conversation_agent import ConversationAgent


router = APIRouter()


class AgentCapabilityResponse(BaseModel):
    """Model for agent capability response."""
    name: str = Field(..., description="Capability name")
    description: str = Field(..., description="Capability description")


class AgentResponse(BaseModel):
    """Model for agent response."""
    agent_id: str = Field(..., description="Agent ID")
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    capabilities: List[AgentCapabilityResponse] = Field(default_factory=list, description="Agent capabilities")
    is_active: bool = Field(..., description="Whether the agent is active")


class AgentCreate(BaseModel):
    """Model for creating an agent."""
    agent_type: str = Field(..., description="Agent type (reasoning, research, coding, conversation)")
    name: str = Field(..., description="Agent name")
    description: Optional[str] = Field(None, description="Agent description")
    model: Optional[str] = Field(None, description="LLM model to use")
    temperature: Optional[float] = Field(None, description="LLM temperature")
    system_message: Optional[str] = Field(None, description="Custom system message")


@router.get("", response_model=List[AgentResponse])
async def list_agents(
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    current_user: User = Depends(get_current_user)
):
    """
    List all available agents.
    
    Args:
        agent_type: Optional agent type filter
        current_user: Current authenticated user
        
    Returns:
        List of agent information
    """
    try:
        # Collect all agents from the orchestrator
        agents = []
        
        for agent_id, agent_obj in orchestrator.agents.items():
            # Skip if agent type filter is provided and doesn't match
            if agent_type and not _check_agent_type(agent_obj, agent_type):
                continue
            
            # Get agent information
            agent_info = agent_obj.to_dict()
            agents.append(agent_info)
        
        return agents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing agents: {str(e)}")


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str = Path(..., description="Agent ID"),
    current_user: User = Depends(get_current_user)
):
    """
    Get information about a specific agent.
    
    Args:
        agent_id: Agent ID
        current_user: Current authenticated user
        
    Returns:
        Agent information
    """
    try:
        # Check if agent exists
        if agent_id not in orchestrator.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Get agent information
        agent_obj = orchestrator.agents[agent_id]
        agent_info = agent_obj.to_dict()
        
        return agent_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting agent: {str(e)}")


@router.post("", response_model=AgentResponse)
async def create_agent(
    agent_data: AgentCreate = Body(...),
    current_user: User = Depends(verify_admin)
):
    """
    Create a new agent.
    
    Args:
        agent_data: Agent creation parameters
        current_user: Current authenticated admin user
        
    Returns:
        Created agent information
    """
    try:
        # Generate a unique agent ID
        agent_id = f"{agent_data.agent_type}-{len(orchestrator.agents) + 1}"
        
        # Create the agent based on its type
        if agent_data.agent_type == "reasoning":
            agent = ReasoningAgent(
                agent_id=agent_id,
                name=agent_data.name,
                description=agent_data.description,
                system_message=agent_data.system_message,
                model=agent_data.model or "claude-3-7-sonnet-20250219",
                temperature=agent_data.temperature or 0.2
            )
        elif agent_data.agent_type == "research":
            agent = ResearchAgent(
                agent_id=agent_id,
                name=agent_data.name,
                description=agent_data.description,
                system_message=agent_data.system_message,
                model=agent_data.model or "claude-3-7-sonnet-20250219",
                temperature=agent_data.temperature or 0.3
            )
        elif agent_data.agent_type == "coding":
            agent = CodingAgent(
                agent_id=agent_id,
                name=agent_data.name,
                description=agent_data.description,
                system_message=agent_data.system_message,
                model=agent_data.model or "claude-3-7-sonnet-20250219",
                temperature=agent_data.temperature or 0.2
            )
        elif agent_data.agent_type == "conversation":
            agent = ConversationAgent(
                agent_id=agent_id,
                name=agent_data.name,
                description=agent_data.description,
                system_message=agent_data.system_message,
                model=agent_data.model or "claude-3-7-sonnet-20250219",
                temperature=agent_data.temperature or 0.7
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported agent type: {agent_data.agent_type}")
        
        # Register the agent with the orchestrator
        orchestrator.agents[agent_id] = agent
        
        # Register the agent config
        agent_config = {
            "agent_id": agent_id,
            "name": agent_data.name,
            "system_message": agent.system_message,
            "llm_config": {
                "model": agent_data.model or "claude-3-7-sonnet-20250219",
                "temperature": agent_data.temperature
            },
            "is_initiator": agent_data.agent_type == "conversation",
            "is_termination_agent": False
        }
        orchestrator.register_agent_config(agent_config)
        
        # Start the agent
        await agent.start()
        
        return agent.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating agent: {str(e)}")


@router.delete("/{agent_id}", response_model=Dict[str, bool])
async def delete_agent(
    agent_id: str = Path(..., description="Agent ID"),
    current_user: User = Depends(verify_admin)
):
    """
    Delete an agent.
    
    Args:
        agent_id: Agent ID
        current_user: Current authenticated admin user
        
    Returns:
        Success status
    """
    try:
        # Check if agent exists
        if agent_id not in orchestrator.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Get the agent
        agent = orchestrator.agents[agent_id]
        
        # Stop the agent
        await agent.stop()
        
        # Remove the agent from the orchestrator
        del orchestrator.agents[agent_id]
        
        # Remove the agent config
        if agent_id in orchestrator.agent_configs:
            del orchestrator.agent_configs[agent_id]
        
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting agent: {str(e)}")


def _check_agent_type(agent, agent_type: str) -> bool:
    """
    Check if an agent is of a specific type.
    
    Args:
        agent: Agent object
        agent_type: Agent type to check
        
    Returns:
        True if the agent is of the specified type, False otherwise
    """
    if agent_type == "reasoning" and isinstance(agent, ReasoningAgent):
        return True
    elif agent_type == "research" and isinstance(agent, ResearchAgent):
        return True
    elif agent_type == "coding" and isinstance(agent, CodingAgent):
        return True
    elif agent_type == "conversation" and isinstance(agent, ConversationAgent):
        return True
    return False
