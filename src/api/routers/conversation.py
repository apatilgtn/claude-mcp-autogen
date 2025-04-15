"""
Conversation management routes for the FastAPI application.
This module provides endpoints for managing conversations between agents.
"""

from typing import Dict, List, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, Body, Query
from pydantic import BaseModel, Field

from src.api.dependencies import get_current_user, User
from src.core.orchestrator import orchestrator


router = APIRouter()


class ConversationCreate(BaseModel):
    """Model for creating a conversation."""
    agent_ids: List[str] = Field(..., description="List of agent IDs to include in the conversation")
    task_description: str = Field(..., description="Description of the conversation task")
    max_rounds: int = Field(10, description="Maximum number of conversation rounds")


class ConversationStart(BaseModel):
    """Model for starting a conversation."""
    message: str = Field(..., description="Initial message to start the conversation")


class ConversationResponse(BaseModel):
    """Model for conversation response."""
    id: str = Field(..., description="Conversation ID")
    status: str = Field(..., description="Conversation status")
    current_round: int = Field(..., description="Current conversation round")
    max_rounds: int = Field(..., description="Maximum conversation rounds")
    message_count: int = Field(..., description="Number of messages in the conversation")


class MessageResponse(BaseModel):
    """Model for message response."""
    id: str = Field(..., description="Message ID")
    sender: str = Field(..., description="Sender ID")
    receiver: str = Field(..., description="Receiver ID")
    content: Dict[str, Any] = Field(..., description="Message content")
    timestamp: float = Field(..., description="Message timestamp")


@router.post("", response_model=ConversationResponse)
async def create_conversation(
    conversation: ConversationCreate = Body(...),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new conversation between agents.
    
    Args:
        conversation: Conversation creation parameters
        current_user: Current authenticated user
        
    Returns:
        Conversation information
    """
    try:
        conversation_id = await orchestrator.create_conversation(
            agent_ids=conversation.agent_ids,
            task_description=conversation.task_description,
            max_rounds=conversation.max_rounds
        )
        
        return orchestrator.get_conversation_status(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating conversation: {str(e)}")


@router.post("/{conversation_id}/start", response_model=ConversationResponse)
async def start_conversation(
    conversation_id: str,
    start_params: ConversationStart = Body(...),
    current_user: User = Depends(get_current_user)
):
    """
    Start a conversation with an initial message.
    
    Args:
        conversation_id: Conversation ID
        start_params: Start parameters including initial message
        current_user: Current authenticated user
        
    Returns:
        Updated conversation information
    """
    try:
        await orchestrator.start_conversation(
            conversation_id=conversation_id,
            initial_message=start_params.message
        )
        
        return orchestrator.get_conversation_status(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting conversation: {str(e)}")


@router.get("", response_model=List[ConversationResponse])
async def list_conversations(
    status: Optional[str] = Query(None, description="Filter by conversation status"),
    current_user: User = Depends(get_current_user)
):
    """
    List all conversations.
    
    Args:
        status: Optional status filter
        current_user: Current authenticated user
        
    Returns:
        List of conversation information
    """
    try:
        # Get all conversations
        conversations = [
            orchestrator.get_conversation_status(conv_id)
            for conv_id in orchestrator.conversations.keys()
        ]
        
        # Apply status filter if provided
        if status:
            conversations = [conv for conv in conversations if conv["status"] == status]
        
        return conversations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing conversations: {str(e)}")


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get conversation information.
    
    Args:
        conversation_id: Conversation ID
        current_user: Current authenticated user
        
    Returns:
        Conversation information
    """
    try:
        return orchestrator.get_conversation_status(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting conversation: {str(e)}")


@router.get("/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_conversation_messages(
    conversation_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get all messages in a conversation.
    
    Args:
        conversation_id: Conversation ID
        current_user: Current authenticated user
        
    Returns:
        List of messages
    """
    try:
        return orchestrator.get_conversation_messages(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting messages: {str(e)}")
