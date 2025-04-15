"""
Conversation agent implementation.
This agent specializes in natural language conversations and user interaction.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from src.agents.base_agent import BaseAgent, AgentCapability
from src.core.mcp import MCPMessage
from src.core.llm_provider import get_completion
from src.core.memory import ConversationMemory


class ConversationAgent(BaseAgent):
    """
    Agent specialized in natural language conversations and user interaction.
    
    This agent focuses on maintaining engaging, natural conversations and
    handling user interactions in a responsive and personable manner.
    """
    
    def __init__(self, 
                 agent_id: str,
                 name: str,
                 description: Optional[str] = None,
                 capabilities: Optional[List[AgentCapability]] = None,
                 system_message: Optional[str] = None,
                 model: str = "claude-3-7-sonnet-20250219",
                 temperature: float = 0.7,
                 memory_capacity: int = 20,
                 persona: Optional[Dict[str, Any]] = None):
        """
        Initialize a conversation agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            description: Description of the agent's purpose and functionality
            capabilities: List of capabilities the agent possesses
            system_message: System message to guide the agent's behavior
            model: LLM model to use
            temperature: Temperature setting for the LLM
            memory_capacity: Number of conversation turns to remember
            persona: Optional personality traits and characteristics
        """
        if not system_message:
            system_message = (
                f"You are {name}, an AI assistant specialized in natural, engaging conversations. "
                "Your responses should be friendly, personable, and appropriately detailed. "
                "Adapt your tone and style to the context of the conversation and the apparent "
                "preferences of the human. Ask meaningful follow-up questions when appropriate, "
                "but don't be overly inquisitive. Your goal is to be helpful and create a positive "
                "interaction experience."
            )
            
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description or f"Conversation agent focused on natural user interactions",
            capabilities=capabilities or [],
            system_message=system_message
        )
        
        # Add default conversation capabilities
        if not self.has_capability("natural_conversation"):
            self.add_capability(AgentCapability(
                name="natural_conversation",
                description="Engage in natural, flowing conversations"
            ))
            
        if not self.has_capability("contextual_understanding"):
            self.add_capability(AgentCapability(
                name="contextual_understanding",
                description="Maintain and utilize conversation context"
            ))
        
        self.model = model
        self.temperature = temperature
        self.conversation_memory = ConversationMemory(capacity=memory_capacity)
        self.persona = persona or {
            "friendliness": 0.8,
            "formality": 0.5,
            "verbosity": 0.6,
            "curiosity": 0.7,
            "humor": 0.5
        }
    
    async def process_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """
        Process an incoming conversation message.
        
        Args:
            message: The incoming message to process
            
        Returns:
            A response message continuing the conversation
        """
        user_message = message.content.get("message", "")
        if not user_message:
            logger.warning(f"Empty message received from {message.sender}")
            return None
        
        # Add to conversation memory
        self.conversation_memory.add_user_message(user_message)
        
        # Generate a response
        conversation_history = self.conversation_memory.get_recent_messages()
        response = await self._generate_response(user_message, conversation_history)
        
        # Add to conversation memory
        self.conversation_memory.add_assistant_message(response)
        
        # Create and return the response
        return MCPMessage(
            sender=self.agent_id,
            receiver=message.sender,
            content={"message": response},
            reply_to=message.id,
            trace_id=message.trace_id,
            metadata=message.metadata
        )
    
    async def _generate_response(self, message: str, history: List[Dict[str, str]]) -> str:
        """
        Generate a response based on the message and conversation history.
        
        Args:
            message: The current message to respond to
            history: Conversation history
            
        Returns:
            The generated response
        """
        # Build the prompt with conversation history
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        
        # Add conversation history
        for item in history:
            role = "user" if item["type"] == "user" else "assistant"
            messages.append({"role": role, "content": item["content"]})
        
        # If the current message isn't the last one in history, add it
        if not history or history[-1]["type"] != "user" or history[-1]["content"] != message:
            messages.append({"role": "user", "content": message})
        
        # Get completion from LLM
        response = await get_completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        
        return response
    
    def _build_system_prompt(self) -> str:
        """
        Build a system prompt that incorporates persona characteristics.
        
        Returns:
            The customized system prompt
        """
        base_prompt = self.system_message
        
        # Add persona-specific instructions
        persona_instructions = []
        
        if self.persona["friendliness"] > 0.7:
            persona_instructions.append("Be very warm and friendly in your responses.")
        elif self.persona["friendliness"] < 0.3:
            persona_instructions.append("Maintain a more neutral, professional tone.")
        
        if self.persona["formality"] > 0.7:
            persona_instructions.append("Use formal language and phrasing.")
        elif self.persona["formality"] < 0.3:
            persona_instructions.append("Use casual, conversational language.")
        
        if self.persona["verbosity"] > 0.7:
            persona_instructions.append("Provide detailed, comprehensive responses.")
        elif self.persona["verbosity"] < 0.3:
            persona_instructions.append("Keep responses concise and to the point.")
        
        if self.persona["curiosity"] > 0.7:
            persona_instructions.append("Show curiosity by asking thoughtful follow-up questions occasionally.")
        
        if self.persona["humor"] > 0.7:
            persona_instructions.append("Incorporate appropriate light humor when natural.")
        
        # Add persona instructions to prompt if any exist
        if persona_instructions:
            persona_prompt = "\n\nPersonality guidelines:\n" + "\n".join(f"- {i}" for i in persona_instructions)
            return base_prompt + persona_prompt
            
        return base_prompt