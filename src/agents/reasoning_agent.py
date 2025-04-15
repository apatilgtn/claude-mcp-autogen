"""
Reasoning agent implementation.
This agent specializes in complex reasoning and problem-solving tasks.
"""

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from src.agents.base_agent import BaseAgent, AgentCapability
from src.core.mcp import MCPMessage
from src.core.llm_provider import get_completion


class ReasoningAgent(BaseAgent):
    """
    Agent specialized in complex reasoning and problem-solving tasks.
    
    This agent uses a multi-step reasoning approach and is designed to handle
    tasks that require careful analysis and logical thinking.
    """
    
    def __init__(self, 
                 agent_id: str,
                 name: str,
                 description: Optional[str] = None,
                 capabilities: Optional[List[AgentCapability]] = None,
                 system_message: Optional[str] = None,
                 model: str = "claude-3-7-sonnet-20250219",
                 temperature: float = 0.2,
                 max_reasoning_steps: int = 5):
        """
        Initialize a reasoning agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            description: Description of the agent's purpose and functionality
            capabilities: List of capabilities the agent possesses
            system_message: System message to guide the agent's behavior
            model: LLM model to use
            temperature: Temperature setting for the LLM
            max_reasoning_steps: Maximum number of reasoning steps
        """
        if not system_message:
            system_message = (
                f"You are {name}, an AI assistant specialized in complex reasoning. "
                "When presented with a problem, break it down into steps and think through "
                "each part carefully. Consider multiple perspectives and approaches before "
                "drawing conclusions. Your reasoning should be explicit, thorough, and logical. "
                "Explain your thought process clearly and identify any assumptions you make."
            )
            
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description or f"Reasoning agent specialized in complex problem-solving",
            capabilities=capabilities or [],
            system_message=system_message
        )
        
        # Add default reasoning capabilities
        if not self.has_capability("step_by_step_reasoning"):
            self.add_capability(AgentCapability(
                name="step_by_step_reasoning",
                description="Break down complex problems into sequential steps"
            ))
            
        if not self.has_capability("multi_perspective_analysis"):
            self.add_capability(AgentCapability(
                name="multi_perspective_analysis",
                description="Analyze problems from multiple perspectives or frameworks"
            ))
        
        self.model = model
        self.temperature = temperature
        self.max_reasoning_steps = max_reasoning_steps
    
    async def process_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """
        Process an incoming message with a reasoning approach.
        
        Args:
            message: The incoming message to process
            
        Returns:
            A response message with the reasoning result
        """
        user_message = message.content.get("message", "")
        if not user_message:
            logger.warning(f"Empty message received from {message.sender}")
            return None
        
        # Start the reasoning process
        reasoning_result = await self._reason(user_message)
        
        # Create and return the response
        return MCPMessage(
            sender=self.agent_id,
            receiver=message.sender,
            content={"message": reasoning_result},
            reply_to=message.id,
            trace_id=message.trace_id,
            metadata=message.metadata
        )
    
    async def _reason(self, query: str) -> str:
        """
        Apply multi-step reasoning to a query.
        
        Args:
            query: The query to reason about
            
        Returns:
            The reasoning result
        """
        # Step 1: Understand the problem
        problem_understanding = await get_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": f"{self.system_message}\n\nYour first task is to understand the problem thoroughly. Identify the key aspects of the problem, constraints, and what is being asked."},
                {"role": "user", "content": query}
            ],
            temperature=self.temperature
        )
        
        # Step 2: Break down into sub-problems
        problem_breakdown = await get_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": f"{self.system_message}\n\nNow break down the problem into smaller, more manageable sub-problems or components."},
                {"role": "user", "content": query},
                {"role": "assistant", "content": problem_understanding},
                {"role": "user", "content": "Please break this down into sub-problems or components that need to be addressed."}
            ],
            temperature=self.temperature
        )
        
        # Step 3: Generate alternative approaches
        approaches = await get_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": f"{self.system_message}\n\nGenerate multiple approaches or strategies to solve the problem."},
                {"role": "user", "content": query},
                {"role": "assistant", "content": problem_understanding + "\n\n" + problem_breakdown},
                {"role": "user", "content": "What are different approaches or strategies we could use to solve this?"}
            ],
            temperature=self.temperature * 1.5  # Slightly higher temperature for creative approaches
        )
        
        # Step 4: Evaluate approaches and select best
        evaluation = await get_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": f"{self.system_message}\n\nEvaluate the different approaches and select the most promising one, explaining your reasoning."},
                {"role": "user", "content": query},
                {"role": "assistant", "content": problem_understanding + "\n\n" + problem_breakdown + "\n\n" + approaches},
                {"role": "user", "content": "Please evaluate these approaches and select the most promising one."}
            ],
            temperature=self.temperature
        )
        
        # Step 5: Execute chosen approach and formulate final answer
        solution = await get_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": f"{self.system_message}\n\nImplement the chosen approach to solve the problem and provide a final answer."},
                {"role": "user", "content": query},
                {"role": "assistant", "content": problem_understanding + "\n\n" + problem_breakdown + "\n\n" + approaches + "\n\n" + evaluation},
                {"role": "user", "content": "Please implement this approach and provide the final answer or solution."}
            ],
            temperature=self.temperature
        )
        
        # Optional: Create a clean, clear final response
        final_response = await get_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": f"{self.system_message}\n\nCreate a clear, concise final response that addresses the original query. Your response should be well-structured and easy to understand."},
                {"role": "user", "content": query},
                {"role": "assistant", "content": solution},
                {"role": "user", "content": "Please provide a clear, well-structured final response to the original query."}
            ],
            temperature=self.temperature
        )
        
        # Store the reasoning steps in memory
        self.remember("last_reasoning", {
            "query": query,
            "understanding": problem_understanding,
            "breakdown": problem_breakdown,
            "approaches": approaches,
            "evaluation": evaluation,
            "solution": solution,
            "final_response": final_response
        })
        
        return final_response
