"""
Coding agent implementation.
This agent specializes in code generation, analysis, and problem-solving.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from src.agents.base_agent import BaseAgent, AgentCapability
from src.core.mcp import MCPMessage
from src.core.llm_provider import get_completion
from src.agents.tools.code_executor import execute_code


class CodingAgent(BaseAgent):
    """
    Agent specialized in code generation and software development tasks.
    
    This agent focuses on writing high-quality code, debugging, and
    providing explanations of coding concepts and implementations.
    """
    
    def __init__(self, 
                 agent_id: str,
                 name: str,
                 description: Optional[str] = None,
                 capabilities: Optional[List[AgentCapability]] = None,
                 system_message: Optional[str] = None,
                 model: str = "claude-3-7-sonnet-20250219",
                 temperature: float = 0.2,
                 supported_languages: Optional[List[str]] = None,
                 use_docker: bool = False):
        """
        Initialize a coding agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            description: Description of the agent's purpose and functionality
            capabilities: List of capabilities the agent possesses
            system_message: System message to guide the agent's behavior
            model: LLM model to use
            temperature: Temperature setting for the LLM
            supported_languages: List of programming languages the agent supports
            use_docker: Whether to use Docker for code execution
        """
        if not system_message:
            system_message = (
                f"You are {name}, an AI assistant specialized in software development and coding. "
                "Your responses should focus on writing clean, efficient, and well-documented code. "
                "Always explain your approach and the rationale behind implementation choices. "
                "When providing code, include comments to explain complex or non-obvious parts. "
                "Consider edge cases and potential optimizations. If asked to debug or improve "
                "existing code, analyze it thoroughly and suggest specific improvements."
            )
            
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description or f"Coding agent specialized in software development",
            capabilities=capabilities or [],
            system_message=system_message
        )
        
        # Default supported languages if none provided
        self.supported_languages = supported_languages or [
            "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", 
            "Go", "Rust", "Ruby", "PHP", "Swift", "Kotlin", "SQL", "HTML/CSS"
        ]
        
        # Add default coding capabilities
        if not self.has_capability("code_generation"):
            self.add_capability(AgentCapability(
                name="code_generation",
                description="Generate high-quality code in multiple languages"
            ))
            
        if not self.has_capability("code_explanation"):
            self.add_capability(AgentCapability(
                name="code_explanation",
                description="Explain code functionality and implementation details"
            ))
            
        if not self.has_capability("code_debugging"):
            self.add_capability(AgentCapability(
                name="code_debugging",
                description="Identify and fix issues in code"
            ))
        
        if use_docker and not self.has_capability("code_execution"):
            self.add_capability(AgentCapability(
                name="code_execution",
                description="Execute code in a sandboxed environment"
            ))
        
        self.model = model
        self.temperature = temperature
        self.use_docker = use_docker
    
    async def process_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """
        Process an incoming coding-related message.
        
        Args:
            message: The incoming message to process
            
        Returns:
            A response message with code or code-related information
        """
        user_message = message.content.get("message", "")
        if not user_message:
            logger.warning(f"Empty message received from {message.sender}")
            return None
        
        # Determine if this is a request to execute code
        execution_request = False
        execution_language = None
        execution_code = None
        
        if "execute" in user_message.lower() or "run" in user_message.lower():
            # Try to extract code blocks for execution
            code_blocks = self._extract_code_blocks(user_message)
            if code_blocks:
                # Use the first code block for execution
                execution_language, execution_code = code_blocks[0]
                execution_request = True
        
        # Handle execution request if identified and supported
        if execution_request and self.use_docker and execution_code:
            execution_result = await self._execute_code(execution_language, execution_code)
            response = await self._format_execution_result(user_message, execution_language, execution_code, execution_result)
        else:
            # Regular code generation or explanation
            response = await self._generate_code_response(user_message)
        
        # Create and return the response
        return MCPMessage(
            sender=self.agent_id,
            receiver=message.sender,
            content={"message": response},
            reply_to=message.id,
            trace_id=message.trace_id,
            metadata=message.metadata
        )
    
    async def _generate_code_response(self, message: str) -> str:
        """
        Generate a coding-related response.
        
        Args:
            message: The message to respond to
            
        Returns:
            The generated response with code
        """
        # Enhance the system prompt with language-specific information
        enhanced_prompt = (
            f"{self.system_message}\n\n"
            f"Supported programming languages: {', '.join(self.supported_languages)}.\n"
            "When generating code, follow these guidelines:\n"
            "1. Include clear comments explaining the code's functionality\n"
            "2. Follow best practices for the chosen language\n"
            "3. Consider edge cases and error handling\n"
            "4. Provide docstrings or function headers as appropriate\n"
            "5. Use proper formatting and naming conventions\n"
            "6. Wrap code blocks in Markdown triple backticks with language specified"
        )
        
        # Get completion from LLM
        response = await get_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": enhanced_prompt},
                {"role": "user", "content": message}
            ],
            temperature=self.temperature
        )
        
        return response
    
    async def _execute_code(self, language: str, code: str) -> Dict[str, Any]:
        """
        Execute code in a sandboxed environment.
        
        Args:
            language: Programming language of the code
            code: Code to execute
            
        Returns:
            Execution result
        """
        if not self.use_docker:
            return {"error": "Code execution is not enabled for this agent"}
        
        # Map language to executor environment
        language_map = {
            "python": "python",
            "javascript": "node",
            "typescript": "ts-node",
            "ruby": "ruby",
            "php": "php",
            "bash": "bash",
            "shell": "bash"
        }
        
        exec_language = language_map.get(language.lower(), language.lower())
        
        try:
            result = await execute_code(exec_language, code)
            return result
        except Exception as e:
            logger.error(f"Error executing code: {e}")
            return {"error": str(e)}
    
    async def _format_execution_result(self, 
                                      message: str, 
                                      language: str, 
                                      code: str, 
                                      result: Dict[str, Any]) -> str:
        """
        Format code execution result into a readable response.
        
        Args:
            message: Original user message
            language: Programming language of the code
            code: Executed code
            result: Execution result
            
        Returns:
            Formatted response with execution result
        """
        # Format the execution result for display
        if "error" in result:
            execution_output = f"Error: {result['error']}"
            
            # Get suggestions to fix the error
            fix_suggestions = await get_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"{self.system_message}\n\nYou are helping debug code that produced an error. Analyze the error and suggest fixes."},
                    {"role": "user", "content": f"This {language} code:\n\n```{language}\n{code}\n```\n\nProduced this error:\n\n{result['error']}\n\nWhat's causing the error and how can I fix it?"}
                ],
                temperature=self.temperature
            )
            
            response = (
                f"I executed your {language} code, but it resulted in an error:\n\n"
                f"```\n{execution_output}\n```\n\n"
                f"### Debugging Suggestions\n\n{fix_suggestions}"
            )
        else:
            execution_output = result.get("output", "No output")
            execution_time = result.get("execution_time", "Unknown")
            
            # Get explanation of the code and output
            explanation = await get_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"{self.system_message}\n\nYou are explaining code execution results. Be concise but informative."},
                    {"role": "user", "content": f"This {language} code:\n\n```{language}\n{code}\n```\n\nProduced this output:\n\n{execution_output}\n\nPlease briefly explain what the code does and interpret the output."}
                ],
                temperature=self.temperature
            )
            
            response = (
                f"I executed your {language} code. Here are the results:\n\n"
                f"### Output\n```\n{execution_output}\n```\n\n"
                f"### Execution Time\n{execution_time} seconds\n\n"
                f"### Explanation\n{explanation}"
            )
        
        return response
    
    def _extract_code_blocks(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract code blocks from text.
        
        Args:
            text: Text containing code blocks
            
        Returns:
            List of (language, code) tuples
        """
        # Match Markdown code blocks with language specification
        pattern = r"```(\w+)\n([\s\S]*?)```"
        code_blocks = re.findall(pattern, text)
        
        # Also try to match code blocks without language specification
        if not code_blocks:
            pattern = r"```\n([\s\S]*?)```"
            no_lang_blocks = re.findall(pattern, text)
            code_blocks = [("text", block) for block in no_lang_blocks]
        
        return code_blocks