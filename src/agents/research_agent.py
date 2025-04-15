"""
Research agent implementation.
This agent specializes in research, information gathering, and analysis.
"""

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from src.agents.base_agent import BaseAgent, AgentCapability
from src.core.mcp import MCPMessage
from src.core.llm_provider import get_completion
from src.agents.tools.web_search import search_web
from src.agents.tools.data_analyzer import analyze_data


class ResearchAgent(BaseAgent):
    """
    Agent specialized in research and information gathering.
    
    This agent focuses on collecting, synthesizing, and analyzing information
    from various sources to provide comprehensive research results.
    """
    
    def __init__(self, 
                 agent_id: str,
                 name: str,
                 description: Optional[str] = None,
                 capabilities: Optional[List[AgentCapability]] = None,
                 system_message: Optional[str] = None,
                 model: str = "claude-3-7-sonnet-20250219",
                 temperature: float = 0.3,
                 enable_web_search: bool = True,
                 enable_data_analysis: bool = True):
        """
        Initialize a research agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            description: Description of the agent's purpose and functionality
            capabilities: List of capabilities the agent possesses
            system_message: System message to guide the agent's behavior
            model: LLM model to use
            temperature: Temperature setting for the LLM
            enable_web_search: Whether to enable web search capability
            enable_data_analysis: Whether to enable data analysis capability
        """
        if not system_message:
            system_message = (
                f"You are {name}, an AI assistant specialized in research and information gathering. "
                "Your goal is to provide comprehensive, accurate, and well-organized research on a given topic. "
                "When conducting research, gather information from multiple sources, critically evaluate "
                "the reliability of each source, and synthesize the information into a coherent analysis. "
                "Identify key findings, highlight areas of consensus and controversy, and note any gaps "
                "in the available information. Present results in a structured format with proper citations "
                "when possible."
            )
            
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description or f"Research agent specialized in information gathering and analysis",
            capabilities=capabilities or [],
            system_message=system_message
        )
        
        # Add default research capabilities
        if not self.has_capability("information_synthesis"):
            self.add_capability(AgentCapability(
                name="information_synthesis",
                description="Synthesize information from multiple sources"
            ))
            
        if not self.has_capability("critical_evaluation"):
            self.add_capability(AgentCapability(
                name="critical_evaluation",
                description="Critically evaluate sources and information quality"
            ))
        
        if enable_web_search and not self.has_capability("web_search"):
            self.add_capability(AgentCapability(
                name="web_search",
                description="Search the web for information"
            ))
            
        if enable_data_analysis and not self.has_capability("data_analysis"):
            self.add_capability(AgentCapability(
                name="data_analysis",
                description="Analyze datasets and extract insights"
            ))
        
        self.model = model
        self.temperature = temperature
        self.enable_web_search = enable_web_search
        self.enable_data_analysis = enable_data_analysis
    
    async def process_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """
        Process an incoming research-related message.
        
        Args:
            message: The incoming message to process
            
        Returns:
            A response message with research results
        """
        user_message = message.content.get("message", "")
        if not user_message:
            logger.warning(f"Empty message received from {message.sender}")
            return None
        
        # Determine what type of research request this is
        request_type = await self._determine_request_type(user_message)
        
        # Handle the research request based on type
        if request_type == "web_search" and self.enable_web_search:
            search_query = await self._extract_search_query(user_message)
            research_results = await self._conduct_web_research(search_query)
        elif request_type == "data_analysis" and self.enable_data_analysis:
            data_source = message.content.get("data_source")
            if data_source:
                research_results = await self._analyze_data(user_message, data_source)
            else:
                research_results = "I'd be happy to analyze data for you, but I need a data source. Please provide a dataset or data file to analyze."
        else:
            # Default to LLM-based research
            research_results = await self._conduct_general_research(user_message)
        
        # Create and return the response
        return MCPMessage(
            sender=self.agent_id,
            receiver=message.sender,
            content={"message": research_results},
            reply_to=message.id,
            trace_id=message.trace_id,
            metadata=message.metadata
        )
    
    async def _determine_request_type(self, message: str) -> str:
        """
        Determine the type of research request.
        
        Args:
            message: The message to analyze
            
        Returns:
            Request type: "web_search", "data_analysis", or "general"
        """
        # Check for web search indicators
        web_search_indicators = [
            "search for", "find information", "look up", "search the web",
            "recent information", "latest on", "current news", "find out about"
        ]
        
        # Check for data analysis indicators
        data_analysis_indicators = [
            "analyze this data", "data analysis", "dataset", "spreadsheet",
            "csv file", "excel file", "statistics", "trends", "patterns",
            "correlations", "insights from data", "visualize data"
        ]
        
        # Check for indicators in the message
        message_lower = message.lower()
        
        if any(indicator in message_lower for indicator in web_search_indicators) and self.enable_web_search:
            return "web_search"
        elif any(indicator in message_lower for indicator in data_analysis_indicators) and self.enable_data_analysis:
            return "data_analysis"
        else:
            return "general"
    
    async def _extract_search_query(self, message: str) -> str:
        """
        Extract the search query from a message.
        
        Args:
            message: The message to extract from
            
        Returns:
            The extracted search query
        """
        # Use the LLM to extract a clear search query
        search_query = await get_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a search query extractor. Given a user's research request, extract the most effective search query that will yield the most relevant results. The query should be concise but specific enough to return targeted information. Do not include any explanations or additional text in your response - just output the search query itself."},
                {"role": "user", "content": message}
            ],
            temperature=0.1  # Low temperature for focused extraction
        )
        
        return search_query.strip()
    
    async def _conduct_web_research(self, query: str) -> str:
        """
        Conduct web-based research on a topic.
        
        Args:
            query: The search query
            
        Returns:
            Research results
        """
        if not self.enable_web_search:
            return "Web search capability is not enabled for this agent."
        
        try:
            # Perform web search
            search_results = await search_web(query, max_results=5)
            
            # Create a summary of search results
            search_summary = "\n\n".join([
                f"Source: {result['title']} ({result['url']})\n{result['snippet']}"
                for result in search_results
            ])
            
            # Synthesize research findings
            research_report = await get_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"{self.system_message}\n\nYou are preparing a research report based on search results. Synthesize the information, evaluate source reliability, and present findings in a structured, comprehensive manner. Include citations where appropriate."},
                    {"role": "user", "content": f"I need research on: {query}\n\nHere are the search results:\n\n{search_summary}\n\nPlease synthesize this information into a comprehensive research report."}
                ],
                temperature=self.temperature
            )
            
            return research_report
        except Exception as e:
            logger.error(f"Error conducting web research: {e}")
            return f"I encountered an error while conducting web research: {str(e)}. I'll try to answer based on my knowledge instead.\n\n" + await self._conduct_general_research(query)
    
    async def _analyze_data(self, message: str, data_source: str) -> str:
        """
        Analyze data and provide insights.
        
        Args:
            message: The research request
            data_source: Source of the data
            
        Returns:
            Analysis results
        """
        if not self.enable_data_analysis:
            return "Data analysis capability is not enabled for this agent."
        
        try:
            # Analyze the data
            analysis_results = await analyze_data(data_source, message)
            
            # Generate insights based on analysis
            insights = await get_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"{self.system_message}\n\nYou are interpreting data analysis results and extracting meaningful insights. Focus on key patterns, trends, and important findings. Structure your response with clear sections and highlight the most significant discoveries."},
                    {"role": "user", "content": f"Data analysis request: {message}\n\nAnalysis results:\n\n{analysis_results}\n\nPlease provide insights and interpretations based on these results."}
                ],
                temperature=self.temperature
            )
            
            return f"# Data Analysis Results\n\n{analysis_results}\n\n# Insights and Interpretation\n\n{insights}"
        except Exception as e:
            logger.error(f"Error analyzing data: {e}")
            return f"I encountered an error while analyzing the data: {str(e)}. Please check that the data source is valid and try again."
    
    async def _conduct_general_research(self, topic: str) -> str:
        """
        Conduct general research on a topic using the LLM's knowledge.
        
        Args:
            topic: The research topic
            
        Returns:
            Research results
        """
        # Build a comprehensive research prompt
        research_prompt = (
            f"{self.system_message}\n\n"
            "Provide a comprehensive research report on the given topic. Your report should:\n"
            "1. Begin with an executive summary or overview\n"
            "2. Cover key aspects and dimensions of the topic\n"
            "3. Present multiple perspectives or approaches when relevant\n"
            "4. Cite relevant concepts, theories, or frameworks\n"
            "5. Acknowledge limitations in current understanding\n"
            "6. End with conclusions or recommendations if appropriate\n\n"
            "Structure your response in a clear, organized manner with appropriate headings and subheadings."
        )
        
        # Generate the research report
        research_report = await get_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": research_prompt},
                {"role": "user", "content": f"I need comprehensive research on: {topic}"}
            ],
            temperature=self.temperature
        )
        
        return research_report
