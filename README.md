# Claude-Inspired MCP System with AutoGen

## Overview

This project implements a Claude-inspired Multi-Client Protocol (MCP) system using AutoGen as the orchestration engine. The system provides a sophisticated agent framework that enables complex, multi-agent interactions in a highly modular and extensible architecture.

## Key Features

- **MCP Protocol Implementation**: Multi-agent communication framework inspired by Claude's architecture
- **AutoGen Orchestration**: Leverages AutoGen for agent coordination and task management
- **Specialized Agents**:
  - **Reasoning Agent**: Specialized in complex reasoning and problem-solving
  - **Research Agent**: Focused on information gathering and analysis
  - **Coding Agent**: Generates code and provides programming assistance
  - **Conversation Agent**: Handles natural language interactions
- **Tool Integration**:
  - Web search capabilities
  - Code execution in sandboxed environments
  - Data analysis utilities
  - File management system
- **API Backend**: FastAPI implementation for HTTP access to the agent system
- **UI Interface**: Streamlit-based UI for interactive access and visualization
- **Containerization**: Docker and Docker Compose support for easy deployment

## Architecture

The system follows a modular architecture:

```
claude-mcp-autogen/
├── src/
│   ├── agents/         # Agent implementations and tools
│   ├── core/           # Core MCP protocol and orchestration logic
│   ├── api/            # FastAPI application
│   ├── ui/             # Streamlit user interface
│   └── utils/          # Utilities and helpers
```

### Core Components

1. **MCP Protocol**: Provides the message bus and communication channels for agents
2. **Orchestrator**: Manages agent interactions and conversation flow using AutoGen
3. **Memory System**: Manages conversation history, semantic knowledge, and episodic memory
4. **LLM Provider**: Interface to various LLM backends (Claude, GPT)
5. **Tool System**: Integrates external capabilities like web search and code execution

## Getting Started

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (optional, for containerized deployment)
- API keys for LLM providers (Anthropic, OpenAI)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/claude-mcp-autogen.git
   cd claude-mcp-autogen
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a configuration file:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

### Running the System

#### Using Docker (recommended)

```bash
docker-compose up
```

This will start both the API server and the UI in separate containers.

#### Manual Start

1. Start the API server:
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. Start the UI (in a separate terminal):
   ```bash
   streamlit run src/ui/streamlit_app.py
   ```

### Configuration

The system can be configured through:
- Environment variables (prefixed with `MCP_`)
- Configuration files in the `config/` directory
- Command line parameters

## Usage Examples

### Creating an Agent Conversation

```python
from src.core.orchestrator import orchestrator

# Define agent IDs
agent_ids = ["reasoning-agent", "research-agent"]

# Create a conversation
conversation_id = await orchestrator.create_conversation(
    agent_ids=agent_ids,
    task_description="Analyze the impact of quantum computing on cryptography",
    max_rounds=10
)

# Start the conversation
await orchestrator.start_conversation(
    conversation_id=conversation_id,
    initial_message="What are the main implications of quantum computing for current encryption methods?"
)

# Get conversation results
status = orchestrator.get_conversation_status(conversation_id)
messages = orchestrator.get_conversation_messages(conversation_id)
```

### Using the API

```bash
# Create a new conversation
curl -X POST "http://localhost:8000/api/conversations" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_ids": ["reasoning-agent", "research-agent"],
    "task_description": "Analyze quantum computing impact on cryptography",
    "max_rounds": 10
  }'

# Start the conversation
curl -X POST "http://localhost:8000/api/conversations/{conversation_id}/start" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the main implications of quantum computing for current encryption methods?"
  }'
```

## Extending the System

### Adding New Agents

Create a new agent class inheriting from `BaseAgent` in the `src/agents/` directory.

### Adding New Tools

Implement new tools in the `src/agents/tools/` directory and register them with the appropriate agents.

## License

[MIT License](LICENSE)

## Acknowledgments

- This project is inspired by Claude's architecture and capabilities
- Built on the AutoGen framework for agent orchestration
- Uses FastAPI and Streamlit for the backend and UI
