"""
Streamlit application for the MCP system.
This module provides a user interface for interacting with the MCP system.
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Any, Optional

import streamlit as st
import httpx
from streamlit_chat import message

# Configure page
st.set_page_config(
    page_title="Claude-Inspired MCP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Base URL
API_URL = os.environ.get("API_URL", "http://localhost:8000")


# Authentication functions
def get_auth_header():
    """Get authorization header with JWT token."""
    token = st.session_state.get("token")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


async def login(username: str, password: str) -> bool:
    """
    Login and get JWT token.
    
    Args:
        username: Username
        password: Password
        
    Returns:
        True if login successful, False otherwise
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_URL}/api/auth/token",
                data={"username": username, "password": password}
            )
            
            if response.status_code == 200:
                data = response.json()
                st.session_state["token"] = data["access_token"]
                st.session_state["username"] = username
                return True
            else:
                return False
    except Exception as e:
        st.error(f"Error during login: {str(e)}")
        return False


async def get_agents() -> List[Dict[str, Any]]:
    """
    Get list of available agents.
    
    Returns:
        List of agent information
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_URL}/api/agents",
                headers=get_auth_header()
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error fetching agents: {response.text}")
                return []
    except Exception as e:
        st.error(f"Error fetching agents: {str(e)}")
        return []


async def create_conversation(agent_ids: List[str], task_description: str) -> Optional[str]:
    """
    Create a new conversation.
    
    Args:
        agent_ids: List of agent IDs
        task_description: Task description
        
    Returns:
        Conversation ID if successful, None otherwise
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_URL}/api/conversations",
                json={
                    "agent_ids": agent_ids,
                    "task_description": task_description,
                    "max_rounds": 20
                },
                headers=get_auth_header()
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["id"]
            else:
                st.error(f"Error creating conversation: {response.text}")
                return None
    except Exception as e:
        st.error(f"Error creating conversation: {str(e)}")
        return None


async def start_conversation(conversation_id: str, message: str) -> bool:
    """
    Start a conversation with an initial message.
    
    Args:
        conversation_id: Conversation ID
        message: Initial message
        
    Returns:
        True if successful, False otherwise
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_URL}/api/conversations/{conversation_id}/start",
                json={"message": message},
                headers=get_auth_header()
            )
            
            if response.status_code == 200:
                return True
            else:
                st.error(f"Error starting conversation: {response.text}")
                return False
    except Exception as e:
        st.error(f"Error starting conversation: {str(e)}")
        return False


async def get_conversation_status(conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    Get conversation status.
    
    Args:
        conversation_id: Conversation ID
        
    Returns:
        Conversation status if successful, None otherwise
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_URL}/api/conversations/{conversation_id}",
                headers=get_auth_header()
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error getting conversation status: {response.text}")
                return None
    except Exception as e:
        st.error(f"Error getting conversation status: {str(e)}")
        return None


async def get_conversation_messages(conversation_id: str) -> List[Dict[str, Any]]:
    """
    Get conversation messages.
    
    Args:
        conversation_id: Conversation ID
        
    Returns:
        List of messages
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_URL}/api/conversations/{conversation_id}/messages",
                headers=get_auth_header()
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error getting conversation messages: {response.text}")
                return []
    except Exception as e:
        st.error(f"Error getting conversation messages: {str(e)}")
        return []


# UI Components
def login_form():
    """Display login form."""
    st.title("Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if asyncio.run(login(username, password)):
                st.success("Login successful!")
                time.sleep(1)
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")


def sidebar():
    """Display sidebar with navigation."""
    st.sidebar.title("Claude-Inspired MCP")
    
    if "username" in st.session_state:
        st.sidebar.write(f"Logged in as: {st.session_state['username']}")
        
        # Navigation
        page = st.sidebar.radio(
            "Navigation",
            options=["Conversations", "Agents", "Settings", "About"]
        )
        
        # Logout button
        if st.sidebar.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()
        
        return page
    
    return None


def display_agents_page():
    """Display agents page."""
    st.title("Available Agents")
    
    agents = asyncio.run(get_agents())
    
    if not agents:
        st.warning("No agents available")
        return
    
    # Display agents in columns
    cols = st.columns(3)
    
    for i, agent in enumerate(agents):
        with cols[i % 3]:
            agent_card(agent)


def agent_card(agent: Dict[str, Any]):
    """
    Display an agent card.
    
    Args:
        agent: Agent information
    """
    with st.container():
        st.subheader(agent["name"])
        st.write(agent["description"])
        
        # Display capabilities
        if agent.get("capabilities"):
            st.write("**Capabilities:**")
            for capability in agent["capabilities"]:
                st.write(f"- {capability['name']}: {capability['description']}")
        
        # Agent status
        status = "🟢 Active" if agent.get("is_active") else "🔴 Inactive"
        st.write(f"**Status:** {status}")


def display_conversations_page():
    """Display conversations page."""
    st.title("Conversations")
    
    # Tabs for new conversation and existing conversations
    tab1, tab2 = st.tabs(["New Conversation", "Existing Conversations"])
    
    with tab1:
        create_conversation_form()
    
    with tab2:
        view_conversations()


def create_conversation_form():
    """Display form to create a new conversation."""
    st.subheader("Create New Conversation")
    
    # Get available agents
    agents = asyncio.run(get_agents())
    
    if not agents:
        st.warning("No agents available to create a conversation")
        return
    
    # Create form
    with st.form("new_conversation_form"):
        # Select agents
        agent_options = {agent["name"]: agent["agent_id"] for agent in agents}
        selected_agents = st.multiselect(
            "Select Agents",
            options=list(agent_options.keys()),
            default=[list(agent_options.keys())[0]]
        )
        
        # Task description
        task_description = st.text_area(
            "Task Description",
            placeholder="Describe the task or conversation purpose..."
        )
        
        # Initial message
        initial_message = st.text_area(
            "Initial Message",
            placeholder="Enter your initial message to start the conversation..."
        )
        
        # Submit button
        submit = st.form_submit_button("Create Conversation")
        
        if submit:
            if not selected_agents:
                st.error("Please select at least one agent")
            elif not task_description:
                st.error("Please provide a task description")
            elif not initial_message:
                st.error("Please provide an initial message")
            else:
                # Get agent IDs
                agent_ids = [agent_options[name] for name in selected_agents]
                
                # Create conversation
                with st.spinner("Creating conversation..."):
                    conversation_id = asyncio.run(create_conversation(agent_ids, task_description))
                    
                    if conversation_id:
                        # Start conversation
                        success = asyncio.run(start_conversation(conversation_id, initial_message))
                        
                        if success:
                            st.success("Conversation created and started!")
                            
                            # Store conversation in session state
                            st.session_state["current_conversation"] = conversation_id
                            
                            # Redirect to conversation view
                            time.sleep(1)
                            st.experimental_rerun()
                        else:
                            st.error("Failed to start conversation")
                    else:
                        st.error("Failed to create conversation")


def view_conversations():
    """Display existing conversations."""
    st.subheader("Existing Conversations")
    
    # Get conversations from API
    conversations = []  # Replace with API call
    
    if not conversations:
        st.info("No existing conversations found")
        return
    
    # Display conversations
    for conversation in conversations:
        st.write(f"**Conversation:** {conversation['id']}")
        st.write(f"Status: {conversation['status']}")
        
        if st.button(f"View Conversation {conversation['id']}"):
            st.session_state["current_conversation"] = conversation["id"]
            st.experimental_rerun()


def display_conversation_view(conversation_id: str):
    """
    Display a conversation view.
    
    Args:
        conversation_id: Conversation ID
    """
    # Get conversation status
    status = asyncio.run(get_conversation_status(conversation_id))
    
    if not status:
        st.error(f"Conversation {conversation_id} not found")
        if st.button("Back to Conversations"):
            del st.session_state["current_conversation"]
            st.experimental_rerun()
        return
    
    # Display conversation info
    st.title(f"Conversation: {conversation_id}")
    st.write(f"**Status:** {status['status']}")
    st.write(f"**Round:** {status['current_round']}/{status['max_rounds']}")
    
    # Get messages
    messages = asyncio.run(get_conversation_messages(conversation_id))
    
    # Display messages
    st.subheader("Messages")
    
    if not messages:
        st.info("No messages yet")
    else:
        for msg in messages:
            content = msg["content"].get("message", "")
            sender = msg["sender"]
            
            # Determine if it's a user message
            is_user = sender.startswith("user")
            
            # Display message using streamlit_chat
            message(content, is_user=is_user, key=f"msg_{msg['id']}")
    
    # Back button
    if st.button("Back to Conversations"):
        del st.session_state["current_conversation"]
        st.experimental_rerun()


def display_settings_page():
    """Display settings page."""
    st.title("Settings")
    
    # API connection settings
    st.subheader("API Connection")
    
    api_url = st.text_input("API URL", value=API_URL)
    
    if st.button("Save Settings"):
        # Save settings
        st.session_state["api_url"] = api_url
        st.success("Settings saved!")


def display_about_page():
    """Display about page."""
    st.title("About Claude-Inspired MCP")
    
    st.write("""
    This application is a Claude-inspired Multi-Client Protocol (MCP) system using AutoGen as the orchestration engine.
    
    The system enables complex multi-agent interactions, with specialized agents for reasoning, research, coding, and conversation tasks.
    """)
    
    st.subheader("Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **MCP Protocol Implementation**
        - **AutoGen Orchestration**
        - **Specialized Agents**
        - **Tool Integration**
        """)
    
    with col2:
        st.markdown("""
        - **API Backend with FastAPI**
        - **UI Interface with Streamlit**
        - **Docker Containerization**
        - **Extensible Architecture**
        """)
    
    st.subheader("System Architecture")
    
    st.write("""
    The system follows a modular architecture with the following components:
    
    1. **MCP Protocol**: Message bus and communication channels
    2. **Orchestrator**: Manages agent interactions using AutoGen
    3. **Memory System**: Manages conversation and semantic memory
    4. **LLM Provider**: Interface to various LLM backends
    5. **Tool System**: Integrates external capabilities
    """)


# Main application
def main():
    """Main application entry point."""
    # Initialize session state
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Conversations"
    
    # Check if user is logged in
    if "token" not in st.session_state:
        login_form()
        return
    
    # Show sidebar and get selected page
    selected_page = sidebar()
    
    # If current conversation is set, show conversation view
    if "current_conversation" in st.session_state:
        display_conversation_view(st.session_state["current_conversation"])
        return
    
    # Display selected page
    if selected_page == "Conversations":
        display_conversations_page()
    elif selected_page == "Agents":
        display_agents_page()
    elif selected_page == "Settings":
        display_settings_page()
    elif selected_page == "About":
        display_about_page()


if __name__ == "__main__":
    main()
