#!/bin/bash

# Development runner script for Claude-inspired MCP system
# This script starts the FastAPI and Streamlit applications in development mode

# Ensure we're in the project root directory
cd "$(dirname "$0")/.."

# Set environment variables
export PYTHONPATH=$(pwd)
export MCP_ENVIRONMENT=development
export MCP_LOG_LEVEL=DEBUG

# Check for .env file and load it if present
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    set -a
    source .env
    set +a
fi

# Check if required environment variables are set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY is not set"
    echo "Please set it in your .env file or environment"
    exit 1
fi

# Create required directories if they don't exist
mkdir -p data/conversations
mkdir -p tmp

# Function to start the API server
start_api() {
    echo "Starting FastAPI server on port 8000"
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
}

# Function to start the UI
start_ui() {
    echo "Starting Streamlit UI on port 8501"
    streamlit run src/ui/streamlit_app.py
}

# Check command line arguments
if [ "$1" == "api" ]; then
    start_api
elif [ "$1" == "ui" ]; then
    start_ui
elif [ "$1" == "all" ] || [ -z "$1" ]; then
    # Start both in separate processes
    echo "Starting both API and UI"
    
    # Start API in background
    start_api &
    API_PID=$!
    
    # Wait a bit for API to start
    sleep 2
    
    # Start UI in background
    start_ui &
    UI_PID=$!
    
    # Handle termination
    trap "kill $API_PID $UI_PID; exit" SIGINT SIGTERM
    
    # Wait for both processes
    wait $API_PID $UI_PID
else
    echo "Unknown command: $1"
    echo "Usage: $0 [api|ui|all]"
    exit 1
fi
