#!/bin/bash

# Setup script for Claude-inspired MCP system
# This script initializes the project environment

# Ensure we're in the project root directory
cd "$(dirname "$0")/.."

# Create a virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    echo "Virtual environment created in 'venv' directory"
else
    echo "Virtual environment already exists"
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if dev flag is passed
if [ "$1" == "--dev" ]; then
    echo "Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Create required directories
echo "Creating required directories..."
mkdir -p data/conversations
mkdir -p tmp
mkdir -p logs

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "Created .env file. Please edit it to set your API keys"
else
    echo ".env file already exists"
fi

# Create config directory and example config
mkdir -p config
if [ ! -f "config/default.yaml" ]; then
    echo "Creating example configuration..."
    cat > config/default.yaml << EOL
# Default configuration for Claude-inspired MCP system

name: "Claude-Inspired MCP"
version: "0.1.0"
environment: "development"
log_level: "INFO"
data_dir: "./data"
temp_dir: "./tmp"
enable_docker: false

# API settings
api_host: "0.0.0.0"
api_port: 8000
api_workers: 4
api_timeout: 600
api_reload: true

# UI settings
ui_enabled: true
ui_host: "0.0.0.0"
ui_port: 8501

# LLM settings
default_llm_provider: "anthropic"
default_llm_model: "claude-3-7-sonnet-20250219"

# Agent settings
max_agents: 10
default_agent_timeout: 60
EOL
    echo "Created example configuration in config/default.yaml"
else
    echo "Configuration file already exists"
fi

# Make run script executable
chmod +x scripts/run_dev.sh

echo ""
echo "Setup complete! You can now run the system with:"
echo "  ./scripts/run_dev.sh"
echo ""
echo "But first, make sure to edit the .env file to set your API keys."