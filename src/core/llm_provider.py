"""
LLM Provider for the MCP system.
This module provides a unified interface to various LLM providers.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Union

import anthropic
from anthropic import Anthropic
from loguru import logger

from src.core.config import settings


async def get_completion(
    messages: List[Dict[str, str]],
    model: str = "claude-3-7-sonnet-20250219",
    temperature: float = 0.7,
    max_tokens: int = 4000,
    top_p: float = 0.95,
    stop_sequences: Optional[List[str]] = None,
    stream: bool = False
) -> str:
    """
    Get a completion from an LLM.
    
    Args:
        messages: List of messages in the conversation
        model: LLM model to use
        temperature: Temperature setting
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling parameter
        stop_sequences: Optional stop sequences
        stream: Whether to stream the response
        
    Returns:
        The generated completion text
    """
    provider = _determine_provider(model)
    
    if provider == "anthropic":
        return await _anthropic_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop_sequences=stop_sequences,
            stream=stream
        )
    elif provider == "openai":
        return await _openai_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop_sequences=stop_sequences,
            stream=stream
        )
    else:
        raise ValueError(f"Unknown provider for model {model}")


def _determine_provider(model: str) -> str:
    """
    Determine the provider based on the model name.
    
    Args:
        model: Model name
        
    Returns:
        Provider name
    """
    if model.startswith(("claude", "anthropic")):
        return "anthropic"
    elif model.startswith(("gpt", "text-davinci", "text-embedding")):
        return "openai"
    else:
        # Default to Anthropic
        return "anthropic"


async def _anthropic_completion(
    messages: List[Dict[str, str]],
    model: str = "claude-3-7-sonnet-20250219",
    temperature: float = 0.7,
    max_tokens: int = 4000,
    top_p: float = 0.95,
