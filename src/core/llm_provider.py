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
    stop_sequences: Optional[List[str]] = None,
    stream: bool = False
) -> str:
    """
    Get a completion from Anthropic's Claude models.
    
    Args:
        messages: List of messages in the conversation
        model: Claude model to use
        temperature: Temperature setting
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling parameter
        stop_sequences: Optional stop sequences
        stream: Whether to stream the response
        
    Returns:
        The generated completion text
    """
    api_key = settings.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Anthropic API key not found in settings")
    
    client = Anthropic(api_key=api_key)
    
    try:
        if stream:
            # Streaming response
            response = client.messages.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_sequences,
                stream=True
            )
            
            # Accumulate streamed content
            content = ""
            async for chunk in response:
                if chunk.type == "content_block_delta":
                    content += chunk.delta.text
            
            return content
        else:
            # Non-streaming response
            response = client.messages.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_sequences
            )
            
            return response.content[0].text
    except Exception as e:
        logger.error(f"Error with Anthropic API: {e}")
        raise


async def _openai_completion(
    messages: List[Dict[str, str]],
    model: str = "gpt-4",
    temperature: float = 0.7,
    max_tokens: int = 4000,
    top_p: float = 0.95,
    stop_sequences: Optional[List[str]] = None,
    stream: bool = False
) -> str:
    """
    Get a completion from OpenAI models.
    
    Args:
        messages: List of messages in the conversation
        model: OpenAI model to use
        temperature: Temperature setting
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling parameter
        stop_sequences: Optional stop sequences
        stream: Whether to stream the response
        
    Returns:
        The generated completion text
    """
    import openai
    
    api_key = settings.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in settings")
    
    openai.api_key = api_key
    
    try:
        if stream:
            # Streaming response
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_sequences,
                stream=True
            )
            
            # Accumulate streamed content
            content = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.get("content"):
                    content += chunk.choices[0].delta.content
            
            return content
        else:
            # Non-streaming response
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_sequences
            )
            
            return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error with OpenAI API: {e}")
        raise


async def embed_text(
    text: str,
    model: str = "text-embedding-3-large"
) -> List[float]:
    """
    Get text embeddings from an embedding model.
    
    Args:
        text: Text to embed
        model: Embedding model to use
        
    Returns:
        The text embedding vector
    """
    provider = _determine_provider(model)
    
    if provider == "openai":
        return await _openai_embedding(text, model)
    else:
        raise ValueError(f"Embedding not supported for provider {provider}")


async def _openai_embedding(
    text: str,
    model: str = "text-embedding-3-large"
) -> List[float]:
    """
    Get text embeddings from OpenAI models.
    
    Args:
        text: Text to embed
        model: Embedding model to use
        
    Returns:
        The text embedding vector
    """
    import openai
    
    api_key = settings.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in settings")
    
    openai.api_key = api_key
    
    try:
        response = openai.Embedding.create(
            model=model,
            input=text
        )
        
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error with OpenAI embedding API: {e}")
        raise