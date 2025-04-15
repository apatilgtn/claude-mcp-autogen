"""
Web search tool for agents.
This module provides functionality to search the web for information.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional

import httpx
from loguru import logger

from src.core.config import settings


async def search_web(query: str, 
                    max_results: int = 5, 
                    search_type: str = "web") -> List[Dict[str, Any]]:
    """
    Search the web for information based on a query.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        search_type: Type of search (web, news, image)
        
    Returns:
        List of search results
    """
    # Determine which search API to use based on configuration
    search_api = settings.get("SEARCH_API", "serper")
    
    if search_api == "serper":
        return await _search_with_serper(query, max_results, search_type)
    elif search_api == "serpapi":
        return await _search_with_serpapi(query, max_results, search_type)
    elif search_api == "mock":
        return await _mock_search(query, max_results, search_type)
    else:
        logger.warning(f"Unknown search API: {search_api}, using mock search instead")
        return await _mock_search(query, max_results, search_type)


async def _search_with_serper(query: str, 
                             max_results: int = 5, 
                             search_type: str = "web") -> List[Dict[str, Any]]:
    """
    Search using the Serper API.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        search_type: Type of search (web, news, image)
        
    Returns:
        List of search results
    """
    api_key = settings.get("SERPER_API_KEY")
    if not api_key:
        logger.error("Serper API key not found, falling back to mock search")
        return await _mock_search(query, max_results, search_type)
    
    try:
        # Prepare request
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "q": query,
            "num": max_results
        }
        
        # Execute search
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            search_data = response.json()
        
        # Process results
        results = []
        
        # Process organic results
        organic_results = search_data.get("organic", [])
        for result in organic_results[:max_results]:
            results.append({
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "source": "serper",
                "type": "web"
            })
        
        return results
    except Exception as e:
        logger.error(f"Error searching with Serper: {e}")
        return await _mock_search(query, max_results, search_type)


async def _search_with_serpapi(query: str, 
                              max_results: int = 5, 
                              search_type: str = "web") -> List[Dict[str, Any]]:
    """
    Search using the SerpAPI.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        search_type: Type of search (web, news, image)
        
    Returns:
        List of search results
    """
    api_key = settings.get("SERPAPI_API_KEY")
    if not api_key:
        logger.error("SerpAPI key not found, falling back to mock search")
        return await _mock_search(query, max_results, search_type)
    
    try:
        # Prepare request
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": api_key,
            "num": max_results
        }
        
        # Execute search
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            search_data = response.json()
        
        # Process results
        results = []
        
        # Process organic results
        organic_results = search_data.get("organic_results", [])
        for result in organic_results[:max_results]:
            results.append({
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "source": "serpapi",
                "type": "web"
            })
        
        return results
    except Exception as e:
        logger.error(f"Error searching with SerpAPI: {e}")
        return await _mock_search(query, max_results, search_type)


async def _mock_search(query: str, 
                      max_results: int = 5, 
                      search_type: str = "web") -> List[Dict[str, Any]]:
    """
    Mock search function for testing or when API keys are unavailable.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        search_type: Type of search (web, news, image)
        
    Returns:
        List of mock search results
    """
    # Create mock results based on the query
    results = []
    
    # Simulate some latency for realism
    await asyncio.sleep(0.5)
    
    # Generate mock results
    for i in range(min(max_results, 3)):
        results.append({
            "title": f"Mock Result {i+1} for {query}",
            "url": f"https://example.com/result/{i+1}",
            "snippet": f"This is a mock search result for the query '{query}'. It contains some information that might be relevant to your search. This is result number {i+1}.",
            "source": "mock",
            "type": search_type
        })
    
    return results
