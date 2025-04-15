"""
FastAPI application for the MCP system.
This module provides the API entry point and route configuration.
"""

import os
import time
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from loguru import logger

from src.core.config import settings
from src.core.orchestrator import orchestrator
from src.api.dependencies import get_current_user
from src.api.routers import conversation, agents, tasks, system


# Initialize FastAPI app
app = FastAPI(
    title="Claude-Inspired MCP",
    description="Multi-Client Protocol system inspired by Claude using AutoGen",
    version=settings.get("version", "0.1.0"),
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add API routers
app.include_router(conversation.router, prefix="/api/conversations", tags=["conversations"])
app.include_router(agents.router, prefix="/api/agents", tags=["agents"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["tasks"])
app.include_router(system.router, prefix="/api/system", tags=["system"])

# Serve static files if they exist
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log HTTP requests."""
    start_time = time.time()
    
    # Generate request ID
    request_id = f"req-{int(start_time * 1000)}"
    request.state.request_id = request_id
    
    # Log request details
    logger.info(f"Request {request_id}: {request.method} {request.url.path}")
    
    # Process the request
    try:
        response = await call_next(request)
        
        # Log response details
        process_time = time.time() - start_time
        logger.info(f"Response {request_id}: {response.status_code} (took {process_time:.3f}s)")
        
        return response
    except Exception as e:
        # Log error details
        process_time = time.time() - start_time
        logger.error(f"Error {request_id}: {str(e)} (took {process_time:.3f}s)")
        
        # Return error response
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    # Configure logging
    log_level = settings.get("log_level", "INFO").upper()
    logger.configure(handlers=[{"sink": "stdout", "level": log_level}])
    
    # Log startup information
    logger.info(f"Starting Claude-Inspired MCP API (v{settings.get('version', '0.1.0')})")
    logger.info(f"Environment: {settings.get('environment', 'development')}")
    
    # Initialize orchestrator
    await orchestrator.start()
    logger.info("Orchestrator initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    # Shutdown orchestrator
    await orchestrator.stop()
    logger.info("Orchestrator stopped")
    
    # Log shutdown
    logger.info("API server shutting down")


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {
        "name": "Claude-Inspired MCP",
        "version": settings.get("version", "0.1.0"),
        "status": "active",
        "docs": "/api/docs"
    }


# Health check endpoint
@app.get("/api/health", tags=["system"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.get("version", "0.1.0"),
        "environment": settings.get("environment", "development")
    }
