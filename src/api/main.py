"""
FastAPI application for the MCP system.
This module provides the API entry point and route configuration.
"""

import os
import time
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from loguru import logger

from src.core.config import settings
from src.core.orchestrator import orchestrator
from src.api.dependencies import get_current_user
from src.api.routers import conversation, agents, tasks, system, auth

# -------------------------------------------------
# Initialize FastAPI app
# -------------------------------------------------
app = FastAPI(
    title="Claude-Inspired MCP",
    description="Multi-Agent Cognitive Platform inspired by Claude using AutoGen",
    version=settings.get("version", "0.1.0"),
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# -------------------------------------------------
# CORS Configuration
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get("MCP_CORS_ORIGINS", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Serve Static Files (if needed for frontend)
# -------------------------------------------------
static_dir = os.path.join(os.path.dirname(__file__), "../../../public")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# -------------------------------------------------
# API Route Includes
# -------------------------------------------------
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(conversation.router, prefix="/api/conversations", tags=["conversations"])
app.include_router(agents.router, prefix="/api/agents", tags=["agents"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["tasks"])
app.include_router(system.router, prefix="/api/system", tags=["system"])

# -------------------------------------------------
# Root Health Check
# -------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "MCP API is running",
        "version": settings.get("version", "0.1.0"),
        "status": "OK"
    }

# -------------------------------------------------
# Custom Exception Handling
# -------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"}
    )
