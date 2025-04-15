"""
Code execution tool for agents.
This module provides functionality to execute code in a sandboxed environment.
"""

import os
import time
import tempfile
import asyncio
import json
import subprocess
from typing import Dict, Any, Optional

from loguru import logger

from src.core.config import settings


async def execute_code(language: str, code: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute code in a sandboxed environment.
    
    Args:
        language: Programming language for execution
        code: Code to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Execution result including output and/or errors
    """
    use_docker = settings.get("USE_DOCKER_EXECUTION", False)
    
    if use_docker:
        return await _execute_in_docker(language, code, timeout)
    else:
        return await _execute_in_subprocess(language, code, timeout)


async def _execute_in_docker(language: str, code: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute code in a Docker container for isolation.
    
    Args:
        language: Programming language for execution
        code: Code to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Execution result
    """
    # Map language to Docker image
    language_images = {
        "python": "python:3.10-slim",
        "node": "node:16-alpine",
        "javascript": "node:16-alpine",
        "typescript": "node:16-alpine",
        "ruby": "ruby:3.0-slim",
        "go": "golang:1.18-alpine",
        "php": "php:8.1-cli-alpine",
        "bash": "bash:5.1-alpine",
        "r": "r-base:4.2.0"
    }
    
    docker_image = language_images.get(language.lower(), "python:3.10-slim")
    
    # Create temporary file for code
    with tempfile.NamedTemporaryFile(suffix=_get_file_extension(language), mode='w', delete=False) as f:
        f.write(code)
        code_path = f.name
    
    try:
        # Command map for different languages
        command_map = {
            "python": f"python {os.path.basename(code_path)}",
            "node": f"node {os.path.basename(code_path)}",
            "javascript": f"node {os.path.basename(code_path)}",
            "typescript": f"npx ts-node {os.path.basename(code_path)}",
            "ruby": f"ruby {os.path.basename(code_path)}",
            "go": f"go run {os.path.basename(code_path)}",
            "php": f"php {os.path.basename(code_path)}",
            "bash": f"bash {os.path.basename(code_path)}",
            "r": f"Rscript {os.path.basename(code_path)}"
        }
        
        exec_command = command_map.get(language.lower(), f"python {os.path.basename(code_path)}")
        
        # Start execution timer
        start_time = time.time()
        
        # Docker run command
        docker_cmd = [
            "docker", "run", "--rm",
            "--network=none",  # No network access
            "-v", f"{os.path.dirname(code_path)}:/code",
            "-w", "/code",
            "--memory=256m",  # Memory limit
            "--cpus=0.5",     # CPU limit
            docker_image,
            "sh", "-c", exec_command
        ]
        
        # Execute code in Docker
        process = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            # Kill the process if it times out
            process.kill()
            return {
                "error": f"Execution timed out after {timeout} seconds",
                "execution_time": timeout
            }
        
        execution_time = time.time() - start_time
        
        # Process results
        stdout_str = stdout.decode('utf-8', errors='replace') if stdout else ""
        stderr_str = stderr.decode('utf-8', errors='replace') if stderr else ""
        
        if process.returncode != 0:
            return {
                "error": stderr_str,
                "output": stdout_str,
                "execution_time": round(execution_time, 2),
                "return_code": process.returncode
            }
        else:
            return {
                "output": stdout_str,
                "execution_time": round(execution_time, 2),
                "return_code": process.returncode
            }
            
    except Exception as e:
        logger.error(f"Error executing code in Docker: {e}")
        return {"error": str(e)}
    finally:
        # Clean up the temporary file
        try:
            os.unlink(code_path)
        except:
            pass


def _get_file_extension(language: str) -> str:
    """
    Get the appropriate file extension for a language.
    
    Args:
        language: Programming language
        
    Returns:
        File extension including the dot
    """
    extensions = {
        "python": ".py",
        "node": ".js",
        "javascript": ".js",
        "typescript": ".ts",
        "ruby": ".rb",
        "go": ".go",
        "php": ".php",
        "bash": ".sh",
        "r": ".R"
    }
    
    return extensions.get(language.lower(), ".txt")path)
        except:
            pass


async def _execute_in_subprocess(language: str, code: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute code in a subprocess (less secure than Docker).
    
    Args:
        language: Programming language for execution
        code: Code to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Execution result
    """
    # Security warning if not using Docker
    logger.warning("Executing code without Docker isolation - this is less secure")
    
    # Create temporary file for code
    with tempfile.NamedTemporaryFile(suffix=_get_file_extension(language), mode='w', delete=False) as f:
        f.write(code)
        code_path = f.name
    
    try:
        # Command map for different languages
        command_map = {
            "python": ["python", code_path],
            "node": ["node", code_path],
            "javascript": ["node", code_path],
            "typescript": ["npx", "ts-node", code_path],
            "ruby": ["ruby", code_path],
            "go": ["go", "run", code_path],
            "php": ["php", code_path],
            "bash": ["bash", code_path],
            "r": ["Rscript", code_path]
        }
        
        exec_command = command_map.get(language.lower(), ["python", code_path])
        
        # Start execution timer
        start_time = time.time()
        
        # Execute code
        process = await asyncio.create_subprocess_exec(
            *exec_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            # Kill the process if it times out
            process.kill()
            return {
                "error": f"Execution timed out after {timeout} seconds",
                "execution_time": timeout
            }
        
        execution_time = time.time() - start_time
        
        # Process results
        stdout_str = stdout.decode('utf-8', errors='replace') if stdout else ""
        stderr_str = stderr.decode('utf-8', errors='replace') if stderr else ""
        
        if process.returncode != 0:
            return {
                "error": stderr_str,
                "output": stdout_str,
                "execution_time": round(execution_time, 2),
                "return_code": process.returncode
            }
        else:
            return {
                "output": stdout_str,
                "execution_time": round(execution_time, 2),
                "return_code": process.returncode
            }
            
    except Exception as e:
        logger.error(f"Error executing code in subprocess: {e}")
        return {"error": str(e)}
    finally:
        # Clean up the temporary file
        try:
            os.unlink(code_
