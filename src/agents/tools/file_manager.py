"""
File management tool for agents.
This module provides functionality to read, write, and manipulate files.
"""

import os
import json
import csv
import tempfile
import asyncio
import shutil
from typing import Dict, Any, List, Optional, Union, BinaryIO

import aiofiles
from loguru import logger

from src.core.config import settings


async def read_file(file_path: str, encoding: str = 'utf-8') -> str:
    """
    Read a file asynchronously.
    
    Args:
        file_path: Path to the file
        encoding: File encoding
        
    Returns:
        File contents as a string
    """
    try:
        async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
            content = await f.read()
        return content
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise


async def write_file(file_path: str, content: str, encoding: str = 'utf-8') -> bool:
    """
    Write content to a file asynchronously.
    
    Args:
        file_path: Path to the file
        content: Content to write
        encoding: File encoding
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        async with aiofiles.open(file_path, 'w', encoding=encoding) as f:
            await f.write(content)
        return True
    except Exception as e:
        logger.error(f"Error writing to file {file_path}: {e}")
        return False


async def read_json(file_path: str) -> Dict[str, Any]:
    """
    Read a JSON file asynchronously.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        JSON content as a dictionary
    """
    try:
        content = await read_file(file_path)
        return json.loads(content)
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {e}")
        raise


async def write_json(file_path: str, data: Union[Dict[str, Any], List[Any]]) -> bool:
    """
    Write data to a JSON file asynchronously.
    
    Args:
        file_path: Path to the JSON file
        data: Data to write
        
    Returns:
        True if successful, False otherwise
    """
    try:
        content = json.dumps(data, indent=2)
        return await write_file(file_path, content)
    except Exception as e:
        logger.error(f"Error writing JSON to file {file_path}: {e}")
        return False


async def read_csv(file_path: str, has_header: bool = True) -> List[Dict[str, str]]:
    """
    Read a CSV file asynchronously.
    
    Args:
        file_path: Path to the CSV file
        has_header: Whether the CSV has a header row
        
    Returns:
        CSV content as a list of dictionaries
    """
    try:
        content = await read_file(file_path)
        lines = content.splitlines()
        
        reader = csv.reader(lines)
        rows = list(reader)
        
        if not rows:
            return []
        
        if has_header:
            headers = rows[0]
            data = []
            for row in rows[1:]:
                data.append(dict(zip(headers, row)))
            return data
        else:
            # Create numeric column names
            headers = [f"col{i}" for i in range(len(rows[0]))]
            data = []
            for row in rows:
                data.append(dict(zip(headers, row)))
            return data
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
        raise


async def write_csv(file_path: str, data: List[Dict[str, Any]]) -> bool:
    """
    Write data to a CSV file asynchronously.
    
    Args:
        file_path: Path to the CSV file
        data: Data to write
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not data:
            return await write_file(file_path, "")
        
        # Extract headers from the first item
        headers = list(data[0].keys())
        
        # Create CSV content
        lines = [",".join(headers)]
        for item in data:
            line = ",".join([str(item.get(h, "")) for h in headers])
            lines.append(line)
        
        content = "\n".join(lines)
        return await write_file(file_path, content)
    except Exception as e:
        logger.error(f"Error writing CSV to file {file_path}: {e}")
        return False


async def list_files(directory: str, pattern: Optional[str] = None) -> List[str]:
    """
    List files in a directory asynchronously.
    
    Args:
        directory: Directory path
        pattern: Optional pattern to filter files
        
    Returns:
        List of file paths
    """
    try:
        files = []
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                if not pattern or (pattern and pattern in file):
                    files.append(file_path)
        return files
    except Exception as e:
        logger.error(f"Error listing files in directory {directory}: {e}")
        raise


async def create_directory(directory: str) -> bool:
    """
    Create a directory asynchronously.
    
    Args:
        directory: Directory path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {e}")
        return False


async def delete_file(file_path: str) -> bool:
    """
    Delete a file asynchronously.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        return False


async def copy_file(source: str, destination: str) -> bool:
    """
    Copy a file asynchronously.
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        shutil.copy2(source, destination)
        return True
    except Exception as e:
        logger.error(f"Error copying file from {source} to {destination}: {e}")
        return False


async def move_file(source: str, destination: str) -> bool:
    """
    Move a file asynchronously.
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        shutil.move(source, destination)
        return True
    except Exception as e:
        logger.error(f"Error moving file from {source} to {destination}: {e}")
        return False


async def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a file asynchronously.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File information
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        
        stats = os.stat(file_path)
        
        return {
            "path": file_path,
            "filename": os.path.basename(file_path),
            "directory": os.path.dirname(file_path),
            "size": stats.st_size,
            "creation_time": stats.st_ctime,
            "modification_time": stats.st_mtime,
            "access_time": stats.st_atime,
            "is_file": os.path.isfile(file_path),
            "is_directory": os.path.isdir(file_path),
            "extension": os.path.splitext(file_path)[1],
        }
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {e}")
        raise
