"""Utility functions for web search agent."""

import os
import requests
from typing import Optional


def get_mcp_server_session_url(
    host: str,
    project_id: str,
    mcp_slug: str,
    api_key: str
) -> Optional[str]:
    """
    Create an MCP server session and return the session URL.
    
    Args:
        host: The LangDB host URL (e.g., "https://api.langdb.ai")
        project_id: The LangDB project ID
        mcp_slug: The MCP server slug (last part of MCP server identifier)
        api_key: The LangDB API key
        
    Returns:
        The MCP server session URL if successful, None otherwise
    """
    session_endpoint = f"{host}/mcp-servers/{mcp_slug}/session"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "x-project-id": project_id
    }
    
    try:
        print(f"Creating MCP session at: {session_endpoint}")
        response = requests.post(session_endpoint, headers=headers, json={})
        response.raise_for_status()
        
        session_data = response.json()
        session_id = session_data.get("id")
        
        if session_id:
            mcp_url = f"{host}/mcp/{session_id}"
            print(f"Successfully created MCP session: {mcp_url}")
            return mcp_url
        else:
            print(f"No session ID in response: {session_data}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error creating MCP server session: {e}")
        print(f"Request URL: {session_endpoint}")
        print(f"Headers: {headers}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None


def get_dynamic_mcp_url(mcp_slug: str) -> Optional[str]:
    """
    Get the dynamic MCP server URL using environment variables.
    
    Args:
        mcp_slug: The MCP server slug (last part of MCP server identifier)
    
    Returns:
        The MCP server session URL if successful, None otherwise
    """
    host = os.getenv("LANGDB_BASE_URL")
    project_id = os.getenv("LANGDB_PROJECT_ID")
    api_key = os.getenv("LANGDB_API_KEY")
    
    if not all([host, project_id, api_key]):
        print("Missing required environment variables for MCP server session")
        return None
    
    return get_mcp_server_session_url(host, project_id, mcp_slug, api_key)