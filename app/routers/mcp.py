"""
MCP (Model Context Protocol) Router

Compatible with Claude Desktop and Custom GPT
Provides tools for memory save/search
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

router = APIRouter()


class MCPToolRequest(BaseModel):
    """MCP tool execution request"""
    tool_name: str
    parameters: Dict[str, Any]


class MCPToolResponse(BaseModel):
    """MCP tool execution response"""
    content: List[Dict[str, str]]
    isError: Optional[bool] = False


@router.post("/execute/{tool_name}")
async def execute_mcp_tool(tool_name: str, parameters: Dict[str, Any]):
    """
    Execute MCP tool
    
    Supported tools:
    - darwinSaveMemory
    - darwinSearchMemory
    - darwinQueryAI
    """
    from .memory_rest import save_memory, search_memory, MemorySaveRequest, MemorySearchRequest
    
    try:
        if tool_name == "darwinSaveMemory":
            request = MemorySaveRequest(**parameters)
            result = await save_memory(request)
            
            return MCPToolResponse(
                content=[{
                    "type": "text",
                    "text": f"✅ Memory saved: {result.conversation_id}"
                }]
            )
        
        elif tool_name == "darwinSearchMemory":
            request = MemorySearchRequest(**parameters)
            result = await search_memory(request)
            
            return MCPToolResponse(
                content=[{
                    "type": "text",
                    "text": f"Found {result.total_results} results for: {result.query}"
                }]
            )
        
        elif tool_name == "darwinQueryAI":
            # TODO: Implement multi-AI query
            return MCPToolResponse(
                content=[{
                    "type": "text",
                    "text": "AI query not yet implemented in Core 2.0"
                }]
            )
        
        else:
            return MCPToolResponse(
                content=[{
                    "type": "text",
                    "text": f"Unknown tool: {tool_name}"
                }],
                isError=True
            )
            
    except Exception as e:
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": f"Error: {str(e)}"
            }],
            isError=True
        )


# Legacy aliases for backward compatibility
@router.post("/darwinSaveMemory")
async def darwin_save_memory_legacy(
    title: Optional[str] = None,
    content: str = "",
    domain: str = "general",
    platform: str = "other",
    tags: List[str] = []
):
    """Legacy endpoint for Custom GPT"""
    return await execute_mcp_tool("darwinSaveMemory", {
        "title": title,
        "content": content,
        "domain": domain,
        "platform": platform,
        "tags": tags
    })


@router.post("/darwinSearchMemory")
async def darwin_search_memory_legacy(
    query: str = "",
    top_k: int = 10,
    domain: Optional[str] = None
):
    """Legacy endpoint for Custom GPT"""
    return await execute_mcp_tool("darwinSearchMemory", {
        "query": query,
        "top_k": top_k,
        "domain": domain
    })

