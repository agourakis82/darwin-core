"""
Darwin Core 2025 - MCP Router (FastAPI)
Expõe Model Context Protocol via HTTP REST
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .mcp_tools import get_darwin_mcp_tools

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp", tags=["MCP"])


# ==================== Schemas ====================

class MCPToolCallRequest(BaseModel):
    """Request para chamar um tool MCP"""
    tool: str = Field(..., description="Nome do tool a executar")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Argumentos do tool")

class MCPToolCallResponse(BaseModel):
    """Response de execução de tool"""
    success: bool
    tool: str
    result: Any = None
    error: str = None

class MCPToolListResponse(BaseModel):
    """Lista de tools disponíveis"""
    tools: List[Dict[str, Any]]


# ==================== Endpoints ====================

@router.get("/tools", response_model=MCPToolListResponse)
async def list_mcp_tools():
    """
    Lista todos os MCP tools disponíveis
    
    Retorna schemas completos com descrição e inputSchema
    """
    tools = get_darwin_mcp_tools()
    return MCPToolListResponse(tools=tools.get_tools_list())


@router.post("/call", response_model=MCPToolCallResponse)
async def call_mcp_tool(request: MCPToolCallRequest):
    """
    Executa um MCP tool
    
    Exemplo:
    ```json
    {
      "tool": "darwin_rag_query",
      "arguments": {
        "query": "What are biomaterials?",
        "top_k": 5
      }
    }
    ```
    """
    tools = get_darwin_mcp_tools()
    
    logger.info(f"MCP call: tool={request.tool}, args={request.arguments}")
    
    result = await tools.call_tool(request.tool, request.arguments)
    
    return MCPToolCallResponse(
        success=result["success"],
        tool=request.tool,
        result=result.get("result"),
        error=result.get("error")
    )


@router.get("/health")
async def mcp_health():
    """Health check do MCP server"""
    return {
        "status": "healthy",
        "service": "darwin-mcp",
        "tools_available": len(get_darwin_mcp_tools().get_tools_list())
    }

