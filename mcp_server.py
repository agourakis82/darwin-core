#!/usr/bin/env python3
"""
DARWIN MCP Server for Claude Desktop
Provides semantic memory tools via Model Context Protocol
"""

import os
import sys
import json
import asyncio
import requests
from typing import Any, Dict, List

# MCP SDK
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("ERROR: MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Configuration
DARWIN_API_URL = os.getenv("DARWIN_API_URL", "http://localhost:8090")
DARWIN_API_TOKEN = os.getenv("DARWIN_API_TOKEN", "darwin_local_dev_token")

# Initialize MCP Server
app = Server("darwin")

@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available DARWIN tools"""
    return [
        Tool(
            name="darwin_save_memory",
            description="Save information to DARWIN semantic memory. Use this to remember important facts, research notes, or any information you want to recall later.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to save (research notes, facts, observations, etc.)"
                    },
                    "title": {
                        "type": "string",
                        "description": "A descriptive title for this memory"
                    },
                    "domain": {
                        "type": "string",
                        "enum": ["general", "biomaterials", "chemistry", "medicine", "pharmacology", 
                                "mathematics", "physics", "quantum", "philosophy", "infrastructure"],
                        "description": "Knowledge domain",
                        "default": "general"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization (e.g., ['research', 'important'])"
                    }
                },
                "required": ["content", "title"]
            }
        ),
        Tool(
            name="darwin_search_memory",
            description="Search DARWIN semantic memory. Use this to recall previously saved information, find related research, or retrieve facts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (natural language)"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "domain": {
                        "type": "string",
                        "enum": ["general", "biomaterials", "chemistry", "medicine", "pharmacology", 
                                "mathematics", "physics", "quantum", "philosophy", "infrastructure"],
                        "description": "Filter by domain (optional)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="darwin_health",
            description="Check DARWIN system health and status",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""
    
    if name == "darwin_save_memory":
        return await save_memory(arguments)
    elif name == "darwin_search_memory":
        return await search_memory(arguments)
    elif name == "darwin_health":
        return await check_health()
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def save_memory(args: Dict[str, Any]) -> List[TextContent]:
    """Save to semantic memory"""
    try:
        payload = {
            "content": args["content"],
            "title": args["title"],
            "domain": args.get("domain", "general"),
            "platform": "claude_desktop",
            "tags": args.get("tags", [])
        }
        
        response = requests.post(
            f"{DARWIN_API_URL}/api/v1/memory/save",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            conv_id = data.get("conversation_id", "unknown")
            return [TextContent(
                type="text",
                text=f"✅ Memory saved successfully!\n\nID: {conv_id}\nTitle: {args['title']}\nDomain: {args.get('domain', 'general')}\n\nYou can search for this later using darwin_search_memory."
            )]
        else:
            return [TextContent(
                type="text",
                text=f"❌ Failed to save memory: {response.status_code}\n{response.text}"
            )]
            
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Error saving memory: {str(e)}")]

async def search_memory(args: Dict[str, Any]) -> List[TextContent]:
    """Search semantic memory"""
    try:
        payload = {
            "query": args["query"],
            "top_k": args.get("top_k", 5)
        }
        
        if "domain" in args:
            payload["domain"] = args["domain"]
        
        response = requests.post(
            f"{DARWIN_API_URL}/api/v1/memory/search",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            total = data.get("total_results", 0)
            results = data.get("results", [])
            
            if total == 0:
                return [TextContent(
                    type="text",
                    text=f"🔍 No results found for: '{args['query']}'\n\nTry different keywords or save new information first."
                )]
            
            # Format results
            output = f"🔍 Found {total} result(s) for: '{args['query']}'\n\n"
            
            for i, r in enumerate(results[:5], 1):
                score = r.get("score", 0)
                title = r.get("title", "Untitled")
                content = r.get("content", "")[:300]  # First 300 chars
                domain = r.get("domain", "general")
                tags = ", ".join(r.get("tags", []))
                
                output += f"[{i}] {title}\n"
                output += f"    Domain: {domain} | Score: {score:.3f}\n"
                if tags:
                    output += f"    Tags: {tags}\n"
                output += f"    {content}...\n\n"
            
            return [TextContent(type="text", text=output)]
            
        else:
            return [TextContent(
                type="text",
                text=f"❌ Search failed: {response.status_code}\n{response.text}"
            )]
            
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Error searching: {str(e)}")]

async def check_health() -> List[TextContent]:
    """Check system health"""
    try:
        response = requests.get(f"{DARWIN_API_URL}/api/v1/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "unknown")
            version = data.get("version", "unknown")
            uptime = data.get("uptime_seconds", 0)
            
            output = f"🟢 DARWIN System Health\n\n"
            output += f"Status: {status}\n"
            output += f"Version: {version}\n"
            output += f"Uptime: {uptime:.0f}s\n\n"
            
            components = data.get("components", {})
            output += "Components:\n"
            for comp, state in components.items():
                emoji = "✅" if state == "operational" else "❌"
                output += f"  {emoji} {comp}: {state}\n"
            
            return [TextContent(type="text", text=output)]
        else:
            return [TextContent(
                type="text",
                text=f"❌ Health check failed: {response.status_code}"
            )]
            
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"❌ Cannot connect to DARWIN API at {DARWIN_API_URL}\n\nError: {str(e)}\n\nMake sure:\n1. kubectl port-forward -n darwin svc/darwin-core 8090:8090 is running\n2. DARWIN Core pod is healthy"
        )]

async def main():
    """Run MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())

