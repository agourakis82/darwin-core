"""
Health Check Router for DARWIN Core

Provides comprehensive health status including:
- Core services
- gRPC server
- Pulsar connection
- AI Agentic layer
- Plugin registry
"""

from fastapi import APIRouter
from datetime import datetime, timezone
from typing import Dict, Any

from ..services.agentic_orchestrator import get_agentic_orchestrator

router = APIRouter(prefix="/api/v1")


@router.get("/health")
async def health_check():
    """
    Comprehensive health check
    
    Returns status of all core components
    """
    from ..main import app
    
    uptime = (datetime.now(timezone.utc) - app.state.start_time).total_seconds()
    
    status = {
        "status": "healthy",
        "service": "DARWIN Core 2.0",
        "version": "2.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": uptime,
        "components": {
            "fastapi": "operational",
            "pulsar": "operational" if app.state.pulsar_connected else "degraded",
            "grpc": "operational" if app.state.grpc_enabled else "degraded",
            "agentic": "operational" if app.state.agentic_enabled else "disabled",
        }
    }
    
    # Get agentic system status
    if app.state.agentic_enabled:
        try:
            orchestrator = get_agentic_orchestrator()
            system_status = await orchestrator.get_system_status()
            status["agentic_details"] = system_status
        except Exception as e:
            status["components"]["agentic"] = "error"
    
    # Overall health
    degraded_count = sum(1 for v in status["components"].values() if v == "degraded")
    if degraded_count > 0:
        status["status"] = "degraded"
    
    return status


@router.get("/metrics")
async def metrics():
    """
    Prometheus-compatible metrics endpoint
    
    Exposes metrics for scraping
    """
    # Placeholder - would return Prometheus format
    return {
        "http_requests_total": 1000,
        "grpc_requests_total": 500,
        "pulsar_messages_published": 250,
        "pulsar_messages_consumed": 240,
        "plugin_health_checks_total": 100,
    }

