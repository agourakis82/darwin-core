"""
DARWIN Core 2.0 - Modular Architecture

Core components:
- Semantic Memory + RAG++
- Continuous Learning (ML/RL)
- Multi-AI Debate (GPT-4, Claude, o3, Ollama, Gemini)
- HuggingFace Integration (local models via vLLM/Ollama)
- MCP Server
- ChatGPT + Claude connectors
- AI Agentic Layer (self-healing)

HTTP/2 (FastAPI REST) for external
gRPC (HTTP/2) for internal plugin communication
Apache Pulsar for event-driven architecture
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

# OpenTelemetry instrumentation
try:
    from opentelemetry import trace
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.grpc import GrpcAioInstrumentorServer
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    import os
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

# Core services
from .services.pulsar_client import get_pulsar_client
from .grpc_services.core_server import start_grpc_server, stop_grpc_server
from .services.agentic_orchestrator import get_agentic_orchestrator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("darwin.core")


# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown"""
    logger.info("=" * 60)
    logger.info("🚀 DARWIN CORE 2.0 - Starting")
    logger.info("=" * 60)
    
    # Store start time
    app.state.start_time = datetime.now(timezone.utc)
    
    # === STARTUP ===
    
    # 1. Connect to Apache Pulsar
    try:
        pulsar = get_pulsar_client()
        await pulsar.connect()
        app.state.pulsar_connected = True
    except Exception as e:
        logger.warning(f"⚠️ Pulsar unavailable: {e}")
        app.state.pulsar_connected = False
    
    # 2. Start gRPC server
    try:
        grpc_server = await start_grpc_server(port=50051)
        app.state.grpc_server = grpc_server
        app.state.grpc_enabled = True
    except Exception as e:
        logger.warning(f"⚠️ gRPC server failed: {e}")
        app.state.grpc_enabled = False
    
    # 3. Initialize AI Agentic Orchestrator
    try:
        orchestrator = get_agentic_orchestrator()
        await orchestrator.initialize()
        app.state.agentic_enabled = True
    except Exception as e:
        logger.warning(f"⚠️ Agentic orchestrator failed: {e}")
        app.state.agentic_enabled = False
    
    # 4. Load routers
    try:
        from .routers import memory_rest, health, models, corpus, oauth
        from .api import mcp_router  # Router MCP correto com tools
        from .api import memory_router  # New K8s memory router
        
        # Initialize memory system
        memory_router.init_memory_system()
        
        app.include_router(memory_rest.router, tags=["Memory"])
        app.include_router(memory_router.router, tags=["Memory K8s"])
        app.include_router(mcp_router.router, tags=["MCP"])  # MCP com /mcp/tools, /mcp/health, /mcp/call
        app.include_router(health.router, tags=["Health"])
        app.include_router(models.router, tags=["Models"])
        app.include_router(corpus.router, tags=["Corpus"])
        app.include_router(oauth.router, tags=["OAuth"])
        
        logger.info("✅ Core routers loaded (6 routers) with MCP Tools")
        app.state.routers_loaded = True
        
    except Exception as e:
        logger.error(f"❌ Failed to load routers: {e}")
        app.state.routers_loaded = False
    
    # 5. Start Auto-Training Pipeline
    try:
        from .services.auto_training_pipeline import get_auto_training_pipeline
        
        pipeline = get_auto_training_pipeline()
        await pipeline.start()
        app.state.auto_training_enabled = True
        logger.info("✅ Auto-training pipeline started")
        
    except Exception as e:
        logger.warning(f"⚠️ Auto-training failed: {e}")
        app.state.auto_training_enabled = False
    
    logger.info("=" * 60)
    logger.info("✅ DARWIN CORE 2.0 - Ready")
    logger.info(f"   Pulsar: {app.state.pulsar_connected}")
    logger.info(f"   gRPC: {app.state.grpc_enabled}")
    logger.info(f"   Agentic: {app.state.agentic_enabled}")
    logger.info(f"   Routers: {app.state.routers_loaded}")
    logger.info("=" * 60)
    
    yield
    
    # === SHUTDOWN ===
    
    logger.info("🛑 DARWIN CORE 2.0 - Shutting down")
    
    # Stop auto-training pipeline
    if app.state.auto_training_enabled:
        from .services.auto_training_pipeline import get_auto_training_pipeline
        pipeline = get_auto_training_pipeline()
        await pipeline.stop()
    
    # Stop agentic orchestrator
    if app.state.agentic_enabled:
        orchestrator = get_agentic_orchestrator()
        await orchestrator.shutdown()
    
    # Stop gRPC server
    if app.state.grpc_enabled:
        await stop_grpc_server()
    
    # Disconnect Pulsar
    if app.state.pulsar_connected:
        pulsar = get_pulsar_client()
        await pulsar.disconnect()
    
    logger.info("✅ DARWIN CORE 2.0 - Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="DARWIN Core 2.0",
    description="Modular AI Research Brain - Core Services",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register MCP and Custom GPT API routers (Darwin 2025 K8s Redeploy)
try:
    from .api.mcp_router import router as mcp_router_v2
    from .api.gpt_endpoints import router as gpt_router
    
    app.include_router(mcp_router_v2)
    app.include_router(gpt_router)
    logger.info("✅ MCP and Custom GPT routers loaded (Darwin 2025)")
except Exception as e:
    logger.warning(f"⚠️ Failed to load MCP/GPT routers: {e}")

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": exc.errors(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "DARWIN Core 2.0",
        "status": "operational",
        "version": "2.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# Initialize OpenTelemetry if available
if OTEL_AVAILABLE:
    # Configure resource
    resource = Resource.create({
        "service.name": "darwin-core",
        "service.version": "2.0.0",
        "deployment.environment": "production"
    })
    
    # Setup tracer provider
    provider = TracerProvider(resource=resource)
    
    # OTLP exporter (to Jaeger/Tempo)
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317")
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    
    # Set global tracer
    trace.set_tracer_provider(provider)
    
    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)
    
    # Instrument gRPC (will be applied to server)
    GrpcAioInstrumentorServer().instrument()
    
    logger.info("✅ OpenTelemetry instrumentation enabled")
    logger.info(f"   OTLP endpoint: {otlp_endpoint}")

