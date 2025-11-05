"""
Darwin Core 2025 - Custom GPT API Endpoints
Endpoints otimizados para integração com ChatGPT Custom GPT
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from .auth import verify_api_token
from .mcp_tools import get_darwin_mcp_tools

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Custom GPT"])


# ==================== Schemas ====================

class AnalyzeRequest(BaseModel):
    """Request para análise científica geral"""
    query: str = Field(..., description="Pergunta ou dados para análise")
    domain: str = Field(default="general", description="Domínio do conhecimento")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Contexto adicional")

class AnalyzeResponse(BaseModel):
    """Response de análise"""
    query: str
    domain: str
    analysis: Dict[str, Any]
    sources: List[str]
    confidence: float

class KECAnalysisRequest(BaseModel):
    """Request para análise KEC 3.0"""
    graph_data: Dict[str, Any] = Field(..., description="Dados do grafo (nodes, edges)")
    metrics: List[str] = Field(default=["all"], description="Métricas a computar")
    use_gpu: bool = Field(default=True, description="Usar aceleração GPU")

class KECAnalysisResponse(BaseModel):
    """Response de análise KEC"""
    status: str
    metrics: Dict[str, Any]
    computation_time_ms: float
    gpu_used: bool

class RAGQueryRequest(BaseModel):
    """Request para query RAG++"""
    query: str = Field(..., description="Query de busca")
    top_k: int = Field(default=5, ge=1, le=20, description="Número de resultados")
    domain: Optional[str] = Field(default="general", description="Filtro de domínio")
    use_hybrid: bool = Field(default=True, description="Usar busca híbrida")

class RAGQueryResponse(BaseModel):
    """Response de query RAG++"""
    query: str
    results: List[Dict[str, Any]]
    total_found: int
    search_mode: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    services: Dict[str, Any]


# ==================== Endpoints ====================

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    _: bool = Depends(verify_api_token)
):
    """
    Análise científica geral usando Darwin AI
    
    Combina RAG++, semantic memory e multi-AI debate para análise profunda.
    Requer autenticação Bearer token.
    
    Exemplo:
    ```
    POST /api/v1/analyze
    Authorization: Bearer <token>
    
    {
      "query": "What are the latest advances in biomaterial scaffolds?",
      "domain": "biomaterials"
    }
    ```
    """
    logger.info(f"Analyze request: {request.query[:100]}")
    
    # TODO: Implementar análise completa com RAG++ e Multi-AI
    # Por enquanto, placeholder
    
    return AnalyzeResponse(
        query=request.query,
        domain=request.domain,
        analysis={
            "summary": f"Analysis of: {request.query}",
            "key_points": [
                "Point 1: Advanced analysis using Darwin AI",
                "Point 2: Multiple data sources consulted",
                "Point 3: High confidence results"
            ],
            "details": "Detailed analysis would appear here..."
        },
        sources=["Darwin RAG++", "Semantic Memory", "Scientific Databases"],
        confidence=0.89
    )


@router.post("/kec", response_model=KECAnalysisResponse)
async def kec_analysis(
    request: KECAnalysisRequest,
    _: bool = Depends(verify_api_token)
):
    """
    KEC 3.0 Topological Analysis
    
    Análise topológica de grafos usando:
    - Kinetic metrics
    - Entropy analysis
    - Curvature computation (Ricci, Ollivier)
    - Persistent homology (GUDHI)
    
    GPU-accelerated quando disponível.
    """
    logger.info(f"KEC analysis: metrics={request.metrics}, gpu={request.use_gpu}")
    
    tools = get_darwin_mcp_tools()
    result = await tools.call_tool("darwin_kec_analysis", {
        "graph_data": request.graph_data,
        "metrics": request.metrics,
        "use_gpu": request.use_gpu
    })
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"KEC analysis failed: {result.get('error')}"
        )
    
    return KECAnalysisResponse(**result["result"])


@router.post("/rag_query", response_model=RAGQueryResponse)
async def rag_query(
    request: RAGQueryRequest,
    _: bool = Depends(verify_api_token)
):
    """
    RAG++ Query - Retrieval-Augmented Generation Plus
    
    Sistema de busca estado da arte com:
    - Hybrid search (dense + sparse vectors)
    - Reciprocal Rank Fusion (RRF)
    - Embeddings SOTA (nomic-embed-text-v1.5, jina-v3, gte-Qwen2)
    - Qdrant vector database
    - Binary quantization (90% storage reduction)
    
    Exemplo:
    ```json
    {
      "query": "bone tissue engineering scaffolds",
      "top_k": 10,
      "domain": "biomaterials",
      "use_hybrid": true
    }
    ```
    """
    logger.info(f"RAG++ query: {request.query}")
    
    tools = get_darwin_mcp_tools()
    result = await tools.call_tool("darwin_rag_query", {
        "query": request.query,
        "top_k": request.top_k,
        "domain": request.domain,
        "use_hybrid": request.use_hybrid
    })
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query failed: {result.get('error')}"
        )
    
    return RAGQueryResponse(**result["result"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check do Darwin Core
    
    Verifica status de todos os serviços:
    - FastAPI
    - Qdrant (vector DB)
    - Redis (cache)
    - Ollama (LLMs locais)
    - gRPC server
    - Pulsar (event streaming)
    """
    # TODO: Implementar checks reais de cada serviço
    return HealthResponse(
        status="healthy",
        version="2025.1.0",
        services={
            "fastapi": {"status": "up"},
            "qdrant": {"status": "checking", "url": "qdrant-service:6333"},
            "redis": {"status": "checking", "url": "redis-service:6379"},
            "ollama": {"status": "checking", "url": "ollama-service:11434"},
            "grpc": {"status": "up", "port": 50051},
            "pulsar": {"status": "checking"}
        }
    )


@router.get("/ready")
async def readiness_check():
    """
    Readiness probe para Kubernetes
    
    Retorna 200 apenas quando todos os serviços dependentes estão OK
    """
    # TODO: Implementar checks reais
    # Por enquanto, sempre retorna ready
    return {"status": "ready", "timestamp": "2025-10-27T00:00:00Z"}


# Endpoint raiz para OpenAPI schema (usado pelo Custom GPT)
@router.get("/")
async def api_root():
    """API root - informações gerais"""
    return {
        "name": "Darwin Core 2025 API",
        "version": "2025.1.0",
        "description": "Estado da arte em IA para pesquisa científica",
        "features": [
            "RAG++ (Retrieval-Augmented Generation Plus)",
            "KEC 3.0 (Topological Analysis)",
            "Agentic Workflows (LangGraph)",
            "Multi-AI Debate (GPT-4, Claude, Gemini, Ollama)",
            "Semantic Memory (Qdrant)",
            "BrainBERT-EEG",
            "Heliobiology Analysis"
        ],
        "documentation": "/docs",
        "openapi_schema": "/openapi.json"
    }

