"""Multi-AI Hub Router - Sistema revolucionário de orchestração de múltiplas IAs."""

import os
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends, Query, Path, BackgroundTasks
from fastapi.responses import JSONResponse

from ..core.logging import get_logger
from ..models.multi_ai_models import (
    AIProvider, AIModel, ScientificDomain,
    ChatRequest, ChatResponse, ConversationCreate, ConversationHistory,
    ContextSyncRequest, ModelRecommendation, MultiAIHealthCheck,
    UsageAnalytics, PerformanceMetrics, ExtractedInsight,
    BiomaterialsChatRequest, MathematicalChatRequest, PhilosophyChatRequest,
    ResearchChatRequest, CrossDomainChatRequest, RoutingRule, RoutingDecision
)

# Import Multi-AI components
from .chat_orchestrator import ChatOrchestrator
from .context_bridge import ContextBridge
from .conversation_manager import ConversationManager

# Import AI clients with error handling
try:
    from .ai_clients import ChatGPTClient, ClaudeClient, GeminiClient
    AI_CLIENTS_AVAILABLE = True
except ImportError as e:
    ChatGPTClient = ClaudeClient = GeminiClient = None
    AI_CLIENTS_AVAILABLE = False

logger = get_logger("multi_ai.router")

# Create router
router = APIRouter(prefix="/api/v1/multi-ai", tags=["Multi-AI Hub"])


class MultiAIHub:
    """Hub revolucionário de orchestração multi-AI."""
    
    def __init__(self):
        self.enabled = False
        self.orchestrator = ChatOrchestrator()
        self.context_bridge = ContextBridge()
        self.conversation_manager = ConversationManager(context_bridge=self.context_bridge)
        
        # AI Clients
        self.ai_clients: Dict[AIProvider, Any] = {}
        self._client_health: Dict[AIProvider, bool] = {}
        
        # Statistics
        self._global_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "avg_response_time": 0.0,
            "uptime_start": datetime.now(timezone.utc)
        }
        
    async def initialize(self):
        """Inicializar Multi-AI Hub completo."""
        logger.info("🚀 Initializing Multi-AI Hub - Revolutionary AI Orchestration System")
        
        try:
            # Initialize core components
            await self.orchestrator.initialize()
            await self.context_bridge.initialize()
            await self.conversation_manager.initialize()
            
            # Initialize AI clients if available
            if AI_CLIENTS_AVAILABLE:
                await self._initialize_ai_clients()
            else:
                logger.warning("AI client libraries not available - running in simulation mode")
            
            self.enabled = True
            logger.info("✅ Multi-AI Hub initialized successfully!")
            logger.info("🎯 Ready for intelligent AI routing and orchestration")
            
        except Exception as e:
            logger.error(f"Failed to initialize Multi-AI Hub: {e}")
            self.enabled = False
            raise
    
    async def shutdown(self):
        """Shutdown Multi-AI Hub."""
        logger.info("🛑 Shutting down Multi-AI Hub...")
        
        # Shutdown components
        await self.orchestrator.shutdown()
        await self.context_bridge.shutdown()
        await self.conversation_manager.shutdown()
        
        # Shutdown AI clients
        for client in self.ai_clients.values():
            if hasattr(client, 'shutdown'):
                await client.shutdown()
        
        self.enabled = False
        logger.info("✅ Multi-AI Hub shutdown complete")
    
    async def _initialize_ai_clients(self):
        """Inicializar clientes de IA."""
        
        # Get API keys from environment
        api_keys = {
            AIProvider.CHATGPT: os.getenv("OPENAI_API_KEY"),
            AIProvider.CLAUDE: os.getenv("ANTHROPIC_API_KEY"), 
            AIProvider.GEMINI: os.getenv("GOOGLE_AI_API_KEY")
        }
        
        # Initialize available clients
        for ai_provider, api_key in api_keys.items():
            if not api_key:
                logger.warning(f"No API key found for {ai_provider} - skipping initialization")
                continue
                
            try:
                if ai_provider == AIProvider.CHATGPT and ChatGPTClient:
                    client = ChatGPTClient(api_key)
                    await client.initialize()
                    self.ai_clients[ai_provider] = client
                    self._client_health[ai_provider] = True
                    
                elif ai_provider == AIProvider.CLAUDE and ClaudeClient:
                    client = ClaudeClient(api_key)
                    await client.initialize()
                    self.ai_clients[ai_provider] = client
                    self._client_health[ai_provider] = True
                    
                elif ai_provider == AIProvider.GEMINI and GeminiClient:
                    client = GeminiClient(api_key)
                    await client.initialize()
                    self.ai_clients[ai_provider] = client
                    self._client_health[ai_provider] = True
                    
                logger.info(f"✅ {ai_provider} client initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize {ai_provider} client: {e}")
                self._client_health[ai_provider] = False
    
    async def chat_with_routing(self, request: ChatRequest) -> ChatResponse:
        """Chat com roteamento inteligente."""
        
        if not self.enabled:
            raise HTTPException(status_code=503, detail="Multi-AI Hub not initialized")
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Route request to best AI
            routing_decision = await self.orchestrator.route_request(request)
            
            # Get AI client
            selected_ai = routing_decision.selected_ai
            if selected_ai not in self.ai_clients:
                # Fallback to available AI
                available_ais = [ai for ai, healthy in self._client_health.items() if healthy]
                if not available_ais:
                    raise HTTPException(status_code=503, detail="No AI providers available")
                selected_ai = available_ais[0]
                logger.warning(f"Fallback to {selected_ai} - original selection unavailable")
            
            # Prepare messages
            messages = [{"role": "user", "content": request.message}]
            
            # Get relevant context if conversation exists
            if request.conversation_id:
                relevant_context = await self.context_bridge.get_relevant_context(
                    request.conversation_id,
                    request.domain,
                    selected_ai,
                    keywords=None,
                    max_contexts=5
                )
                
                # Add context to messages
                if relevant_context:
                    context_content = "\n".join([
                        f"Context: {ctx.content[:200]}..."
                        for ctx in relevant_context
                    ])
                    messages.insert(0, {"role": "system", "content": f"Relevant context:\n{context_content}"})
            
            # Make AI call
            client = self.ai_clients[selected_ai]
            ai_response = await client.chat(
                messages=messages,
                model=routing_decision.selected_model,
                domain=request.domain,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # Create response
            response = ChatResponse(
                message=ai_response["content"],
                ai_provider=selected_ai,
                model=routing_decision.selected_model,
                conversation_id=request.conversation_id or f"new_{int(datetime.now().timestamp())}",
                domain=request.domain,
                usage=ai_response.get("usage"),
                routing_reason=routing_decision.reasoning,
                confidence_score=routing_decision.confidence,
                metadata={
                    "processing_time_ms": routing_decision.processing_time_ms,
                    "fallback_used": routing_decision.fallback_used,
                    "context_used": len(relevant_context) if request.conversation_id else 0
                }
            )
            
            # Update conversation if ID provided
            if request.conversation_id:
                from ..models.multi_ai_models import ChatMessage
                
                # Add user message
                user_msg = ChatMessage(role="user", content=request.message)
                await self.conversation_manager.add_message(
                    request.conversation_id, user_msg, None,
                    tokens_used=ai_response.get("usage", {}).get("prompt_tokens", 0)
                )
                
                # Add AI response
                ai_msg = ChatMessage(role="assistant", content=response.message)
                await self.conversation_manager.add_message(
                    request.conversation_id, ai_msg, selected_ai,
                    tokens_used=ai_response.get("usage", {}).get("completion_tokens", 0),
                    cost=ai_response.get("estimated_cost", 0.0)
                )
            
            # Update performance tracking
            await self.orchestrator.update_performance(
                selected_ai, routing_decision.selected_model, request.domain,
                latency=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                success=True,
                quality_score=8.0  # Could be enhanced with actual quality assessment
            )
            
            # Update global statistics
            self._global_stats["total_requests"] += 1
            self._global_stats["successful_requests"] += 1
            if ai_response.get("usage"):
                self._global_stats["total_tokens"] += ai_response["usage"].get("total_tokens", 0)
            
            return response
            
        except Exception as e:
            logger.error(f"Chat routing error: {e}")
            self._global_stats["total_requests"] += 1
            self._global_stats["failed_requests"] += 1
            raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")
    
    async def direct_chat(self, ai_model: AIModel, request: ChatRequest) -> ChatResponse:
        """Chat direto com modelo específico."""
        
        # Determine AI provider from model
        model_to_provider = {
            AIModel.GPT_4_TURBO: AIProvider.CHATGPT,
            AIModel.GPT_4: AIProvider.CHATGPT,
            AIModel.GPT_3_5_TURBO: AIProvider.CHATGPT,
            AIModel.CLAUDE_3_5_SONNET: AIProvider.CLAUDE,
            AIModel.CLAUDE_3_SONNET: AIProvider.CLAUDE,
            AIModel.CLAUDE_3_HAIKU: AIProvider.CLAUDE,
            AIModel.GEMINI_PRO: AIProvider.GEMINI,
            AIModel.GEMINI_PRO_VISION: AIProvider.GEMINI
        }
        
        ai_provider = model_to_provider.get(ai_model)
        if not ai_provider or ai_provider not in self.ai_clients:
            raise HTTPException(status_code=404, detail=f"AI model {ai_model} not available")
        
        # Override routing to use specific AI/model
        request.preferred_ai = ai_provider
        request.model = ai_model
        
        return await self.chat_with_routing(request)


# Global hub instance
hub = MultiAIHub()


# Dependency for checking hub status
async def get_hub():
    """Dependency para obter hub inicializado."""
    if not hub.enabled:
        raise HTTPException(status_code=503, detail="Multi-AI Hub not initialized")
    return hub


# ==================== CORE CHAT ENDPOINTS ====================

@router.post("/chat", response_model=ChatResponse)
async def unified_chat(request: ChatRequest, hub_instance=Depends(get_hub)):
    """
    🎯 **Chat Unificado com Roteamento Inteligente**
    
    Sistema revolucionário que roteia automaticamente para a melhor IA baseado no domínio científico:
    - **Claude**: Matemática, algoritmos, filosofia, consciência
    - **ChatGPT**: Biomateriais, engenharia, código, debugging  
    - **Gemini**: Literatura, research synthesis, escrita acadêmica
    
    **Features:**
    - Roteamento automático por domínio
    - Context sharing entre IAs
    - Performance learning
    - Fallback inteligente
    """
    return await hub_instance.chat_with_routing(request)


@router.post("/chat/direct/{ai_model}", response_model=ChatResponse)
async def direct_ai_chat(
    request: ChatRequest,
    ai_model: AIModel = Path(..., description="Modelo específico de IA"),
    hub_instance=Depends(get_hub)
):
    """
    🎯 **Chat Direto com Modelo Específico**
    
    Bypass do roteamento para usar diretamente um modelo específico:
    - `gpt-4-turbo`: ChatGPT mais avançado
    - `claude-3-5-sonnet`: Claude top reasoning
    - `gemini-pro`: Google AI research-focused
    """
    return await hub_instance.direct_chat(ai_model, request)


# ==================== SPECIALIZED CHAT ENDPOINTS ====================

@router.post("/chat/biomaterials", response_model=ChatResponse)
async def biomaterials_chat(request: BiomaterialsChatRequest, hub_instance=Depends(get_hub)):
    """🧬 **Chat Especializado em Biomateriais** - Otimizado para ChatGPT"""
    base_request = ChatRequest(
        message=request.message,
        domain=ScientificDomain.BIOMATERIALS,
        preferred_ai=AIProvider.CHATGPT,
        **request.dict(exclude={"message", "domain", "scaffold_type", "material_properties", "kec_metrics"})
    )
    return await hub_instance.chat_with_routing(base_request)


@router.post("/chat/mathematical", response_model=ChatResponse)
async def mathematical_chat(request: MathematicalChatRequest, hub_instance=Depends(get_hub)):
    """🔢 **Chat Especializado em Matemática** - Otimizado para Claude"""
    base_request = ChatRequest(
        message=request.message,
        domain=ScientificDomain.MATHEMATICAL_PROOFS,
        preferred_ai=AIProvider.CLAUDE,
        **request.dict(exclude={"message", "proof_type", "complexity_level", "mathematical_domain"})
    )
    return await hub_instance.chat_with_routing(base_request)


@router.post("/chat/philosophy", response_model=ChatResponse)
async def philosophy_chat(request: PhilosophyChatRequest, hub_instance=Depends(get_hub)):
    """🤔 **Chat Especializado em Filosofia** - Otimizado para Claude"""
    base_request = ChatRequest(
        message=request.message,
        domain=ScientificDomain.PHILOSOPHY,
        preferred_ai=AIProvider.CLAUDE,
        **request.dict(exclude={"message", "philosophical_school", "ethical_framework", "consciousness_level"})
    )
    return await hub_instance.chat_with_routing(base_request)


@router.post("/chat/research", response_model=ChatResponse)
async def research_chat(request: ResearchChatRequest, hub_instance=Depends(get_hub)):
    """📚 **Chat Especializado em Research** - Otimizado para Gemini"""
    base_request = ChatRequest(
        message=request.message,
        domain=ScientificDomain.RESEARCH_SYNTHESIS,
        preferred_ai=AIProvider.GEMINI,
        **request.dict(exclude={"message", "research_field", "paper_type", "citation_style", "target_journal"})
    )
    return await hub_instance.chat_with_routing(base_request)


@router.post("/chat/cross-domain", response_model=ChatResponse)
async def cross_domain_chat(request: CrossDomainChatRequest, hub_instance=Depends(get_hub)):
    """🔗 **Chat Interdisciplinar** - Roteamento inteligente baseado no domínio primário"""
    base_request = ChatRequest(
        message=request.message,
        domain=request.primary_domain,
        **request.dict(exclude={"message", "primary_domain", "secondary_domains", "integration_approach", "complexity_preference"})
    )
    return await hub_instance.chat_with_routing(base_request)


# ==================== CONVERSATION MANAGEMENT ====================

@router.post("/conversations/create", response_model=Dict[str, str])
async def create_conversation(request: ConversationCreate, hub_instance=Depends(get_hub)):
    """
    🗨️ **Criar Nova Conversa**
    
    Organize conversas por domínio científico e projeto de pesquisa.
    """
    conversation_id = await hub_instance.conversation_manager.create_conversation(request)
    return {"conversation_id": conversation_id, "status": "created"}


@router.get("/conversations", response_model=List[ConversationHistory])
async def list_conversations(
    domain: Optional[ScientificDomain] = Query(None, description="Filtrar por domínio"),
    ai_provider: Optional[AIProvider] = Query(None, description="Filtrar por IA"),
    project_id: Optional[str] = Query(None, description="Filtrar por projeto"),
    status: Optional[str] = Query(None, description="Filtrar por status"),
    limit: int = Query(50, ge=1, le=100, description="Limite de resultados"),
    hub_instance=Depends(get_hub)
):
    """
    📋 **Listar Conversas**
    
    Liste conversas com filtros avançados por domínio, IA, projeto e status.
    """
    return await hub_instance.conversation_manager.list_conversations(
        domain=domain, ai_provider=ai_provider, project_id=project_id, 
        status=status, limit=limit
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationHistory)
async def get_conversation(
    conversation_id: str = Path(..., description="ID da conversa"),
    include_insights: bool = Query(True, description="Incluir insights extraídos"),
    hub_instance=Depends(get_hub)
):
    """
    🔍 **Recuperar Conversa Específica**
    
    Obtenha histórico completo da conversa com insights automáticos.
    """
    conversation = await hub_instance.conversation_manager.get_conversation(
        conversation_id, include_insights=include_insights
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


# ==================== CONTEXT MANAGEMENT ====================

@router.post("/context/sync", response_model=Dict[str, Any])
async def sync_context(request: ContextSyncRequest, hub_instance=Depends(get_hub)):
    """
    🔄 **Sincronização Manual de Contexto**
    
    Sincronize contexto entre diferentes IAs para continuidade de conversas.
    """
    return await hub_instance.context_bridge.sync_context(request)


@router.get("/context/timeline/{domain}", response_model=Dict[str, Any])
async def get_research_timeline(
    domain: ScientificDomain = Path(..., description="Domínio científico"),
    hub_instance=Depends(get_hub)
):
    """
    📈 **Timeline de Pesquisa por Domínio**
    
    Visualize a evolução de insights e descobertas em um domínio específico.
    """
    timeline = await hub_instance.context_bridge.get_research_timeline(domain)
    if not timeline:
        raise HTTPException(status_code=404, detail="No timeline found for domain")
    
    from dataclasses import asdict
    return asdict(timeline)


@router.get("/context/connections", response_model=List[Dict[str, Any]])
async def get_cross_domain_connections(
    domain: Optional[ScientificDomain] = Query(None, description="Domínio específico"),
    hub_instance=Depends(get_hub)
):
    """
    🌐 **Conexões Entre Domínios**
    
    Descubra conexões interdisciplinares automáticas descobertas pelo sistema.
    """
    connections = await hub_instance.context_bridge.get_cross_domain_connections(domain)
    
    from dataclasses import asdict
    return [asdict(conn) for conn in connections]


# ==================== MODEL MANAGEMENT ====================

@router.get("/models/available", response_model=Dict[str, List[str]])
async def get_available_models(hub_instance=Depends(get_hub)):
    """
    🤖 **Modelos de IA Disponíveis**
    
    Liste todos os modelos disponíveis por provedor de IA.
    """
    available_models = {}
    
    for ai_provider, client in hub_instance.ai_clients.items():
        if hub_instance._client_health.get(ai_provider, False):
            models = client.get_available_models() if hasattr(client, 'get_available_models') else []
            available_models[ai_provider.value] = [model.value for model in models]
    
    return available_models


@router.post("/models/recommend", response_model=ModelRecommendation)
async def recommend_model(
    question: str = Query(..., description="Pergunta ou tópico"),
    context: Optional[str] = Query(None, description="Contexto adicional"),
    hub_instance=Depends(get_hub)
):
    """
    🎯 **Recomendação Inteligente de Modelo**
    
    Obtenha recomendação do melhor modelo/IA para sua pergunta específica.
    """
    return await hub_instance.orchestrator.get_model_recommendation(question, {"context": context})


# ==================== ROUTING CONFIGURATION ====================

@router.get("/routing/rules", response_model=Dict[str, Dict[str, Any]])
async def get_routing_rules(hub_instance=Depends(get_hub)):
    """
    ⚙️ **Regras de Roteamento**
    
    Visualize as regras de roteamento por domínio científico.
    """
    rules = {}
    for domain, rule in hub_instance.orchestrator.routing_rules.items():
        # Convert RoutingRule to dict manually since it's a Pydantic model
        rules[domain.value] = rule.dict() if hasattr(rule, 'dict') else {
            "domain": rule.domain.value,
            "preferred_ai": rule.preferred_ai.value,
            "model": rule.model.value,
            "confidence_threshold": rule.confidence_threshold,
            "keywords": rule.keywords,
            "fallback_ai": rule.fallback_ai.value if rule.fallback_ai else None,
            "reasoning": rule.reasoning
        }
    
    return rules


@router.get("/routing/stats", response_model=Dict[str, Any])
async def get_routing_stats(hub_instance=Depends(get_hub)):
    """
    📊 **Estatísticas de Roteamento**
    
    Analise padrões de uso e eficiência do roteamento inteligente.
    """
    return hub_instance.orchestrator.get_routing_stats()


# ==================== ANALYTICS ENDPOINTS ====================

@router.get("/analytics/usage", response_model=Dict[str, Any])
async def get_usage_analytics(
    time_range: str = Query("24h", regex="^(1h|6h|24h|7d|30d)$", description="Período de análise"),
    hub_instance=Depends(get_hub)
):
    """
    📈 **Estatísticas de Uso por IA**
    
    Analise uso, performance e custos de cada provedor de IA.
    """
    return {
        "time_range": time_range,
        "global_stats": hub_instance._global_stats,
        "orchestrator_stats": hub_instance.orchestrator.get_routing_stats(),
        "conversation_stats": hub_instance.conversation_manager.get_statistics(),
        "context_stats": hub_instance.context_bridge.get_statistics()
    }


@router.get("/analytics/performance", response_model=List[Dict[str, Any]])
async def get_performance_analytics(hub_instance=Depends(get_hub)):
    """
    ⚡ **Performance Comparativa**
    
    Compare latência, qualidade e sucesso entre diferentes IAs.
    """
    performance_data = []
    
    for ai_provider, client in hub_instance.ai_clients.items():
        if hub_instance._client_health.get(ai_provider, False):
            health = await client.health_check() if hasattr(client, 'health_check') else {}
            performance_data.append({
                "ai_provider": ai_provider.value,
                "status": "healthy" if health.get("healthy", False) else "unhealthy",
                "specialties": client.get_specialties() if hasattr(client, 'get_specialties') else [],
                "health_details": health
            })
    
    return performance_data


@router.get("/analytics/costs", response_model=Dict[str, Any])
async def get_cost_analytics(hub_instance=Depends(get_hub)):
    """
    💰 **Análise de Custos**
    
    Monitore custos por IA, domínio e projeto de pesquisa.
    """
    return {
        "total_estimated_cost": hub_instance._global_stats["total_cost"],
        "total_tokens": hub_instance._global_stats["total_tokens"],
        "cost_per_token": hub_instance._global_stats["total_cost"] / max(hub_instance._global_stats["total_tokens"], 1),
        "cost_breakdown": "Would include per-AI cost analysis in production"
    }


@router.get("/insights/extracted", response_model=List[Dict[str, Any]])
async def get_extracted_insights(
    domain: Optional[ScientificDomain] = Query(None, description="Filtrar por domínio"),
    limit: int = Query(20, ge=1, le=100, description="Limite de resultados"),
    hub_instance=Depends(get_hub)
):
    """
    💡 **Insights Extraídos Automaticamente**
    
    Descubra insights e descobertas automáticas extraídas das conversas.
    """
    # Get insights from conversation manager
    insights = []
    
    conversations = await hub_instance.conversation_manager.list_conversations(
        domain=domain, limit=limit
    )
    
    for conv in conversations:
        conv_insights = await hub_instance.conversation_manager.extract_conversation_insights(conv.conversation_id)
        
        for insight in conv_insights:
            from dataclasses import asdict
            insights.append(asdict(insight))
    
    return insights


# ==================== HEALTH CHECK ====================

@router.get("/health", response_model=MultiAIHealthCheck)
async def health_check():
    """
    🏥 **Health Check Completo do Multi-AI Hub**
    
    Monitore status de todos os componentes do sistema revolucionário.
    """
    if not hub.enabled:
        return MultiAIHealthCheck(
            status="unhealthy",
            ai_providers={},
            routing_engine={"status": "offline"},
            context_bridge={"status": "offline"},
            conversation_manager={"status": "offline"},
            uptime_seconds=0.0
        )
    
    # Check AI providers
    ai_providers_status = {}
    for ai_provider, client in hub.ai_clients.items():
        try:
            client_health = await client.health_check() if hasattr(client, 'health_check') else {"healthy": False}
            ai_providers_status[ai_provider] = client_health
        except Exception as e:
            ai_providers_status[ai_provider] = {"healthy": False, "error": str(e)}
    
    # Check core components
    orchestrator_health = await hub.orchestrator.health_check()
    context_health = await hub.context_bridge.health_check()
    conversation_health = await hub.conversation_manager.health_check()
    
    # Determine overall status
    all_healthy = (
        orchestrator_health.get("healthy", False) and
        context_health.get("healthy", False) and
        conversation_health.get("healthy", False) and
        any(status.get("healthy", False) for status in ai_providers_status.values())
    )
    
    uptime_seconds = (datetime.now(timezone.utc) - hub._global_stats["uptime_start"]).total_seconds()
    
    return MultiAIHealthCheck(
        status="healthy" if all_healthy else "degraded",
        ai_providers=ai_providers_status,
        routing_engine=orchestrator_health,
        context_bridge=context_health,
        conversation_manager=conversation_health,
        total_conversations=len(hub.conversation_manager.conversations),
        uptime_seconds=uptime_seconds
    )


# ==================== INITIALIZATION ====================

async def initialize_multi_ai_hub():
    """Inicializar Multi-AI Hub."""
    try:
        await hub.initialize()
        logger.info("🎉 Multi-AI Hub initialization complete!")
    except Exception as e:
        logger.error(f"Multi-AI Hub initialization failed: {e}")
        raise


async def shutdown_multi_ai_hub():
    """Shutdown Multi-AI Hub."""
    try:
        await hub.shutdown()
        logger.info("Multi-AI Hub shutdown complete")
    except Exception as e:
        logger.error(f"Multi-AI Hub shutdown error: {e}")


# Export hub instance for use in main app
__all__ = ["router", "hub", "initialize_multi_ai_hub", "shutdown_multi_ai_hub"]