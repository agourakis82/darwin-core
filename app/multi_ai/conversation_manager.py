"""Conversation Manager - Gerenciamento avançado de conversas por domínio científico."""

import asyncio
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections import defaultdict

from ..core.logging import get_logger
from ..models.multi_ai_models import (
    AIProvider, ScientificDomain, ChatMessage, ConversationCreate, 
    ConversationHistory, ExtractedInsight
)

logger = get_logger("multi_ai.conversation_manager")


@dataclass
class ConversationThread:
    """Thread de conversa com metadata rica."""
    id: str
    title: Optional[str] = None
    domain: Optional[ScientificDomain] = None
    participants: Set[AIProvider] = field(default_factory=set)
    messages: List[ChatMessage] = field(default_factory=list)
    insights: List[Dict[str, Any]] = field(default_factory=list)
    project_id: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    status: str = "active"  # active, paused, archived, completed
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_tokens: int = 0
    estimated_cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchProject:
    """Projeto de pesquisa com múltiplas conversas."""
    id: str
    title: str
    description: Optional[str] = None
    primary_domain: ScientificDomain = ScientificDomain.INTERDISCIPLINARY
    secondary_domains: List[ScientificDomain] = field(default_factory=list)
    conversations: Set[str] = field(default_factory=set)  # conversation_ids
    collaborators: Set[AIProvider] = field(default_factory=set)
    key_insights: List[str] = field(default_factory=list)
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "active"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationManager:
    """Sistema de gerenciamento avançado de conversas por domínio científico."""
    
    def __init__(self, context_bridge=None):
        self.enabled = False
        self.context_bridge = context_bridge  # Integration with Context Bridge
        
        # Core storage
        self.conversations: Dict[str, ConversationThread] = {}
        self.projects: Dict[str, ResearchProject] = {}
        
        # Organization indexes
        self.domain_conversations: Dict[ScientificDomain, Set[str]] = defaultdict(set)
        self.ai_conversations: Dict[AIProvider, Set[str]] = defaultdict(set)
        self.project_conversations: Dict[str, Set[str]] = defaultdict(set)
        
        # Research tracking
        self.active_collaborations: Dict[str, Dict[str, Any]] = {}  # multi-AI collaborations
        self.insight_pipeline: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.domain_analytics: Dict[ScientificDomain, Dict[str, Any]] = defaultdict(dict)
        
        # Statistics
        self._stats = {
            "total_conversations": 0,
            "active_conversations": 0,
            "total_projects": 0,
            "cross_domain_conversations": 0,
            "multi_ai_conversations": 0,
            "insights_extracted": 0,
            "avg_conversation_length": 0.0
        }
        
    async def initialize(self):
        """Inicializar Conversation Manager."""
        logger.info("Initializing Conversation Manager...")
        
        # Initialize domain analytics
        await self._initialize_domain_analytics()
        
        # Load existing conversations (placeholder)
        await self._load_conversations()
        
        self.enabled = True
        logger.info("✅ Conversation Manager initialized with domain-based organization")
        logger.info(f"Tracking {len(ScientificDomain)} scientific domains")
    
    async def shutdown(self):
        """Shutdown Conversation Manager."""
        logger.info("Shutting down Conversation Manager...")
        
        # Save conversations and projects
        await self._save_conversations()
        
        self.enabled = False
        logger.info("✅ Conversation Manager shutdown")
    
    async def create_conversation(self, request: ConversationCreate) -> str:
        """Criar nova conversa."""
        if not self.enabled:
            raise RuntimeError("Conversation Manager not initialized")
        
        try:
            # Generate conversation ID
            conversation_id = str(uuid.uuid4())
            
            # Create conversation thread
            thread = ConversationThread(
                id=conversation_id,
                title=request.title or f"Conversation {conversation_id[:8]}",
                domain=request.domain,
                participants=set(request.participants or []),
                project_id=None,
                tags=set(),
                metadata=request.metadata or {}
            )
            
            # Store conversation
            self.conversations[conversation_id] = thread
            
            # Update indexes
            if request.domain:
                self.domain_conversations[request.domain].add(conversation_id)
            
            for ai in thread.participants:
                self.ai_conversations[ai].add(conversation_id)
            
            # Update statistics
            self._stats["total_conversations"] += 1
            self._stats["active_conversations"] += 1
            
            if len(thread.participants) > 1:
                self._stats["multi_ai_conversations"] += 1
            
            # Initialize domain analytics if new domain
            if request.domain and request.domain not in self.domain_analytics:
                await self._initialize_domain_specific_analytics(request.domain)
            
            logger.info(f"Created conversation: {conversation_id} ({thread.title}) in domain {request.domain}")
            
            return conversation_id
            
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            raise
    
    async def add_message(self, conversation_id: str, message: ChatMessage, 
                         ai_provider: Optional[AIProvider] = None,
                         tokens_used: Optional[int] = None,
                         cost: Optional[float] = None) -> Dict[str, Any]:
        """Adicionar mensagem à conversa."""
        if not self.enabled:
            raise RuntimeError("Conversation Manager not initialized")
        
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        try:
            thread = self.conversations[conversation_id]
            
            # Add message to thread
            thread.messages.append(message)
            thread.updated_at = datetime.now(timezone.utc)
            
            # Update participant tracking
            if ai_provider and ai_provider not in thread.participants:
                thread.participants.add(ai_provider)
                self.ai_conversations[ai_provider].add(conversation_id)
            
            # Update usage statistics
            if tokens_used:
                thread.total_tokens += tokens_used
            if cost:
                thread.estimated_cost += cost
            
            # Extract insights from message
            insights = await self._extract_message_insights(message, thread.domain, conversation_id)
            if insights:
                thread.insights.extend(insights)
                self.insight_pipeline[conversation_id].extend(insights)
                self._stats["insights_extracted"] += len(insights)
            
            # Sync context if Context Bridge available
            if self.context_bridge and ai_provider:
                await self._sync_message_context(conversation_id, message, ai_provider)
            
            # Update domain analytics
            if thread.domain:
                await self._update_domain_analytics(thread.domain, message, ai_provider)
            
            # Check for collaboration patterns
            if len(thread.participants) > 1:
                await self._track_collaboration(conversation_id, ai_provider, message)
            
            logger.debug(f"Added message to conversation {conversation_id}: {len(message.content)} chars")
            
            return {
                "message_added": True,
                "conversation_id": conversation_id,
                "message_count": len(thread.messages),
                "participants": list(thread.participants),
                "insights_extracted": len(insights),
                "total_tokens": thread.total_tokens
            }
            
        except Exception as e:
            logger.error(f"Error adding message to conversation {conversation_id}: {e}")
            raise
    
    async def get_conversation(self, conversation_id: str, 
                             include_insights: bool = True) -> Optional[ConversationHistory]:
        """Obter conversa completa."""
        if conversation_id not in self.conversations:
            return None
        
        try:
            thread = self.conversations[conversation_id]
            
            # Convert to ConversationHistory model
            history = ConversationHistory(
                conversation_id=conversation_id,
                title=thread.title,
                domain=thread.domain,
                messages=thread.messages,
                participants=list(thread.participants),
                created_at=thread.created_at,
                updated_at=thread.updated_at,
                total_messages=len(thread.messages),
                total_tokens=thread.total_tokens,
                estimated_cost=thread.estimated_cost,
                metadata={
                    "status": thread.status,
                    "project_id": thread.project_id,
                    "tags": list(thread.tags),
                    "insights_count": len(thread.insights) if include_insights else 0,
                    **thread.metadata
                }
            )
            
            if include_insights:
                history.metadata["insights"] = thread.insights
            
            return history
            
        except Exception as e:
            logger.error(f"Error retrieving conversation {conversation_id}: {e}")
            return None
    
    async def list_conversations(self, 
                               domain: Optional[ScientificDomain] = None,
                               ai_provider: Optional[AIProvider] = None,
                               project_id: Optional[str] = None,
                               status: Optional[str] = None,
                               limit: int = 50) -> List[ConversationHistory]:
        """Listar conversas com filtros."""
        
        try:
            # Get conversation IDs based on filters
            conversation_ids = set(self.conversations.keys())
            
            if domain:
                conversation_ids &= self.domain_conversations.get(domain, set())
            
            if ai_provider:
                conversation_ids &= self.ai_conversations.get(ai_provider, set())
            
            if project_id:
                conversation_ids &= self.project_conversations.get(project_id, set())
            
            # Apply status filter
            if status:
                conversation_ids = {
                    cid for cid in conversation_ids 
                    if self.conversations[cid].status == status
                }
            
            # Sort by update time and limit
            sorted_conversations = sorted(
                [self.conversations[cid] for cid in conversation_ids],
                key=lambda x: x.updated_at,
                reverse=True
            )[:limit]
            
            # Convert to ConversationHistory
            results = []
            for thread in sorted_conversations:
                history = await self.get_conversation(thread.id, include_insights=False)
                if history:
                    results.append(history)
            
            return results
            
        except Exception as e:
            logger.error(f"Error listing conversations: {e}")
            return []
    
    async def create_research_project(self, title: str, description: Optional[str] = None,
                                    primary_domain: ScientificDomain = ScientificDomain.INTERDISCIPLINARY,
                                    secondary_domains: Optional[List[ScientificDomain]] = None) -> str:
        """Criar projeto de pesquisa."""
        
        try:
            project_id = str(uuid.uuid4())
            
            project = ResearchProject(
                id=project_id,
                title=title,
                description=description,
                primary_domain=primary_domain,
                secondary_domains=secondary_domains or []
            )
            
            self.projects[project_id] = project
            self._stats["total_projects"] += 1
            
            logger.info(f"Created research project: {project_id} ({title})")
            
            return project_id
            
        except Exception as e:
            logger.error(f"Error creating research project: {e}")
            raise
    
    async def add_conversation_to_project(self, conversation_id: str, project_id: str) -> bool:
        """Adicionar conversa a projeto."""
        
        if conversation_id not in self.conversations or project_id not in self.projects:
            return False
        
        try:
            # Update conversation
            self.conversations[conversation_id].project_id = project_id
            
            # Update project
            self.projects[project_id].conversations.add(conversation_id)
            self.projects[project_id].updated_at = datetime.now(timezone.utc)
            
            # Update conversation participants to project collaborators
            thread = self.conversations[conversation_id]
            self.projects[project_id].collaborators.update(thread.participants)
            
            # Update index
            self.project_conversations[project_id].add(conversation_id)
            
            logger.info(f"Added conversation {conversation_id} to project {project_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding conversation to project: {e}")
            return False
    
    async def get_domain_analytics(self, domain: ScientificDomain) -> Dict[str, Any]:
        """Obter analytics de domínio específico."""
        
        analytics = self.domain_analytics.get(domain, {})
        
        # Add real-time statistics
        domain_conversations = self.domain_conversations.get(domain, set())
        
        analytics.update({
            "total_conversations": len(domain_conversations),
            "active_conversations": sum(
                1 for cid in domain_conversations 
                if self.conversations[cid].status == "active"
            ),
            "total_messages": sum(
                len(self.conversations[cid].messages) 
                for cid in domain_conversations
            ),
            "total_insights": sum(
                len(self.conversations[cid].insights) 
                for cid in domain_conversations
            ),
            "participating_ais": len(set().union(*[
                self.conversations[cid].participants 
                for cid in domain_conversations
            ])) if domain_conversations else 0
        })
        
        return analytics
    
    async def get_collaboration_analysis(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Analisar padrões de colaboração entre IAs."""
        
        try:
            if conversation_id:
                # Analyze specific conversation
                if conversation_id not in self.conversations:
                    return {}
                
                thread = self.conversations[conversation_id]
                return await self._analyze_conversation_collaboration(thread)
            else:
                # Analyze global collaboration patterns
                return await self._analyze_global_collaboration()
                
        except Exception as e:
            logger.error(f"Error analyzing collaboration: {e}")
            return {}
    
    async def extract_conversation_insights(self, conversation_id: str) -> List[ExtractedInsight]:
        """Extrair insights de conversa específica."""
        
        if conversation_id not in self.conversations:
            return []
        
        try:
            thread = self.conversations[conversation_id]
            insights = []
            
            for insight_data in thread.insights:
                insight = ExtractedInsight(
                    insight_id=insight_data.get("id", str(uuid.uuid4())),
                    content=insight_data.get("content", ""),
                    source_conversation=conversation_id,
                    domains=[thread.domain] if thread.domain else [],
                    confidence=insight_data.get("confidence", 0.7),
                    novelty_score=insight_data.get("novelty_score"),
                    related_concepts=insight_data.get("concepts", []),
                    potential_applications=insight_data.get("applications", []),
                    extracted_at=datetime.fromisoformat(insight_data.get("timestamp", datetime.now(timezone.utc).isoformat())),
                    ai_source=AIProvider(insight_data.get("source_ai", "chatgpt"))
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error extracting conversation insights: {e}")
            return []
    
    # Private methods
    
    async def _extract_message_insights(self, message: ChatMessage, 
                                      domain: Optional[ScientificDomain],
                                      conversation_id: str) -> List[Dict[str, Any]]:
        """Extrair insights de mensagem."""
        
        insights = []
        
        if not message.content or len(message.content) < 50:
            return insights
        
        try:
            # Use Context Bridge if available
            if self.context_bridge:
                extracted_insights = await self.context_bridge.extract_insights([message], domain)
                
                for insight in extracted_insights:
                    insights.append({
                        "id": str(uuid.uuid4()),
                        "type": insight.get("type", "general"),
                        "content": insight.get("content", ""),
                        "summary": insight.get("summary", ""),
                        "confidence": insight.get("importance", 0.7),
                        "concepts": insight.get("concepts", []),
                        "keywords": insight.get("keywords", []),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source_ai": message.metadata.get("ai_provider") if message.metadata else None
                    })
            
        except Exception as e:
            logger.error(f"Error extracting message insights: {e}")
        
        return insights
    
    async def _sync_message_context(self, conversation_id: str, message: ChatMessage, 
                                  ai_provider: AIProvider):
        """Sincronizar contexto da mensagem com Context Bridge."""
        
        if not self.context_bridge:
            return
        
        try:
            from ..models.multi_ai_models import ContextSyncRequest
            
            sync_request = ContextSyncRequest(
                conversation_id=conversation_id,
                context={"message_content": message.content},
                source_ai=ai_provider,
                target_ais=None,  # Sync to all
                priority="medium"
            )
            
            await self.context_bridge.sync_context(sync_request)
            
        except Exception as e:
            logger.error(f"Error syncing message context: {e}")
    
    async def _update_domain_analytics(self, domain: ScientificDomain, 
                                     message: ChatMessage, 
                                     ai_provider: Optional[AIProvider]):
        """Atualizar analytics de domínio."""
        
        if domain not in self.domain_analytics:
            self.domain_analytics[domain] = {
                "message_count": 0,
                "ai_usage": defaultdict(int),
                "avg_message_length": 0.0,
                "total_chars": 0,
                "last_activity": None
            }
        
        analytics = self.domain_analytics[domain]
        analytics["message_count"] += 1
        analytics["total_chars"] += len(message.content)
        analytics["avg_message_length"] = analytics["total_chars"] / analytics["message_count"]
        analytics["last_activity"] = datetime.now(timezone.utc).isoformat()
        
        if ai_provider:
            analytics["ai_usage"][ai_provider.value] += 1
    
    async def _track_collaboration(self, conversation_id: str, 
                                 ai_provider: Optional[AIProvider], 
                                 message: ChatMessage):
        """Rastrear padrões de colaboração."""
        
        if not ai_provider:
            return
        
        if conversation_id not in self.active_collaborations:
            self.active_collaborations[conversation_id] = {
                "participants": set(),
                "message_flow": [],
                "collaboration_score": 0.0,
                "started_at": datetime.now(timezone.utc).isoformat()
            }
        
        collab = self.active_collaborations[conversation_id]
        collab["participants"].add(ai_provider.value)
        collab["message_flow"].append({
            "ai": ai_provider.value,
            "timestamp": message.timestamp.isoformat() if message.timestamp else datetime.now(timezone.utc).isoformat(),
            "length": len(message.content),
            "role": message.role
        })
        
        # Calculate collaboration score
        if len(collab["participants"]) > 1:
            collab["collaboration_score"] = len(collab["participants"]) * 0.3 + len(collab["message_flow"]) * 0.1
    
    async def _analyze_conversation_collaboration(self, thread: ConversationThread) -> Dict[str, Any]:
        """Analisar colaboração em conversa específica."""
        
        if len(thread.participants) <= 1:
            return {"collaboration_type": "single_ai", "participants": list(thread.participants)}
        
        return {
            "collaboration_type": "multi_ai",
            "participants": list(thread.participants),
            "participant_count": len(thread.participants),
            "message_count": len(thread.messages),
            "collaboration_data": self.active_collaborations.get(thread.id, {})
        }
    
    async def _analyze_global_collaboration(self) -> Dict[str, Any]:
        """Analisar padrões globais de colaboração."""
        
        multi_ai_conversations = [
            cid for cid, thread in self.conversations.items() 
            if len(thread.participants) > 1
        ]
        
        ai_pairs = defaultdict(int)
        for thread in self.conversations.values():
            if len(thread.participants) > 1:
                participants = sorted(list(thread.participants), key=lambda x: x.value)
                for i, ai1 in enumerate(participants):
                    for ai2 in participants[i+1:]:
                        ai_pairs[(ai1.value, ai2.value)] += 1
        
        return {
            "multi_ai_conversations": len(multi_ai_conversations),
            "total_conversations": len(self.conversations),
            "collaboration_rate": len(multi_ai_conversations) / max(len(self.conversations), 1),
            "popular_ai_pairs": dict(sorted(ai_pairs.items(), key=lambda x: x[1], reverse=True)[:5])
        }
    
    async def _initialize_domain_analytics(self):
        """Inicializar analytics por domínio."""
        
        for domain in ScientificDomain:
            await self._initialize_domain_specific_analytics(domain)
    
    async def _initialize_domain_specific_analytics(self, domain: ScientificDomain):
        """Inicializar analytics específicas de domínio."""
        
        self.domain_analytics[domain] = {
            "message_count": 0,
            "ai_usage": defaultdict(int),
            "avg_message_length": 0.0,
            "total_chars": 0,
            "last_activity": None,
            "specialized_metrics": {}
        }
    
    async def _load_conversations(self):
        """Carregar conversas existentes (placeholder)."""
        logger.info("Loading existing conversations...")
        # This would load from actual persistence
    
    async def _save_conversations(self):
        """Salvar conversas (placeholder)."""
        logger.info("Saving conversations...")
        # This would save to actual persistence
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obter estatísticas completas."""
        
        # Calculate dynamic statistics
        active_count = sum(1 for thread in self.conversations.values() if thread.status == "active")
        avg_length = sum(len(thread.messages) for thread in self.conversations.values()) / max(len(self.conversations), 1)
        cross_domain_count = sum(1 for thread in self.conversations.values() if thread.project_id and thread.domain != ScientificDomain.INTERDISCIPLINARY)
        
        self._stats.update({
            "active_conversations": active_count,
            "avg_conversation_length": avg_length,
            "cross_domain_conversations": cross_domain_count
        })
        
        return self._stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check do Conversation Manager."""
        
        return {
            "healthy": self.enabled,
            "status": "operational" if self.enabled else "offline",
            "statistics": self.get_statistics(),
            "domain_count": len(self.domain_analytics),
            "project_count": len(self.projects),
            "last_check": datetime.now(timezone.utc).isoformat()
        }


__all__ = ["ConversationManager", "ConversationThread", "ResearchProject"]