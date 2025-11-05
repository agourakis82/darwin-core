"""Context Bridge - Sistema de compartilhamento de contexto entre IAs."""

import asyncio
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from collections import defaultdict

from ..core.logging import get_logger
from ..models.multi_ai_models import (
    AIProvider, ScientificDomain, ContextSyncRequest, ChatMessage
)

logger = get_logger("multi_ai.context_bridge")


@dataclass
class ContextEntry:
    """Entrada individual de contexto."""
    id: str
    content: str
    source_ai: AIProvider
    domain: Optional[ScientificDomain]
    conversation_id: str
    timestamp: datetime
    importance_score: float
    keywords: List[str]
    related_concepts: List[str]
    context_type: str  # "insight", "definition", "method", "result", "question"
    metadata: Dict[str, Any]


@dataclass
class ResearchTimeline:
    """Timeline de descobertas e insights de pesquisa."""
    domain: ScientificDomain
    entries: List[ContextEntry]
    key_insights: List[str]
    evolution_summary: str
    last_updated: datetime


@dataclass
class CrossDomainConnection:
    """Conexão entre domínios diferentes."""
    domain_a: ScientificDomain
    domain_b: ScientificDomain
    connection_strength: float
    shared_concepts: List[str]
    bridge_insights: List[str]
    examples: List[str]
    last_updated: datetime


class ContextBridge:
    """Sistema de compartilhamento de contexto entre IAs e domínios."""
    
    def __init__(self):
        self.enabled = False
        
        # Core storage
        self.shared_contexts: Dict[str, ContextEntry] = {}
        self.conversation_contexts: Dict[str, List[str]] = defaultdict(list)  # conversation_id -> context_ids
        self.domain_contexts: Dict[ScientificDomain, List[str]] = defaultdict(list)  # domain -> context_ids
        self.ai_contexts: Dict[AIProvider, List[str]] = defaultdict(list)  # ai -> context_ids
        
        # Research organization
        self.research_timelines: Dict[ScientificDomain, ResearchTimeline] = {}
        self.cross_domain_connections: Dict[Tuple[ScientificDomain, ScientificDomain], CrossDomainConnection] = {}
        self.project_contexts: Dict[str, Dict[str, Any]] = {}  # project_id -> context
        
        # Insight detection
        self.insight_patterns = self._initialize_insight_patterns()
        self.concept_network: Dict[str, Set[str]] = defaultdict(set)  # concept -> related_concepts
        
        # Statistics
        self._stats = {
            "total_contexts": 0,
            "cross_ai_syncs": 0,
            "insights_extracted": 0,
            "domain_connections": 0,
            "avg_context_reuse": 0.0
        }
        
    async def initialize(self):
        """Inicializar Context Bridge."""
        logger.info("Initializing Context Bridge...")
        
        # Initialize insight patterns and concept network
        await self._build_concept_network()
        await self._initialize_research_timelines()
        
        self.enabled = True
        logger.info("✅ Context Bridge initialized with cross-AI context sharing")
        logger.info(f"Initialized {len(self.insight_patterns)} insight patterns")
    
    async def shutdown(self):
        """Shutdown Context Bridge."""
        logger.info("Shutting down Context Bridge...")
        
        # Save persistent contexts
        await self._save_persistent_contexts()
        
        self.enabled = False
        logger.info("✅ Context Bridge shutdown")
    
    async def sync_context(self, request: ContextSyncRequest) -> Dict[str, Any]:
        """Sincronizar contexto entre IAs."""
        if not self.enabled:
            raise RuntimeError("Context Bridge not initialized")
        
        try:
            logger.info(f"Syncing context from {request.source_ai} to {request.target_ais or 'all AIs'}")
            
            # Extract and process context
            context_entries = await self._extract_context_entries(request)
            
            # Store in shared context
            stored_ids = []
            for entry in context_entries:
                context_id = await self._store_context_entry(entry)
                stored_ids.append(context_id)
            
            # Update research timelines
            await self._update_research_timelines(context_entries)
            
            # Detect cross-domain connections
            await self._detect_cross_domain_connections(context_entries)
            
            # Update statistics
            self._stats["cross_ai_syncs"] += 1
            self._stats["total_contexts"] = len(self.shared_contexts)
            
            return {
                "success": True,
                "synced_contexts": len(context_entries),
                "stored_ids": stored_ids,
                "target_ais": request.target_ais or "all",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Context sync error: {e}")
            raise
    
    async def get_relevant_context(self, 
                                  conversation_id: str,
                                  domain: Optional[ScientificDomain] = None,
                                  ai_provider: Optional[AIProvider] = None,
                                  keywords: Optional[List[str]] = None,
                                  max_contexts: int = 10) -> List[ContextEntry]:
        """Obter contexto relevante para uma conversa."""
        
        relevant_contexts = []
        
        try:
            # Get conversation-specific contexts
            conv_context_ids = self.conversation_contexts.get(conversation_id, [])
            
            # Get domain-specific contexts if specified
            domain_context_ids = []
            if domain:
                domain_context_ids = self.domain_contexts.get(domain, [])
            
            # Get AI-specific contexts if specified
            ai_context_ids = []
            if ai_provider:
                ai_context_ids = self.ai_contexts.get(ai_provider, [])
            
            # Combine and deduplicate
            all_context_ids = set(conv_context_ids + domain_context_ids + ai_context_ids)
            
            # Score contexts by relevance
            scored_contexts = []
            for context_id in all_context_ids:
                if context_id in self.shared_contexts:
                    context = self.shared_contexts[context_id]
                    relevance_score = await self._calculate_context_relevance(
                        context, domain, keywords
                    )
                    scored_contexts.append((relevance_score, context))
            
            # Sort by relevance and return top contexts
            scored_contexts.sort(key=lambda x: x[0], reverse=True)
            relevant_contexts = [ctx for _, ctx in scored_contexts[:max_contexts]]
            
            logger.info(f"Retrieved {len(relevant_contexts)} relevant contexts for conversation {conversation_id}")
            
        except Exception as e:
            logger.error(f"Error retrieving relevant context: {e}")
        
        return relevant_contexts
    
    async def extract_insights(self, messages: List[ChatMessage], 
                              domain: Optional[ScientificDomain] = None) -> List[Dict[str, Any]]:
        """Extrair insights automaticamente das mensagens."""
        
        insights = []
        
        try:
            for message in messages:
                content = message.content
                
                # Apply insight detection patterns
                for pattern_name, pattern_info in self.insight_patterns.items():
                    if await self._matches_insight_pattern(content, pattern_info, domain):
                        insight = await self._create_insight(
                            content, pattern_name, pattern_info, domain, message
                        )
                        insights.append(insight)
                        
                        logger.info(f"Extracted {pattern_name} insight: {insight['summary'][:100]}...")
            
            # Update statistics
            self._stats["insights_extracted"] += len(insights)
            
        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
        
        return insights
    
    async def get_research_timeline(self, domain: ScientificDomain) -> Optional[ResearchTimeline]:
        """Obter timeline de pesquisa para domínio."""
        return self.research_timelines.get(domain)
    
    async def get_cross_domain_connections(self, 
                                         domain: Optional[ScientificDomain] = None) -> List[CrossDomainConnection]:
        """Obter conexões entre domínios."""
        
        if domain:
            # Return connections involving specific domain
            connections = []
            for (domain_a, domain_b), connection in self.cross_domain_connections.items():
                if domain_a == domain or domain_b == domain:
                    connections.append(connection)
            return connections
        else:
            # Return all connections
            return list(self.cross_domain_connections.values())
    
    async def create_project_context(self, project_id: str, 
                                   title: str,
                                   domains: List[ScientificDomain],
                                   description: Optional[str] = None) -> Dict[str, Any]:
        """Criar contexto de projeto para organizar pesquisa."""
        
        project_context = {
            "project_id": project_id,
            "title": title,
            "domains": [d.value for d in domains],
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "conversations": [],
            "key_insights": [],
            "timeline": [],
            "collaborating_ais": [],
            "metadata": {}
        }
        
        self.project_contexts[project_id] = project_context
        
        logger.info(f"Created project context: {project_id} ({title})")
        
        return project_context
    
    async def _extract_context_entries(self, request: ContextSyncRequest) -> List[ContextEntry]:
        """Extrair entradas de contexto do request."""
        
        entries = []
        context_data = request.context
        
        # Detect context type and extract relevant information
        for key, value in context_data.items():
            if isinstance(value, str) and len(value) > 20:  # Meaningful text content
                
                # Analyze content for importance and type
                importance_score = await self._calculate_importance_score(value)
                context_type = await self._detect_context_type(value)
                keywords = await self._extract_keywords(value)
                concepts = await self._extract_concepts(value)
                
                entry = ContextEntry(
                    id=f"{request.conversation_id}_{key}_{int(datetime.now().timestamp())}",
                    content=value,
                    source_ai=request.source_ai,
                    domain=self._infer_domain_from_context(value),
                    conversation_id=request.conversation_id,
                    timestamp=datetime.now(timezone.utc),
                    importance_score=importance_score,
                    keywords=keywords,
                    related_concepts=concepts,
                    context_type=context_type,
                    metadata={"original_key": key, "sync_priority": request.priority}
                )
                
                entries.append(entry)
        
        return entries
    
    async def _store_context_entry(self, entry: ContextEntry) -> str:
        """Armazenar entrada de contexto."""
        
        # Store in main context storage
        self.shared_contexts[entry.id] = entry
        
        # Index by conversation
        self.conversation_contexts[entry.conversation_id].append(entry.id)
        
        # Index by domain
        if entry.domain:
            self.domain_contexts[entry.domain].append(entry.id)
        
        # Index by AI
        self.ai_contexts[entry.source_ai].append(entry.id)
        
        # Update concept network
        for concept in entry.related_concepts:
            for other_concept in entry.related_concepts:
                if concept != other_concept:
                    self.concept_network[concept].add(other_concept)
        
        return entry.id
    
    async def _update_research_timelines(self, entries: List[ContextEntry]):
        """Atualizar timelines de pesquisa."""
        
        for entry in entries:
            if entry.domain and entry.importance_score > 0.7:  # High importance threshold
                
                if entry.domain not in self.research_timelines:
                    self.research_timelines[entry.domain] = ResearchTimeline(
                        domain=entry.domain,
                        entries=[],
                        key_insights=[],
                        evolution_summary="",
                        last_updated=datetime.now(timezone.utc)
                    )
                
                timeline = self.research_timelines[entry.domain]
                timeline.entries.append(entry)
                timeline.entries.sort(key=lambda x: x.timestamp)
                
                # Update key insights
                if entry.context_type == "insight" and entry.content not in timeline.key_insights:
                    timeline.key_insights.append(entry.content[:200] + "...")
                
                timeline.last_updated = datetime.now(timezone.utc)
    
    async def _detect_cross_domain_connections(self, entries: List[ContextEntry]):
        """Detectar conexões entre domínios."""
        
        # Find entries that mention multiple domains
        for entry in entries:
            if not entry.domain:
                continue
                
            # Check for mentions of other domains in content
            mentioned_domains = []
            for domain in ScientificDomain:
                if domain != entry.domain:
                    domain_keywords = self._get_domain_keywords(domain)
                    if any(keyword.lower() in entry.content.lower() for keyword in domain_keywords):
                        mentioned_domains.append(domain)
            
            # Create or update cross-domain connections
            for mentioned_domain in mentioned_domains:
                sorted_domains = sorted([entry.domain, mentioned_domain], key=lambda x: x.value)
                connection_key = (sorted_domains[0], sorted_domains[1])
                
                if connection_key not in self.cross_domain_connections:
                    self.cross_domain_connections[connection_key] = CrossDomainConnection(
                        domain_a=connection_key[0],
                        domain_b=connection_key[1],
                        connection_strength=0.0,
                        shared_concepts=[],
                        bridge_insights=[],
                        examples=[],
                        last_updated=datetime.now(timezone.utc)
                    )
                
                connection = self.cross_domain_connections[connection_key]
                connection.connection_strength += entry.importance_score
                
                # Add bridge insight if highly important
                if entry.importance_score > 0.8:
                    insight_summary = entry.content[:150] + "..."
                    if insight_summary not in connection.bridge_insights:
                        connection.bridge_insights.append(insight_summary)
                
                connection.last_updated = datetime.now(timezone.utc)
        
        # Update statistics
        self._stats["domain_connections"] = len(self.cross_domain_connections)
    
    def _initialize_insight_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Inicializar padrões de detecção de insights."""
        
        return {
            "novel_finding": {
                "keywords": ["discovered", "novel", "breakthrough", "unprecedented", "first time", "new finding"],
                "importance_weight": 1.0,
                "description": "Novel research findings or discoveries"
            },
            
            "method_innovation": {
                "keywords": ["new method", "approach", "technique", "algorithm", "methodology", "protocol"],
                "importance_weight": 0.9,
                "description": "New methods or approaches"
            },
            
            "theoretical_insight": {
                "keywords": ["theory", "theoretical", "hypothesis", "model", "framework", "principle"],
                "importance_weight": 0.8,
                "description": "Theoretical insights and frameworks"
            },
            
            "practical_application": {
                "keywords": ["application", "clinical", "therapeutic", "treatment", "implementation", "practical"],
                "importance_weight": 0.9,
                "description": "Practical applications and implementations"
            },
            
            "interdisciplinary_connection": {
                "keywords": ["bridge", "connection", "interdisciplinary", "combine", "integrate", "cross-domain"],
                "importance_weight": 0.95,
                "description": "Connections between different disciplines"
            },
            
            "limitation_insight": {
                "keywords": ["limitation", "challenge", "problem", "constraint", "difficulty", "barrier"],
                "importance_weight": 0.6,
                "description": "Important limitations and challenges"
            },
            
            "future_direction": {
                "keywords": ["future", "next step", "potential", "opportunity", "direction", "prospect"],
                "importance_weight": 0.7,
                "description": "Future research directions"
            }
        }
    
    async def _calculate_context_relevance(self, context: ContextEntry, 
                                         domain: Optional[ScientificDomain],
                                         keywords: Optional[List[str]]) -> float:
        """Calcular relevância de contexto."""
        
        relevance = context.importance_score  # Base score
        
        # Domain relevance
        if domain and context.domain == domain:
            relevance += 0.3
        
        # Keyword relevance
        if keywords:
            keyword_matches = sum(1 for keyword in keywords 
                                if keyword.lower() in context.content.lower())
            relevance += (keyword_matches / len(keywords)) * 0.2
        
        # Recency boost
        age_days = (datetime.now(timezone.utc) - context.timestamp).days
        if age_days < 7:
            relevance += 0.1
        elif age_days < 30:
            relevance += 0.05
        
        return min(relevance, 1.0)
    
    async def _calculate_importance_score(self, content: str) -> float:
        """Calcular score de importância do conteúdo."""
        
        score = 0.5  # Base score
        content_lower = content.lower()
        
        # Check for insight patterns
        for pattern_name, pattern_info in self.insight_patterns.items():
            keyword_matches = sum(1 for keyword in pattern_info["keywords"] 
                                if keyword in content_lower)
            if keyword_matches > 0:
                score += (keyword_matches / len(pattern_info["keywords"])) * pattern_info["importance_weight"] * 0.2
        
        # Length factor (longer content might be more detailed)
        if len(content) > 500:
            score += 0.1
        elif len(content) > 200:
            score += 0.05
        
        # Technical depth indicators
        technical_indicators = ["algorithm", "method", "analysis", "result", "conclusion", 
                              "significant", "correlation", "hypothesis", "evidence"]
        technical_matches = sum(1 for indicator in technical_indicators if indicator in content_lower)
        score += min(technical_matches * 0.05, 0.2)
        
        return min(score, 1.0)
    
    async def _detect_context_type(self, content: str) -> str:
        """Detectar tipo de contexto."""
        
        content_lower = content.lower()
        
        type_patterns = {
            "insight": ["insight", "realize", "understand", "discover", "breakthrough"],
            "definition": ["define", "definition", "is", "means", "refers to"],
            "method": ["method", "approach", "technique", "procedure", "protocol"],
            "result": ["result", "outcome", "finding", "data", "measurement"],
            "question": ["?", "question", "wonder", "unclear", "how", "why", "what"]
        }
        
        for context_type, patterns in type_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                return context_type
        
        return "general"
    
    async def _extract_keywords(self, content: str) -> List[str]:
        """Extrair palavras-chave do conteúdo."""
        
        # Simple keyword extraction (could be enhanced with NLP)
        words = content.lower().split()
        
        # Filter out common words and extract meaningful terms
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [word.strip(".,!?;:()[]{}") for word in words 
                   if len(word) > 3 and word not in stop_words]
        
        # Return top keywords by frequency (simple approach)
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [kw for kw, count in keyword_counts.most_common(10)]
    
    async def _extract_concepts(self, content: str) -> List[str]:
        """Extrair conceitos relacionados do conteúdo."""
        
        # Domain-specific concept patterns
        concept_patterns = {
            "kec_metrics": ["h_spectral", "h_forman", "entropy", "percolation", "topology"],
            "biomaterials": ["scaffold", "biocompatible", "tissue", "bone", "cartilage"],
            "consciousness": ["awareness", "qualia", "subjective", "phenomenology", "mind"],
            "mathematics": ["proof", "theorem", "algorithm", "optimization", "analysis"]
        }
        
        found_concepts = []
        content_lower = content.lower()
        
        for concept_group, concepts in concept_patterns.items():
            for concept in concepts:
                if concept in content_lower:
                    found_concepts.append(concept)
        
        return found_concepts
    
    def _infer_domain_from_context(self, content: str) -> Optional[ScientificDomain]:
        """Inferir domínio científico do contexto."""
        
        content_lower = content.lower()
        
        domain_indicators = {
            ScientificDomain.KEC_ANALYSIS: ["kec", "spectral", "forman", "entropy", "topology"],
            ScientificDomain.BIOMATERIALS: ["biomaterial", "scaffold", "tissue", "biocompatible"],
            ScientificDomain.CONSCIOUSNESS: ["consciousness", "awareness", "qualia", "phenomenology"],
            ScientificDomain.MATHEMATICAL_PROOFS: ["proof", "theorem", "lemma", "mathematical"],
            ScientificDomain.LITERATURE_SEARCH: ["literature", "search", "papers", "database"],
            ScientificDomain.CODE_GENERATION: ["code", "programming", "function", "algorithm"]
        }
        
        best_domain = None
        best_score = 0
        
        for domain, indicators in domain_indicators.items():
            score = sum(1 for indicator in indicators if indicator in content_lower)
            if score > best_score:
                best_score = score
                best_domain = domain
        
        return best_domain if best_score > 0 else None
    
    def _get_domain_keywords(self, domain: ScientificDomain) -> List[str]:
        """Obter palavras-chave para domínio específico."""
        
        domain_keywords = {
            ScientificDomain.KEC_ANALYSIS: ["kec", "spectral", "forman", "entropy", "topology", "graph"],
            ScientificDomain.BIOMATERIALS: ["biomaterial", "scaffold", "tissue", "biocompatible", "polymer"],
            ScientificDomain.CONSCIOUSNESS: ["consciousness", "awareness", "qualia", "mind", "subjective"],
            ScientificDomain.MATHEMATICAL_PROOFS: ["proof", "theorem", "mathematics", "algebra", "analysis"],
            ScientificDomain.PHILOSOPHY: ["philosophy", "philosophical", "ethics", "metaphysics"],
            ScientificDomain.CODE_GENERATION: ["code", "programming", "software", "algorithm", "development"]
        }
        
        return domain_keywords.get(domain, [])
    
    async def _matches_insight_pattern(self, content: str, pattern_info: Dict[str, Any],
                                     domain: Optional[ScientificDomain]) -> bool:
        """Verificar se conteúdo corresponde a padrão de insight."""
        
        content_lower = content.lower()
        keyword_matches = sum(1 for keyword in pattern_info["keywords"] 
                            if keyword in content_lower)
        
        # Require at least one keyword match and minimum content length
        return keyword_matches > 0 and len(content) > 50
    
    async def _create_insight(self, content: str, pattern_name: str, 
                            pattern_info: Dict[str, Any], 
                            domain: Optional[ScientificDomain],
                            message: ChatMessage) -> Dict[str, Any]:
        """Criar insight estruturado."""
        
        return {
            "type": pattern_name,
            "description": pattern_info["description"],
            "content": content,
            "summary": content[:200] + "..." if len(content) > 200 else content,
            "domain": domain.value if domain else None,
            "importance": pattern_info["importance_weight"],
            "source_message": {
                "role": message.role,
                "timestamp": message.timestamp.isoformat() if message.timestamp else None
            },
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "keywords": await self._extract_keywords(content),
            "concepts": await self._extract_concepts(content)
        }
    
    async def _build_concept_network(self):
        """Construir rede de conceitos."""
        logger.info("Building concept network...")
        # This would be populated over time with actual usage
    
    async def _initialize_research_timelines(self):
        """Inicializar timelines de pesquisa."""
        logger.info("Initializing research timelines...")
        # This would load existing timelines from persistence
    
    async def _save_persistent_contexts(self):
        """Salvar contextos persistentes."""
        logger.info("Saving persistent contexts...")
        # This would save to actual persistence layer
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obter estatísticas do Context Bridge."""
        
        return {
            **self._stats,
            "total_conversations": len(self.conversation_contexts),
            "domains_tracked": len(self.domain_contexts),
            "ais_tracked": len(self.ai_contexts),
            "research_timelines": len(self.research_timelines),
            "cross_domain_connections": len(self.cross_domain_connections),
            "project_contexts": len(self.project_contexts),
            "concept_network_size": len(self.concept_network)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check do Context Bridge."""
        
        return {
            "healthy": self.enabled,
            "status": "operational" if self.enabled else "offline",
            "statistics": self.get_statistics(),
            "last_check": datetime.now(timezone.utc).isoformat()
        }


__all__ = ["ContextBridge", "ContextEntry", "ResearchTimeline", "CrossDomainConnection"]