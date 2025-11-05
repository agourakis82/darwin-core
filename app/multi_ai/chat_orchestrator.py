"""Chat Orchestrator - Sistema de roteamento inteligente para Multi-AI Hub."""

import re
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass

from ..core.logging import get_logger
from ..models.multi_ai_models import (
    AIProvider, AIModel, ScientificDomain, 
    ChatRequest, ChatResponse, RoutingRule, RoutingDecision,
    ModelRecommendation
)

logger = get_logger("multi_ai.orchestrator")


@dataclass
class PerformanceHistory:
    """Histórico de performance para aprendizado."""
    ai_provider: AIProvider
    model: AIModel
    domain: ScientificDomain
    avg_latency: float
    success_rate: float
    quality_score: float
    total_requests: int
    last_updated: datetime


class ChatOrchestrator:
    """Sistema de roteamento inteligente entre ChatGPT, Claude e Gemini."""
    
    def __init__(self):
        self.enabled = False
        self.performance_history: Dict[str, PerformanceHistory] = {}
        self.routing_rules = self._initialize_routing_rules()
        self.domain_keywords = self._initialize_domain_keywords()
        self._stats = {
            "total_requests": 0,
            "routing_decisions": {},
            "fallback_usage": 0,
            "avg_decision_time": 0.0
        }
        
    async def initialize(self):
        """Inicializar orchestrator."""
        logger.info("Initializing Chat Orchestrator...")
        self.enabled = True
        await self._load_performance_history()
        logger.info("✅ Chat Orchestrator initialized with intelligent routing")
    
    async def shutdown(self):
        """Shutdown orchestrator."""
        logger.info("Shutting down Chat Orchestrator...")
        await self._save_performance_history()
        self.enabled = False
        logger.info("✅ Chat Orchestrator shutdown")
    
    def _initialize_routing_rules(self) -> Dict[ScientificDomain, RoutingRule]:
        """Inicializar regras de roteamento por domínio científico."""
        return {
            # Análise matemática/algorítmica → Claude (reasoning superior)
            ScientificDomain.KEC_ANALYSIS: RoutingRule(
                domain=ScientificDomain.KEC_ANALYSIS,
                preferred_ai=AIProvider.CLAUDE,
                model=AIModel.CLAUDE_3_5_SONNET,
                confidence_threshold=0.85,
                keywords=["kec", "spectral", "forman", "entropy", "percolation", "topology"],
                fallback_ai=AIProvider.CHATGPT,
                reasoning="Claude excels at mathematical analysis and topological reasoning"
            ),
            
            ScientificDomain.MATHEMATICAL_PROOFS: RoutingRule(
                domain=ScientificDomain.MATHEMATICAL_PROOFS,
                preferred_ai=AIProvider.CLAUDE,
                model=AIModel.CLAUDE_3_5_SONNET,
                confidence_threshold=0.9,
                keywords=["proof", "theorem", "lemma", "mathematics", "algebra", "topology", "graph theory"],
                fallback_ai=AIProvider.CHATGPT,
                reasoning="Claude superior for rigorous mathematical proofs and logical reasoning"
            ),
            
            ScientificDomain.ALGORITHM_DESIGN: RoutingRule(
                domain=ScientificDomain.ALGORITHM_DESIGN,
                preferred_ai=AIProvider.CLAUDE,
                model=AIModel.CLAUDE_3_5_SONNET,
                confidence_threshold=0.8,
                keywords=["algorithm", "optimization", "complexity", "data structure", "computational"],
                fallback_ai=AIProvider.CHATGPT,
                reasoning="Claude excellent for algorithmic thinking and optimization strategies"
            ),
            
            # Biomaterials/engenharia → ChatGPT (STEM forte)
            ScientificDomain.BIOMATERIALS: RoutingRule(
                domain=ScientificDomain.BIOMATERIALS,
                preferred_ai=AIProvider.CHATGPT,
                model=AIModel.GPT_4_TURBO,
                confidence_threshold=0.85,
                keywords=["biomaterial", "biocompatible", "tissue engineering", "regenerative medicine"],
                fallback_ai=AIProvider.CLAUDE,
                reasoning="ChatGPT strong in STEM domains and materials engineering"
            ),
            
            ScientificDomain.SCAFFOLD_DESIGN: RoutingRule(
                domain=ScientificDomain.SCAFFOLD_DESIGN,
                preferred_ai=AIProvider.CHATGPT,
                model=AIModel.GPT_4_TURBO,
                confidence_threshold=0.8,
                keywords=["scaffold", "porous", "bone", "cartilage", "3d printing", "fabrication"],
                fallback_ai=AIProvider.CLAUDE,
                reasoning="ChatGPT excels at practical engineering and design problems"
            ),
            
            ScientificDomain.MATERIALS_ENGINEERING: RoutingRule(
                domain=ScientificDomain.MATERIALS_ENGINEERING,
                preferred_ai=AIProvider.CHATGPT,
                model=AIModel.GPT_4_TURBO,
                confidence_threshold=0.8,
                keywords=["material properties", "mechanical", "thermal", "characterization"],
                fallback_ai=AIProvider.CLAUDE,
                reasoning="ChatGPT strong in engineering applications and material science"
            ),
            
            # Research/discovery → Gemini (integration Google Scholar)
            ScientificDomain.LITERATURE_SEARCH: RoutingRule(
                domain=ScientificDomain.LITERATURE_SEARCH,
                preferred_ai=AIProvider.GEMINI,
                model=AIModel.GEMINI_PRO,
                confidence_threshold=0.8,
                keywords=["literature", "search", "papers", "research", "publications", "citations"],
                fallback_ai=AIProvider.CHATGPT,
                reasoning="Gemini optimized for research and Google Scholar integration"
            ),
            
            ScientificDomain.RESEARCH_SYNTHESIS: RoutingRule(
                domain=ScientificDomain.RESEARCH_SYNTHESIS,
                preferred_ai=AIProvider.GEMINI,
                model=AIModel.GEMINI_PRO,
                confidence_threshold=0.85,
                keywords=["synthesis", "review", "meta-analysis", "systematic", "state-of-art"],
                fallback_ai=AIProvider.CLAUDE,
                reasoning="Gemini excellent for synthesizing multiple research sources"
            ),
            
            ScientificDomain.ACADEMIC_WRITING: RoutingRule(
                domain=ScientificDomain.ACADEMIC_WRITING,
                preferred_ai=AIProvider.GEMINI,
                model=AIModel.GEMINI_PRO,
                confidence_threshold=0.8,
                keywords=["academic", "paper", "manuscript", "journal", "publication", "writing"],
                fallback_ai=AIProvider.CLAUDE,
                reasoning="Gemini strong in academic writing and citation management"
            ),
            
            # Philosophy/consciousness → Claude (deep reasoning)
            ScientificDomain.PHILOSOPHY: RoutingRule(
                domain=ScientificDomain.PHILOSOPHY,
                preferred_ai=AIProvider.CLAUDE,
                model=AIModel.CLAUDE_3_5_SONNET,
                confidence_threshold=0.9,
                keywords=["philosophy", "philosophical", "metaphysics", "epistemology", "ontology"],
                fallback_ai=AIProvider.GEMINI,
                reasoning="Claude superior for philosophical reasoning and complex argumentation"
            ),
            
            ScientificDomain.CONSCIOUSNESS: RoutingRule(
                domain=ScientificDomain.CONSCIOUSNESS,
                preferred_ai=AIProvider.CLAUDE,
                model=AIModel.CLAUDE_3_5_SONNET,
                confidence_threshold=0.9,
                keywords=["consciousness", "awareness", "phenomenology", "qualia", "subjective experience"],
                fallback_ai=AIProvider.GEMINI,
                reasoning="Claude excels at consciousness studies and phenomenological analysis"
            ),
            
            ScientificDomain.ETHICS: RoutingRule(
                domain=ScientificDomain.ETHICS,
                preferred_ai=AIProvider.CLAUDE,
                model=AIModel.CLAUDE_3_5_SONNET,
                confidence_threshold=0.85,
                keywords=["ethics", "moral", "ethical", "bioethics", "responsible", "values"],
                fallback_ai=AIProvider.GEMINI,
                reasoning="Claude strong in ethical reasoning and moral philosophy"
            ),
            
            # Code/implementation → ChatGPT (code quality)
            ScientificDomain.CODE_GENERATION: RoutingRule(
                domain=ScientificDomain.CODE_GENERATION,
                preferred_ai=AIProvider.CHATGPT,
                model=AIModel.GPT_4_TURBO,
                confidence_threshold=0.85,
                keywords=["code", "programming", "implementation", "function", "class", "python", "javascript"],
                fallback_ai=AIProvider.CLAUDE,
                reasoning="ChatGPT excellent code generation and programming assistance"
            ),
            
            ScientificDomain.DEBUGGING: RoutingRule(
                domain=ScientificDomain.DEBUGGING,
                preferred_ai=AIProvider.CHATGPT,
                model=AIModel.GPT_4_TURBO,
                confidence_threshold=0.8,
                keywords=["bug", "error", "debug", "fix", "troubleshoot", "exception"],
                fallback_ai=AIProvider.CLAUDE,
                reasoning="ChatGPT strong at debugging and error resolution"
            ),
            
            ScientificDomain.ARCHITECTURE: RoutingRule(
                domain=ScientificDomain.ARCHITECTURE,
                preferred_ai=AIProvider.CHATGPT,
                model=AIModel.GPT_4_TURBO,
                confidence_threshold=0.8,
                keywords=["architecture", "system design", "scalability", "patterns", "microservices"],
                fallback_ai=AIProvider.CLAUDE,
                reasoning="ChatGPT excellent for software architecture and system design"
            )
        }
    
    def _initialize_domain_keywords(self) -> Dict[ScientificDomain, List[str]]:
        """Inicializar keywords expandidas para detecção de domínio."""
        return {
            ScientificDomain.KEC_ANALYSIS: [
                "kec", "spectral", "forman", "entropy", "percolation", "topology",
                "h_spectral", "h_forman", "small-world", "clustering", "betweenness",
                "network analysis", "graph metrics", "connectivity", "centrality"
            ],
            
            ScientificDomain.MATHEMATICAL_PROOFS: [
                "proof", "prove", "theorem", "lemma", "proposition", "corollary",
                "mathematical", "algebra", "topology", "graph theory", "discrete math",
                "combinatorics", "number theory", "analysis", "geometry", "logic"
            ],
            
            ScientificDomain.BIOMATERIALS: [
                "biomaterial", "biocompatible", "tissue engineering", "regenerative medicine",
                "scaffold", "hydrogel", "polymer", "ceramic", "composite", "biodegradable",
                "cell adhesion", "bioactivity", "osseointegration", "cytotoxicity"
            ],
            
            ScientificDomain.SCAFFOLD_DESIGN: [
                "scaffold", "porous", "porosity", "pore size", "interconnectivity",
                "3d printing", "fabrication", "bone", "cartilage", "tissue",
                "mechanical properties", "degradation", "biocompatibility"
            ],
            
            ScientificDomain.LITERATURE_SEARCH: [
                "literature", "papers", "research", "publications", "citations",
                "pubmed", "google scholar", "database", "search", "systematic review",
                "meta-analysis", "bibliography", "references"
            ],
            
            ScientificDomain.CONSCIOUSNESS: [
                "consciousness", "awareness", "phenomenology", "qualia", "subjective",
                "experience", "mind", "cognitive", "neural", "brain", "perception",
                "attention", "self-awareness", "integrated information"
            ],
            
            ScientificDomain.CODE_GENERATION: [
                "code", "programming", "implementation", "function", "class", "method",
                "python", "javascript", "java", "c++", "algorithm", "data structure",
                "api", "framework", "library", "module"
            ]
        }
    
    async def route_request(self, request: ChatRequest) -> RoutingDecision:
        """Rotear request para IA mais apropriada."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # 1. Domain override pelo usuário
            if request.preferred_ai:
                logger.info(f"Using user-preferred AI: {request.preferred_ai}")
                model = self._get_best_model_for_ai(request.preferred_ai, request.domain)
                return RoutingDecision(
                    selected_ai=request.preferred_ai,
                    selected_model=model,
                    confidence=1.0,
                    reasoning="User-specified AI preference",
                    detected_domain=request.domain,
                    fallback_used=False,
                    processing_time_ms=self._calculate_processing_time(start_time)
                )
            
            # 2. Domain-based routing
            if request.domain:
                rule = self.routing_rules.get(request.domain)
                if rule:
                    logger.info(f"Domain-based routing: {request.domain} → {rule.preferred_ai}")
                    return RoutingDecision(
                        selected_ai=rule.preferred_ai,
                        selected_model=rule.model,
                        confidence=rule.confidence_threshold,
                        reasoning=f"Domain-based: {rule.reasoning}",
                        detected_domain=request.domain,
                        fallback_used=False,
                        processing_time_ms=self._calculate_processing_time(start_time)
                    )
            
            # 3. Intelligent domain detection
            detected_domain, confidence = await self._detect_domain(request.message)
            
            if detected_domain and confidence >= 0.7:
                rule = self.routing_rules.get(detected_domain)
                if rule and confidence >= rule.confidence_threshold:
                    logger.info(f"Auto-detected domain: {detected_domain} → {rule.preferred_ai}")
                    return RoutingDecision(
                        selected_ai=rule.preferred_ai,
                        selected_model=rule.model,
                        confidence=confidence,
                        reasoning=f"Auto-detected {detected_domain}: {rule.reasoning}",
                        detected_domain=detected_domain,
                        fallback_used=False,
                        processing_time_ms=self._calculate_processing_time(start_time)
                    )
            
            # 4. Performance-based selection
            best_ai = await self._select_based_on_performance(detected_domain)
            model = self._get_best_model_for_ai(best_ai, detected_domain)
            
            return RoutingDecision(
                selected_ai=best_ai,
                selected_model=model,
                confidence=0.6,  # Lower confidence for performance-based
                reasoning="Selected based on historical performance",
                detected_domain=detected_domain,
                fallback_used=True,
                processing_time_ms=self._calculate_processing_time(start_time)
            )
            
        except Exception as e:
            logger.error(f"Error in routing decision: {e}")
            # Fallback to ChatGPT as default
            return RoutingDecision(
                selected_ai=AIProvider.CHATGPT,
                selected_model=AIModel.GPT_4_TURBO,
                confidence=0.3,
                reasoning=f"Emergency fallback due to error: {str(e)}",
                detected_domain=None,
                fallback_used=True,
                processing_time_ms=self._calculate_processing_time(start_time)
            )
        finally:
            self._stats["total_requests"] += 1
    
    async def _detect_domain(self, message: str) -> Tuple[Optional[ScientificDomain], float]:
        """Detectar domínio científico da mensagem."""
        message_lower = message.lower()
        
        domain_scores = {}
        
        # Calcular scores para cada domínio
        for domain, keywords in self.domain_keywords.items():
            score = 0.0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword.lower() in message_lower:
                    # Weight based on keyword importance and position
                    weight = 1.0
                    if keyword in ["kec", "scaffold", "consciousness", "proof"]:
                        weight = 2.0  # High-importance keywords
                    
                    score += weight
                    matched_keywords.append(keyword)
            
            if matched_keywords:
                # Normalize by message length and keyword count
                normalized_score = score / (len(message.split()) * 0.1 + len(keywords) * 0.1)
                domain_scores[domain] = {
                    'score': normalized_score,
                    'matched_keywords': matched_keywords
                }
        
        if not domain_scores:
            return None, 0.0
        
        # Select domain with highest score
        best_domain = max(domain_scores.keys(), key=lambda d: domain_scores[d]['score'])
        best_score = domain_scores[best_domain]['score']
        
        # Convert to confidence (0-1)
        confidence = min(best_score * 0.3, 1.0)
        
        logger.info(f"Domain detection: '{message[:50]}...' → {best_domain} (confidence: {confidence:.2f})")
        
        return best_domain, confidence
    
    async def _select_based_on_performance(self, domain: Optional[ScientificDomain]) -> AIProvider:
        """Selecionar IA baseado em performance histórica."""
        
        # Default priorities if no performance history
        default_priorities = [AIProvider.CHATGPT, AIProvider.CLAUDE, AIProvider.GEMINI]
        
        if not self.performance_history:
            return default_priorities[0]
        
        # Calculate scores for each AI
        ai_scores = {}
        for ai in AIProvider:
            key = f"{ai}_{domain or 'general'}"
            if key in self.performance_history:
                history = self.performance_history[key]
                # Score = (success_rate * quality_score) / latency_factor
                latency_factor = max(history.avg_latency / 1000, 0.1)  # Convert to seconds
                score = (history.success_rate * history.quality_score) / latency_factor
                ai_scores[ai] = score
            else:
                # Default score for new AIs
                ai_scores[ai] = 0.5
        
        best_ai = max(ai_scores.keys(), key=lambda ai: ai_scores[ai])
        logger.info(f"Performance-based selection: {best_ai} (score: {ai_scores[best_ai]:.2f})")
        
        return best_ai
    
    def _get_best_model_for_ai(self, ai_provider: AIProvider, domain: Optional[ScientificDomain]) -> AIModel:
        """Obter melhor modelo para IA específica."""
        
        model_map = {
            AIProvider.CHATGPT: {
                'default': AIModel.GPT_4_TURBO,
                'alternatives': [AIModel.GPT_4, AIModel.GPT_3_5_TURBO]
            },
            AIProvider.CLAUDE: {
                'default': AIModel.CLAUDE_3_5_SONNET,
                'alternatives': [AIModel.CLAUDE_3_SONNET, AIModel.CLAUDE_3_HAIKU]
            },
            AIProvider.GEMINI: {
                'default': AIModel.GEMINI_PRO,
                'alternatives': [AIModel.GEMINI_PRO_VISION]
            }
        }
        
        # Domain-specific model selection
        if domain in [ScientificDomain.MATHEMATICAL_PROOFS, ScientificDomain.PHILOSOPHY, ScientificDomain.CONSCIOUSNESS]:
            if ai_provider == AIProvider.CLAUDE:
                return AIModel.CLAUDE_3_5_SONNET  # Best for reasoning
        
        return model_map[ai_provider]['default']
    
    async def get_model_recommendation(self, message: str, context: Optional[Dict[str, Any]] = None) -> ModelRecommendation:
        """Obter recomendação de modelo/IA."""
        
        # Detect domain and route
        detected_domain, confidence = await self._detect_domain(message)
        
        request = ChatRequest(message=message, domain=detected_domain, context=context)
        routing_decision = await self.route_request(request)
        
        # Alternative options
        alternatives = []
        for ai in AIProvider:
            if ai != routing_decision.selected_ai:
                model = self._get_best_model_for_ai(ai, detected_domain)
                alternatives.append({
                    "ai_provider": ai,
                    "model": model,
                    "estimated_quality": self._estimate_quality(ai, detected_domain),
                    "estimated_cost": self._estimate_cost(ai, model),
                    "reasoning": f"Alternative option with focus on {ai.value} strengths"
                })
        
        return ModelRecommendation(
            recommended_ai=routing_decision.selected_ai,
            recommended_model=routing_decision.selected_model,
            confidence=routing_decision.confidence,
            reasoning=routing_decision.reasoning,
            domain=detected_domain,
            alternative_options=alternatives
        )
    
    async def update_performance(self, ai_provider: AIProvider, model: AIModel, 
                               domain: Optional[ScientificDomain], 
                               latency: float, success: bool, quality_score: Optional[float] = None):
        """Atualizar histórico de performance."""
        key = f"{ai_provider}_{domain or 'general'}"
        
        if key not in self.performance_history:
            self.performance_history[key] = PerformanceHistory(
                ai_provider=ai_provider,
                model=model,
                domain=domain or ScientificDomain.INTERDISCIPLINARY,
                avg_latency=latency,
                success_rate=1.0 if success else 0.0,
                quality_score=quality_score or 5.0,
                total_requests=1,
                last_updated=datetime.now(timezone.utc)
            )
        else:
            history = self.performance_history[key]
            # Update running averages
            total = history.total_requests
            history.avg_latency = (history.avg_latency * total + latency) / (total + 1)
            history.success_rate = (history.success_rate * total + (1.0 if success else 0.0)) / (total + 1)
            if quality_score:
                history.quality_score = (history.quality_score * total + quality_score) / (total + 1)
            history.total_requests += 1
            history.last_updated = datetime.now(timezone.utc)
        
        # Log performance update
        logger.info(f"Performance updated: {key} - latency: {latency}ms, success: {success}")
    
    def _calculate_processing_time(self, start_time: datetime) -> float:
        """Calcular tempo de processamento em ms."""
        return (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
    
    def _estimate_quality(self, ai_provider: AIProvider, domain: Optional[ScientificDomain]) -> float:
        """Estimar qualidade baseado no domínio."""
        # Domain-specific quality estimates (0-10 scale)
        quality_matrix = {
            AIProvider.CLAUDE: {
                ScientificDomain.MATHEMATICAL_PROOFS: 9.5,
                ScientificDomain.PHILOSOPHY: 9.5,
                ScientificDomain.CONSCIOUSNESS: 9.5,
                ScientificDomain.KEC_ANALYSIS: 9.0,
                'default': 8.0
            },
            AIProvider.CHATGPT: {
                ScientificDomain.BIOMATERIALS: 9.0,
                ScientificDomain.CODE_GENERATION: 9.5,
                ScientificDomain.SCAFFOLD_DESIGN: 9.0,
                ScientificDomain.DEBUGGING: 9.0,
                'default': 8.5
            },
            AIProvider.GEMINI: {
                ScientificDomain.LITERATURE_SEARCH: 9.0,
                ScientificDomain.RESEARCH_SYNTHESIS: 9.0,
                ScientificDomain.ACADEMIC_WRITING: 8.5,
                'default': 8.0
            }
        }
        
        ai_scores = quality_matrix.get(ai_provider, {})
        return ai_scores.get(domain, ai_scores.get('default', 7.0))
    
    def _estimate_cost(self, ai_provider: AIProvider, model: AIModel) -> float:
        """Estimar custo por request (USD)."""
        # Rough cost estimates per 1K tokens (input + output)
        cost_matrix = {
            AIModel.GPT_4_TURBO: 0.01,
            AIModel.GPT_4: 0.03,
            AIModel.GPT_3_5_TURBO: 0.002,
            AIModel.CLAUDE_3_5_SONNET: 0.003,
            AIModel.CLAUDE_3_SONNET: 0.003,
            AIModel.CLAUDE_3_HAIKU: 0.0005,
            AIModel.GEMINI_PRO: 0.00075,
            AIModel.GEMINI_PRO_VISION: 0.00075
        }
        
        return cost_matrix.get(model, 0.005)
    
    async def _load_performance_history(self):
        """Carregar histórico de performance (placeholder)."""
        # TODO: Implementar persistência real
        logger.info("Performance history loaded from memory")
    
    async def _save_performance_history(self):
        """Salvar histórico de performance (placeholder)."""
        # TODO: Implementar persistência real
        logger.info("Performance history saved to memory")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Obter estatísticas de roteamento."""
        return {
            **self._stats,
            "total_performance_records": len(self.performance_history),
            "routing_rules_count": len(self.routing_rules),
            "domain_keywords_count": sum(len(kw) for kw in self.domain_keywords.values())
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check do orchestrator."""
        return {
            "healthy": self.enabled,
            "status": "operational" if self.enabled else "offline",
            "routing_rules": len(self.routing_rules),
            "performance_records": len(self.performance_history),
            "total_requests_routed": self._stats["total_requests"],
            "last_check": datetime.now(timezone.utc).isoformat()
        }


__all__ = ["ChatOrchestrator", "PerformanceHistory"]