"""
Model Router - Intelligent LLM Routing for Darwin 2025

Cost optimization through smart model selection:
- RouteLLM pattern (IBM Research)
- Cascade strategy (cheap → expensive)
- Confidence-based routing
- Performance tracking per model

Cost Reduction:
- 85% cost reduction vs always GPT-4 (IBM Research)
- 95% performance maintained
- 40% of GPT-4 cost (FrugalGPT)

References:
    - "RouteLLM: Learning to Route LLMs with Preference Data" (IBM Research, 2024)
    - "FrugalGPT: How to Use Large Language Models While Reducing Cost and
       Improving Performance" (Chen et al., 2023)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json

try:
    from langchain.schema import BaseMessage, HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model tiers by capability/cost"""
    CHEAP = "cheap"  # GPT-3.5, Llama 8B
    MEDIUM = "medium"  # GPT-4o-mini, Qwen 14B
    EXPENSIVE = "expensive"  # GPT-4, Claude Opus, Llama 70B


class RoutingStrategy(str, Enum):
    """Routing strategies"""
    LEARNED = "learned"  # ML-based router (RouteLLM)
    RULE_BASED = "rule_based"  # Heuristics
    CASCADE = "cascade"  # Try cheap first, escalate if needed
    RANDOM = "random"  # Random selection (baseline)


@dataclass
class ModelConfig:
    """Configuration for a single model"""
    name: str
    tier: ModelTier
    provider: str  # "openai", "anthropic", "local"
    
    # Cost (USD per 1K tokens)
    cost_per_1k_prompt: float
    cost_per_1k_completion: float
    
    # Performance characteristics
    avg_latency_ms: float = 1000.0
    context_window: int = 4096
    quality_score: float = 0.0  # 0-1, learned from data
    
    # Limits
    rate_limit_rpm: int = 60  # Requests per minute
    rate_limit_tpm: int = 90000  # Tokens per minute


@dataclass
class RouterConfig:
    """Configuration for model router"""
    # Strategy
    strategy: RoutingStrategy = RoutingStrategy.CASCADE
    
    # Models (in order of preference for cascade)
    models: List[ModelConfig] = field(default_factory=list)
    
    # Cascade settings
    cascade_confidence_threshold: float = 0.7  # Escalate if confidence < this
    cascade_max_attempts: int = 3  # Max escalations
    
    # Learned router settings
    router_model_path: Optional[str] = None  # Path to trained router
    
    # Performance tracking
    enable_performance_tracking: bool = True
    track_window_size: int = 1000  # Last N requests


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    selected_model: str
    tier: ModelTier
    confidence: float
    reasoning: str
    
    # Cost estimation
    estimated_cost: float
    
    # Alternatives considered
    alternatives: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class QueryCharacteristics:
    """Characteristics of a query for routing"""
    length: int  # Number of tokens
    complexity: str  # "simple", "medium", "complex"
    domain: str  # "general", "scientific", "code", etc.
    requires_reasoning: bool
    requires_creativity: bool
    requires_factual_accuracy: bool


class ModelRouter:
    """
    Intelligent LLM routing for cost optimization
    
    Strategies:
        1. Learned: ML model trained on benchmark data
        2. Rule-based: Heuristics (query length, complexity)
        3. Cascade: Try cheap, escalate if low confidence
        4. Random: Baseline for comparison
    
    Usage:
        >>> router = ModelRouter(config)
        >>> decision = router.route(query, context)
        >>> print(f"Route to: {decision.selected_model}")
        >>> print(f"Cost: ${decision.estimated_cost:.6f}")
    """
    
    def __init__(self, config: Optional[RouterConfig] = None):
        """
        Initialize model router
        
        Args:
            config: Router configuration
        """
        self.config = config or self._default_config()
        
        # Model registry
        self.models: Dict[str, ModelConfig] = {
            model.name: model for model in self.config.models
        }
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        
        # Stats
        self.stats = {
            "requests_total": 0,
            "routes_by_model": {},
            "routes_by_tier": {tier.value: 0 for tier in ModelTier},
            "total_cost": 0.0,
            "cost_saved_vs_expensive": 0.0,
            "avg_confidence": 0.0,
            "escalations": 0
        }
        
        # Initialize model stats
        for model_name in self.models:
            self.stats["routes_by_model"][model_name] = 0
    
    def _default_config(self) -> RouterConfig:
        """Create default configuration with common models"""
        return RouterConfig(
            strategy=RoutingStrategy.CASCADE,
            models=[
                # Cheap tier
                ModelConfig(
                    name="gpt-3.5-turbo",
                    tier=ModelTier.CHEAP,
                    provider="openai",
                    cost_per_1k_prompt=0.0005,
                    cost_per_1k_completion=0.0015,
                    avg_latency_ms=800,
                    context_window=16385
                ),
                ModelConfig(
                    name="llama-3.1-8b",
                    tier=ModelTier.CHEAP,
                    provider="local",
                    cost_per_1k_prompt=0.0,  # Self-hosted
                    cost_per_1k_completion=0.0,
                    avg_latency_ms=500,
                    context_window=8192
                ),
                # Medium tier
                ModelConfig(
                    name="gpt-4o-mini",
                    tier=ModelTier.MEDIUM,
                    provider="openai",
                    cost_per_1k_prompt=0.00015,
                    cost_per_1k_completion=0.0006,
                    avg_latency_ms=1000,
                    context_window=128000
                ),
                # Expensive tier
                ModelConfig(
                    name="gpt-4-turbo",
                    tier=ModelTier.EXPENSIVE,
                    provider="openai",
                    cost_per_1k_prompt=0.01,
                    cost_per_1k_completion=0.03,
                    avg_latency_ms=2000,
                    context_window=128000
                ),
                ModelConfig(
                    name="claude-3-opus",
                    tier=ModelTier.EXPENSIVE,
                    provider="anthropic",
                    cost_per_1k_prompt=0.015,
                    cost_per_1k_completion=0.075,
                    avg_latency_ms=2500,
                    context_window=200000
                )
            ]
        )
    
    def route(
        self,
        query: str,
        context: Optional[str] = None,
        characteristics: Optional[QueryCharacteristics] = None
    ) -> RoutingDecision:
        """
        Route query to appropriate model
        
        Args:
            query: User query
            context: Optional context
            characteristics: Optional pre-computed characteristics
        
        Returns:
            RoutingDecision with selected model and metadata
        """
        self.stats["requests_total"] += 1
        
        # Analyze query if characteristics not provided
        if characteristics is None:
            characteristics = self._analyze_query(query)
        
        # Route based on strategy
        if self.config.strategy == RoutingStrategy.LEARNED:
            decision = self._route_learned(query, characteristics)
        elif self.config.strategy == RoutingStrategy.RULE_BASED:
            decision = self._route_rule_based(query, characteristics)
        elif self.config.strategy == RoutingStrategy.CASCADE:
            decision = self._route_cascade(query, characteristics)
        elif self.config.strategy == RoutingStrategy.RANDOM:
            decision = self._route_random()
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
        
        # Update stats
        self.stats["routes_by_model"][decision.selected_model] += 1
        self.stats["routes_by_tier"][decision.tier.value] += 1
        self.stats["total_cost"] += decision.estimated_cost
        
        # Calculate savings vs always expensive
        expensive_models = [m for m in self.models.values() if m.tier == ModelTier.EXPENSIVE]
        if expensive_models:
            expensive_cost = self._estimate_cost(
                expensive_models[0],
                characteristics.length
            )
            saving = expensive_cost - decision.estimated_cost
            self.stats["cost_saved_vs_expensive"] += saving
        
        # Update confidence average
        if self.stats["requests_total"] == 1:
            self.stats["avg_confidence"] = decision.confidence
        else:
            self.stats["avg_confidence"] = (
                (self.stats["avg_confidence"] * (self.stats["requests_total"] - 1) +
                 decision.confidence) / self.stats["requests_total"]
            )
        
        # Track performance
        if self.config.enable_performance_tracking:
            self._track_performance(query, decision, characteristics)
        
        return decision
    
    def _analyze_query(self, query: str) -> QueryCharacteristics:
        """
        Analyze query to determine characteristics
        
        Heuristics for quick analysis (no LLM call)
        """
        # Simple token count (rough approximation)
        length = len(query.split())
        
        # Complexity heuristics
        if length < 10:
            complexity = "simple"
        elif length < 50:
            complexity = "medium"
        else:
            complexity = "complex"
        
        # Domain detection (simple keyword matching)
        query_lower = query.lower()
        if any(word in query_lower for word in ["function", "code", "algorithm", "implement"]):
            domain = "code"
        elif any(word in query_lower for word in ["study", "research", "paper", "analysis"]):
            domain = "scientific"
        else:
            domain = "general"
        
        # Capability requirements (heuristics)
        requires_reasoning = any(word in query_lower for word in ["why", "how", "explain", "analyze"])
        requires_creativity = any(word in query_lower for word in ["create", "write", "design", "imagine"])
        requires_factual_accuracy = any(word in query_lower for word in ["what", "when", "who", "where"])
        
        return QueryCharacteristics(
            length=length,
            complexity=complexity,
            domain=domain,
            requires_reasoning=requires_reasoning,
            requires_creativity=requires_creativity,
            requires_factual_accuracy=requires_factual_accuracy
        )
    
    def _route_learned(
        self,
        query: str,
        characteristics: QueryCharacteristics
    ) -> RoutingDecision:
        """
        Route using learned model (RouteLLM)
        
        In production: Load trained classifier
        Here: Fallback to rule-based
        """
        logger.warning("Learned routing not yet trained, falling back to rule-based")
        return self._route_rule_based(query, characteristics)
    
    def _route_rule_based(
        self,
        query: str,
        characteristics: QueryCharacteristics
    ) -> RoutingDecision:
        """
        Route using heuristic rules
        
        Rules:
            - Simple + general → cheap
            - Complex + reasoning → expensive
            - Medium → medium tier
            - Code/scientific → medium or expensive
        """
        score_by_model = {}
        
        for model_name, model in self.models.items():
            score = 0.0
            reasoning_parts = []
            
            # Complexity match
            if characteristics.complexity == "simple" and model.tier == ModelTier.CHEAP:
                score += 3.0
                reasoning_parts.append("simple query")
            elif characteristics.complexity == "medium" and model.tier == ModelTier.MEDIUM:
                score += 3.0
                reasoning_parts.append("medium complexity")
            elif characteristics.complexity == "complex" and model.tier == ModelTier.EXPENSIVE:
                score += 3.0
                reasoning_parts.append("complex query")
            
            # Domain match
            if characteristics.domain in ["code", "scientific"] and model.tier != ModelTier.CHEAP:
                score += 2.0
                reasoning_parts.append(f"{characteristics.domain} domain")
            
            # Capability requirements
            if characteristics.requires_reasoning and model.tier == ModelTier.EXPENSIVE:
                score += 2.0
                reasoning_parts.append("reasoning required")
            
            if characteristics.requires_creativity and model.tier == ModelTier.EXPENSIVE:
                score += 1.0
                reasoning_parts.append("creativity required")
            
            # Cost preference (slight bias toward cheaper)
            if model.tier == ModelTier.CHEAP:
                score += 0.5
            
            score_by_model[model_name] = (score, "; ".join(reasoning_parts))
        
        # Select best model
        best_model = max(score_by_model.keys(), key=lambda m: score_by_model[m][0])
        best_score, reasoning = score_by_model[best_model]
        
        # Normalize confidence (0-1)
        max_possible_score = 8.5  # Sum of all bonuses
        confidence = min(best_score / max_possible_score, 1.0)
        
        model_config = self.models[best_model]
        estimated_cost = self._estimate_cost(model_config, characteristics.length)
        
        # Alternatives
        alternatives = sorted(
            [(name, score / max_possible_score) for name, (score, _) in score_by_model.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return RoutingDecision(
            selected_model=best_model,
            tier=model_config.tier,
            confidence=confidence,
            reasoning=reasoning or "default routing",
            estimated_cost=estimated_cost,
            alternatives=alternatives
        )
    
    def _route_cascade(
        self,
        query: str,
        characteristics: QueryCharacteristics
    ) -> RoutingDecision:
        """
        Cascade routing: Try cheap first, escalate if needed
        
        FrugalGPT pattern:
            1. Query cheap model
            2. Evaluate confidence
            3. If confidence < threshold, escalate
            4. Repeat up to max_attempts
        
        Note: This is a planning decision, actual cascade
              happens in the calling code
        """
        # For routing decision, we select the initial model
        # Actual cascade with confidence checking happens during generation
        
        # Start with cheapest model
        cheap_models = [m for m in self.models.values() if m.tier == ModelTier.CHEAP]
        
        if not cheap_models:
            # No cheap models, use rule-based
            return self._route_rule_based(query, characteristics)
        
        selected_model = cheap_models[0]
        
        # Estimate if we'll need escalation (heuristic)
        escalation_probability = 0.0
        
        if characteristics.complexity == "complex":
            escalation_probability += 0.4
        if characteristics.requires_reasoning:
            escalation_probability += 0.3
        if characteristics.domain in ["code", "scientific"]:
            escalation_probability += 0.2
        
        escalation_probability = min(escalation_probability, 1.0)
        confidence = 1.0 - escalation_probability
        
        estimated_cost = self._estimate_cost(selected_model, characteristics.length)
        
        reasoning = f"Cascade: start with {selected_model.name} (escalate prob: {escalation_probability:.2f})"
        
        return RoutingDecision(
            selected_model=selected_model.name,
            tier=selected_model.tier,
            confidence=confidence,
            reasoning=reasoning,
            estimated_cost=estimated_cost,
            alternatives=[(m.name, 1.0 - escalation_probability) for m in self.models.values()]
        )
    
    def _route_random(self) -> RoutingDecision:
        """
        Random routing (baseline for benchmarking)
        """
        import random
        
        selected_model = random.choice(list(self.models.values()))
        
        return RoutingDecision(
            selected_model=selected_model.name,
            tier=selected_model.tier,
            confidence=0.5,  # Random = uncertain
            reasoning="Random selection (baseline)",
            estimated_cost=self._estimate_cost(selected_model, 100),  # Assume avg length
            alternatives=[]
        )
    
    def _estimate_cost(
        self,
        model: ModelConfig,
        prompt_length: int,
        completion_length: int = 200  # Assume average
    ) -> float:
        """
        Estimate cost for a query
        
        Args:
            model: Model configuration
            prompt_length: Length in tokens (approximate)
            completion_length: Expected completion length
        
        Returns:
            Estimated cost in USD
        """
        prompt_cost = (prompt_length / 1000) * model.cost_per_1k_prompt
        completion_cost = (completion_length / 1000) * model.cost_per_1k_completion
        return prompt_cost + completion_cost
    
    def _track_performance(
        self,
        query: str,
        decision: RoutingDecision,
        characteristics: QueryCharacteristics
    ):
        """Track routing decision for analysis"""
        entry = {
            "timestamp": time.time(),
            "query_length": characteristics.length,
            "complexity": characteristics.complexity,
            "domain": characteristics.domain,
            "selected_model": decision.selected_model,
            "tier": decision.tier.value,
            "confidence": decision.confidence,
            "estimated_cost": decision.estimated_cost
        }
        
        self.performance_history.append(entry)
        
        # Keep only last N
        if len(self.performance_history) > self.config.track_window_size:
            self.performance_history.pop(0)
    
    def evaluate_confidence(
        self,
        response: str,
        query: str
    ) -> float:
        """
        Evaluate confidence in a response (for cascade)
        
        Heuristics:
            - Length (too short = low confidence)
            - Uncertainty phrases ("I'm not sure", "maybe")
            - Completeness
        
        In production: Use trained classifier
        """
        confidence = 1.0
        
        response_lower = response.lower()
        
        # Check for uncertainty phrases
        uncertainty_phrases = [
            "i'm not sure", "i don't know", "maybe", "perhaps",
            "it's unclear", "uncertain", "hard to say"
        ]
        
        for phrase in uncertainty_phrases:
            if phrase in response_lower:
                confidence -= 0.2
        
        # Check length (very short = incomplete)
        if len(response.split()) < 10:
            confidence -= 0.3
        
        # Check if question was answered
        if "?" in query and "?" not in response:
            # Good sign - made statement not question
            confidence += 0.1
        
        return max(0.0, min(confidence, 1.0))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        total_requests = self.stats["requests_total"]
        
        if total_requests == 0:
            return {**self.stats, "avg_cost_per_request": 0.0, "cost_saved_percentage": 0.0}
        
        avg_cost = self.stats["total_cost"] / total_requests
        
        # Calculate cost saved percentage
        cost_saved_pct = 0.0
        if self.stats["total_cost"] > 0:
            total_possible = self.stats["total_cost"] + self.stats["cost_saved_vs_expensive"]
            cost_saved_pct = (self.stats["cost_saved_vs_expensive"] / total_possible) * 100
        
        return {
            **self.stats,
            "avg_cost_per_request": avg_cost,
            "cost_saved_percentage": cost_saved_pct
        }
    
    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance stats by model
        
        Returns average metrics per model from history
        """
        if not self.performance_history:
            return {}
        
        performance_by_model = {}
        
        for model_name in self.models:
            model_entries = [
                e for e in self.performance_history
                if e["selected_model"] == model_name
            ]
            
            if model_entries:
                performance_by_model[model_name] = {
                    "count": len(model_entries),
                    "avg_confidence": sum(e["confidence"] for e in model_entries) / len(model_entries),
                    "avg_cost": sum(e["estimated_cost"] for e in model_entries) / len(model_entries)
                }
        
        return performance_by_model


# Factory function
_router_instance: Optional[ModelRouter] = None

def get_model_router(config: Optional[RouterConfig] = None) -> ModelRouter:
    """
    Get model router instance (singleton)
    
    Usage:
        >>> router = get_model_router()
        >>> decision = router.route("What is AI?")
        >>> print(decision.selected_model)
    """
    global _router_instance
    
    if _router_instance is None:
        _router_instance = ModelRouter(config=config)
    
    return _router_instance


if __name__ == "__main__":
    # Example usage
    import sys
    
    def main():
        try:
            # Initialize router
            config = RouterConfig(
                strategy=RoutingStrategy.RULE_BASED
            )
            
            router = ModelRouter(config=config)
            
            # Test queries
            test_queries = [
                ("What is 2+2?", "Simple math"),
                ("Explain quantum computing in detail", "Complex explanation"),
                ("Write a Python function to sort a list", "Code generation"),
                ("Analyze the impact of climate change on biodiversity", "Scientific analysis"),
                ("Tell me a creative story about a robot", "Creative writing")
            ]
            
            print("="*60)
            print("Model Routing Tests")
            print("="*60)
            
            for query, description in test_queries:
                print(f"\n{description}:")
                print(f"Query: {query[:50]}...")
                
                decision = router.route(query)
                
                print(f"✓ Model: {decision.selected_model} ({decision.tier.value})")
                print(f"  Confidence: {decision.confidence:.2f}")
                print(f"  Cost: ${decision.estimated_cost:.6f}")
                print(f"  Reasoning: {decision.reasoning}")
                
                if decision.alternatives:
                    print(f"  Alternatives:")
                    for alt_model, alt_conf in decision.alternatives[:2]:
                        print(f"    - {alt_model}: {alt_conf:.2f}")
            
            # Stats
            print("\n" + "="*60)
            print("Routing Statistics")
            print("="*60)
            stats = router.get_stats()
            
            print(f"Total requests: {stats['requests_total']}")
            print(f"Avg cost per request: ${stats['avg_cost_per_request']:.6f}")
            print(f"Cost saved vs always expensive: {stats['cost_saved_percentage']:.1f}%")
            print(f"Avg confidence: {stats['avg_confidence']:.2f}")
            
            print("\nRoutes by tier:")
            for tier, count in stats['routes_by_tier'].items():
                print(f"  {tier}: {count}")
            
            print("\nRoutes by model:")
            for model, count in stats['routes_by_model'].items():
                if count > 0:
                    print(f"  {model}: {count}")
            
            return 0
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    sys.exit(main())

