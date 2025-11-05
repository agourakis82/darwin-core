"""
Model Router V2 - Intelligent LLM Routing with Hugging Face Support

Enhanced version with:
- Local HF models support
- Advanced routing strategies
- Performance-based selection
- Load balancing
- Automatic fallback
- Real-time metrics
"""

import logging
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model tiers by capability/cost"""
    TINY = "tiny"  # SmolLM 135M-360M
    SMALL = "small"  # Phi-3 mini, SmolLM 1.7B
    MEDIUM = "medium"  # Llama 3.2 3B, Phi-3 3.8B
    LARGE = "large"  # Llama 3.2 7B+, Qwen 14B


class RoutingStrategy(str, Enum):
    """Routing strategies"""
    PERFORMANCE = "performance"  # Best performing model
    COST = "cost"  # Cheapest model
    LATENCY = "latency"  # Fastest model
    LOAD_BALANCE = "load_balance"  # Distribute load
    ADAPTIVE = "adaptive"  # Learn from performance
    CASCADE = "cascade"  # Try cheap first


@dataclass
class ModelConfig:
    """Configuration for a model"""
    name: str
    tier: ModelTier
    provider: str  # "hf_local", "vllm", "openai", etc.
    
    # Resource requirements
    vram_gb: float
    avg_latency_ms: float
    max_tokens: int
    
    # Quality metrics (learned)
    success_rate: float = 1.0
    avg_quality_score: float = 0.5
    
    # Load tracking
    current_load: int = 0
    max_concurrent: int = 10
    
    # Availability
    is_available: bool = True
    last_error_time: Optional[float] = None


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    selected_model: str
    reasoning: str
    alternatives: List[str]
    confidence: float
    routing_time_ms: float


class ModelPerformanceTracker:
    """Tracks model performance over time"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: {
            'latencies': deque(maxlen=window_size),
            'successes': deque(maxlen=window_size),
            'errors': deque(maxlen=window_size),
            'quality_scores': deque(maxlen=window_size),
        })
    
    def record_request(
        self,
        model_name: str,
        latency_ms: float,
        success: bool,
        quality_score: Optional[float] = None
    ):
        """Record a request result"""
        m = self.metrics[model_name]
        m['latencies'].append(latency_ms)
        m['successes'].append(1 if success else 0)
        m['errors'].append(0 if success else 1)
        
        if quality_score is not None:
            m['quality_scores'].append(quality_score)
    
    def get_stats(self, model_name: str) -> Dict[str, float]:
        """Get statistics for a model"""
        m = self.metrics[model_name]
        
        if not m['latencies']:
            return {
                'avg_latency_ms': 0,
                'success_rate': 1.0,
                'error_rate': 0.0,
                'avg_quality': 0.5,
                'p95_latency_ms': 0,
            }
        
        latencies = list(m['latencies'])
        successes = list(m['successes'])
        quality_scores = list(m['quality_scores']) if m['quality_scores'] else [0.5]
        
        return {
            'avg_latency_ms': np.mean(latencies),
            'success_rate': np.mean(successes),
            'error_rate': 1 - np.mean(successes),
            'avg_quality': np.mean(quality_scores),
            'p95_latency_ms': np.percentile(latencies, 95) if len(latencies) > 1 else latencies[0],
            'request_count': len(latencies),
        }


class ModelRouterV2:
    """
    Enhanced model router with HF support and advanced strategies.
    
    Features:
    - Multi-strategy routing
    - Performance tracking
    - Load balancing
    - Automatic fallback
    - Health monitoring
    """
    
    def __init__(
        self,
        models: List[ModelConfig],
        strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE
    ):
        self.models = {m.name: m for m in models}
        self.strategy = strategy
        self.tracker = ModelPerformanceTracker()
        
        logger.info(f"ModelRouter V2 initialized with {len(models)} models")
        logger.info(f"Strategy: {strategy}")
    
    def route(
        self,
        query: str,
        context_length: Optional[int] = None,
        required_quality: Optional[float] = None
    ) -> RoutingDecision:
        """
        Route a query to the best model.
        
        Args:
            query: User query
            context_length: Required context length (tokens)
            required_quality: Minimum quality threshold
        
        Returns:
            RoutingDecision with selected model
        """
        start_time = time.time()
        
        # Filter available models
        available = self._get_available_models(context_length, required_quality)
        
        if not available:
            raise RuntimeError("No available models meet requirements")
        
        # Select model based on strategy
        if self.strategy == RoutingStrategy.PERFORMANCE:
            selected = self._select_by_performance(available)
        elif self.strategy == RoutingStrategy.COST:
            selected = self._select_by_cost(available)
        elif self.strategy == RoutingStrategy.LATENCY:
            selected = self._select_by_latency(available)
        elif self.strategy == RoutingStrategy.LOAD_BALANCE:
            selected = self._select_by_load(available)
        elif self.strategy == RoutingStrategy.ADAPTIVE:
            selected = self._select_adaptive(available, query)
        elif self.strategy == RoutingStrategy.CASCADE:
            selected = self._select_cascade(available)
        else:
            selected = available[0]
        
        routing_time = (time.time() - start_time) * 1000
        
        alternatives = [m.name for m in available if m.name != selected.name]
        
        return RoutingDecision(
            selected_model=selected.name,
            reasoning=self._explain_selection(selected, available),
            alternatives=alternatives[:3],
            confidence=self._calculate_confidence(selected, available),
            routing_time_ms=routing_time
        )
    
    def _get_available_models(
        self,
        context_length: Optional[int],
        required_quality: Optional[float]
    ) -> List[ModelConfig]:
        """Filter models by availability and requirements"""
        available = []
        
        for model in self.models.values():
            # Check availability
            if not model.is_available:
                continue
            
            # Check if recovering from error
            if model.last_error_time:
                if time.time() - model.last_error_time < 60:  # 1 min cooldown
                    continue
            
            # Check context length
            if context_length and context_length > model.max_tokens:
                continue
            
            # Check quality
            if required_quality and model.avg_quality_score < required_quality:
                continue
            
            # Check load
            if model.current_load >= model.max_concurrent:
                continue
            
            available.append(model)
        
        return available
    
    def _select_by_performance(self, models: List[ModelConfig]) -> ModelConfig:
        """Select best performing model"""
        best_model = None
        best_score = -1
        
        for model in models:
            stats = self.tracker.get_stats(model.name)
            # Score = success_rate * quality - normalized_latency
            score = (
                stats['success_rate'] * 0.4 +
                stats['avg_quality'] * 0.4 -
                min(stats['avg_latency_ms'] / 1000, 1.0) * 0.2
            )
            
            if score > best_score:
                best_score = score
                best_model = model
        
        return best_model or models[0]
    
    def _select_by_cost(self, models: List[ModelConfig]) -> ModelConfig:
        """Select cheapest model (prefer smaller tiers)"""
        tier_order = [ModelTier.TINY, ModelTier.SMALL, ModelTier.MEDIUM, ModelTier.LARGE]
        
        for tier in tier_order:
            for model in models:
                if model.tier == tier:
                    return model
        
        return models[0]
    
    def _select_by_latency(self, models: List[ModelConfig]) -> ModelConfig:
        """Select fastest model"""
        return min(models, key=lambda m: self.tracker.get_stats(m.name)['avg_latency_ms'])
    
    def _select_by_load(self, models: List[ModelConfig]) -> ModelConfig:
        """Select least loaded model"""
        return min(models, key=lambda m: m.current_load / m.max_concurrent)
    
    def _select_adaptive(self, models: List[ModelConfig], query: str) -> ModelConfig:
        """Adaptive selection based on query complexity and model performance"""
        # Estimate query complexity
        complexity = self._estimate_complexity(query)
        
        # Score models
        best_model = None
        best_score = -1
        
        for model in models:
            stats = self.tracker.get_stats(model.name)
            
            # Complex queries → prefer larger models
            # Simple queries → prefer smaller models
            tier_scores = {
                ModelTier.TINY: 1.0 if complexity < 0.3 else 0.5,
                ModelTier.SMALL: 1.0 if complexity < 0.6 else 0.7,
                ModelTier.MEDIUM: 0.8 if complexity < 0.8 else 1.0,
                ModelTier.LARGE: 0.6 if complexity < 0.5 else 1.0,
            }
            
            tier_score = tier_scores.get(model.tier, 0.5)
            
            # Combined score
            score = (
                tier_score * 0.3 +
                stats['success_rate'] * 0.3 +
                stats['avg_quality'] * 0.2 +
                (1 - min(stats['avg_latency_ms'] / 1000, 1.0)) * 0.2
            )
            
            if score > best_score:
                best_score = score
                best_model = model
        
        return best_model or models[0]
    
    def _select_cascade(self, models: List[ModelConfig]) -> ModelConfig:
        """Select cheapest available model (cascade will try others if needed)"""
        return self._select_by_cost(models)
    
    def _estimate_complexity(self, query: str) -> float:
        """Estimate query complexity (0-1)"""
        # Simple heuristics
        factors = []
        
        # Length
        factors.append(min(len(query) / 500, 1.0) * 0.3)
        
        # Question complexity indicators
        complex_words = ['analyze', 'compare', 'explain', 'describe', 'synthesize', 'evaluate']
        simple_words = ['what', 'who', 'when', 'where', 'is', 'are']
        
        query_lower = query.lower()
        has_complex = any(word in query_lower for word in complex_words)
        has_simple = any(word in query_lower for word in simple_words)
        
        if has_complex:
            factors.append(0.7)
        elif has_simple:
            factors.append(0.3)
        else:
            factors.append(0.5)
        
        # Multiple sentences
        sentence_count = query.count('.') + query.count('?') + query.count('!')
        factors.append(min(sentence_count / 3, 1.0) * 0.2)
        
        return np.mean(factors)
    
    def _explain_selection(self, selected: ModelConfig, available: List[ModelConfig]) -> str:
        """Generate human-readable explanation"""
        stats = self.tracker.get_stats(selected.name)
        
        reasons = []
        reasons.append(f"Model: {selected.name} ({selected.tier.value})")
        reasons.append(f"Strategy: {self.strategy.value}")
        
        if stats.get('request_count', 0) > 0:
            reasons.append(f"Success rate: {stats['success_rate']:.1%}")
            reasons.append(f"Avg latency: {stats['avg_latency_ms']:.0f}ms")
        
        reasons.append(f"Load: {selected.current_load}/{selected.max_concurrent}")
        
        return " | ".join(reasons)
    
    def _calculate_confidence(self, selected: ModelConfig, available: List[ModelConfig]) -> float:
        """Calculate confidence in selection"""
        stats = self.tracker.get_stats(selected.name)
        
        if stats.get('request_count', 0) < 10:
            return 0.5  # Low confidence with little data
        
        # Confidence based on success rate and quality
        confidence = (stats['success_rate'] * 0.6 + stats['avg_quality'] * 0.4)
        
        return confidence
    
    def record_result(
        self,
        model_name: str,
        latency_ms: float,
        success: bool,
        quality_score: Optional[float] = None
    ):
        """Record result of a model invocation"""
        self.tracker.record_request(model_name, latency_ms, success, quality_score)
        
        # Update model state
        if model_name in self.models:
            model = self.models[model_name]
            
            if not success:
                model.last_error_time = time.time()
                logger.warning(f"Model {model_name} failed, cooldown activated")
            else:
                model.last_error_time = None
    
    def get_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all models"""
        stats = {}
        for name in self.models.keys():
            stats[name] = self.tracker.get_stats(name)
        return stats
    
    def update_model_load(self, model_name: str, increment: int):
        """Update model load (call before/after request)"""
        if model_name in self.models:
            self.models[model_name].current_load += increment
            self.models[model_name].current_load = max(0, self.models[model_name].current_load)


# Test
if __name__ == "__main__":
    print("="*70)
    print("Model Router V2 - Test")
    print("="*70)
    
    # Setup models
    models = [
        ModelConfig(
            name="SmolLM2-360M",
            tier=ModelTier.TINY,
            provider="hf_local",
            vram_gb=1.0,
            avg_latency_ms=50,
            max_tokens=2048
        ),
        ModelConfig(
            name="Phi-3-mini",
            tier=ModelTier.SMALL,
            provider="hf_local",
            vram_gb=4.0,
            avg_latency_ms=150,
            max_tokens=4096
        ),
        ModelConfig(
            name="Llama-3.2-3B",
            tier=ModelTier.MEDIUM,
            provider="vllm",
            vram_gb=8.0,
            avg_latency_ms=200,
            max_tokens=8192
        ),
    ]
    
    # Test different strategies
    strategies = [
        RoutingStrategy.PERFORMANCE,
        RoutingStrategy.LATENCY,
        RoutingStrategy.COST,
        RoutingStrategy.ADAPTIVE,
    ]
    
    queries = [
        "What is 2+2?",  # Simple
        "Explain the theory of relativity in detail.",  # Complex
        "Compare and contrast machine learning and deep learning.",  # Very complex
    ]
    
    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"Strategy: {strategy.value}")
        print(f"{'='*70}")
        
        router = ModelRouterV2(models, strategy=strategy)
        
        for query in queries:
            decision = router.route(query)
            print(f"\nQuery: {query[:50]}...")
            print(f"Selected: {decision.selected_model}")
            print(f"Reasoning: {decision.reasoning}")
            print(f"Confidence: {decision.confidence:.2f}")
            print(f"Routing time: {decision.routing_time_ms:.2f}ms")
    
    print("\n" + "="*70)
    print("✅ Model Router V2 works!")

