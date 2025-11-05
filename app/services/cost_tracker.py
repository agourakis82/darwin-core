"""
Cost Tracker - Real-time LLM Cost Monitoring for Darwin 2025

Features:
- Per-user/project cost attribution
- Real-time monitoring and dashboards
- Budget alerts and limits
- Cost optimization recommendations
- Historical analysis and forecasting

Supports:
- OpenAI (GPT-3.5, GPT-4, embeddings)
- Anthropic (Claude models)
- Local models (hosting costs)
- Cache savings tracking
- Router savings tracking
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import time
import json
from collections import defaultdict

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    logging.warning("tiktoken not available - token counting will be approximate")

logger = logging.getLogger(__name__)


class Provider(str, Enum):
    """LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    OTHER = "other"


class ServiceType(str, Enum):
    """Types of services"""
    COMPLETION = "completion"  # Text generation
    EMBEDDING = "embedding"  # Vector embeddings
    FINE_TUNING = "fine_tuning"  # Model training
    HOSTING = "hosting"  # Infrastructure costs


@dataclass
class PricingTable:
    """Pricing for different models and services"""
    
    # OpenAI (USD per 1K tokens)
    OPENAI_PRICES = {
        "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        "gpt-4o": {"prompt": 0.005, "completion": 0.015},
        "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
        "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
        "text-embedding-3-small": {"input": 0.00002},
        "text-embedding-3-large": {"input": 0.00013},
        "text-embedding-ada-002": {"input": 0.0001}
    }
    
    # Anthropic (USD per 1K tokens)
    ANTHROPIC_PRICES = {
        "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
        "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
        "claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
        "claude-3-5-sonnet": {"prompt": 0.003, "completion": 0.015}
    }
    
    # Local hosting (USD per hour)
    LOCAL_HOSTING_COSTS = {
        "l4-single": 0.60,  # Single L4 GPU
        "l4-dual": 1.20,  # 2x L4 GPUs
        "rtx4000": 0.80,  # RTX 4000 ADA
        "a100-40gb": 2.50,  # A100 40GB
        "a100-80gb": 3.50  # A100 80GB
    }


@dataclass
class CostEvent:
    """Single cost event"""
    timestamp: float
    user_id: str
    project_id: str
    
    # Service details
    provider: Provider
    service_type: ServiceType
    model_name: str
    
    # Usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Cost (USD)
    cost: float = 0.0
    
    # Metadata
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    cached: bool = False  # Was response cached?
    routed_from: Optional[str] = None  # Original model if routed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "project_id": self.project_id,
            "provider": self.provider.value,
            "service_type": self.service_type.value,
            "model_name": self.model_name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "request_id": self.request_id,
            "endpoint": self.endpoint,
            "cached": self.cached,
            "routed_from": self.routed_from
        }


@dataclass
class Budget:
    """Budget configuration"""
    name: str
    limit: float  # USD
    period: str  # "daily", "weekly", "monthly"
    alert_threshold: float = 0.8  # Alert at 80%
    hard_limit: bool = False  # Block requests if exceeded?
    
    # Scope
    user_ids: Optional[List[str]] = None  # Specific users (None = all)
    project_ids: Optional[List[str]] = None  # Specific projects (None = all)


@dataclass
class CostTrackerConfig:
    """Configuration for cost tracker"""
    # Storage
    enable_persistent_storage: bool = False
    storage_path: str = "./data/costs"
    
    # Budgets
    budgets: List[Budget] = field(default_factory=list)
    
    # Alerts
    enable_alerts: bool = True
    alert_webhook_url: Optional[str] = None
    
    # Token counting
    default_encoding: str = "cl100k_base"  # GPT-4 encoding
    
    # Aggregation
    aggregation_window_hours: int = 24


class CostTracker:
    """
    Real-time LLM cost tracking and monitoring
    
    Features:
        - Track costs by user, project, model
        - Budget management and alerts
        - Cost optimization recommendations
        - Historical analysis and forecasting
        - Cache/router savings tracking
    
    Usage:
        >>> tracker = CostTracker()
        >>> 
        >>> # Track a completion
        >>> tracker.track_completion(
        >>>     user_id="user123",
        >>>     project_id="proj456",
        >>>     model="gpt-4-turbo",
        >>>     prompt_tokens=100,
        >>>     completion_tokens=200
        >>> )
        >>> 
        >>> # Get costs
        >>> costs = tracker.get_costs_by_user("user123")
        >>> print(f"Total: ${costs['total']:.2f}")
    """
    
    def __init__(self, config: Optional[CostTrackerConfig] = None):
        """
        Initialize cost tracker
        
        Args:
            config: Tracker configuration
        """
        self.config = config or CostTrackerConfig()
        self.pricing = PricingTable()
        
        # Event storage
        self.events: List[CostEvent] = []
        
        # Aggregated stats
        self.stats = {
            "total_cost": 0.0,
            "total_requests": 0,
            "total_tokens": 0,
            "cache_savings": 0.0,
            "router_savings": 0.0,
            "cost_by_user": defaultdict(float),
            "cost_by_project": defaultdict(float),
            "cost_by_model": defaultdict(float),
            "cost_by_provider": defaultdict(float),
            "requests_by_model": defaultdict(int),
            "tokens_by_model": defaultdict(int)
        }
        
        # Budget tracking
        self.budget_usage = {budget.name: 0.0 for budget in self.config.budgets}
        self.budget_alerts_sent = set()
        
        # Token encoder
        if HAS_TIKTOKEN:
            self.encoder = tiktoken.get_encoding(self.config.default_encoding)
        else:
            self.encoder = None
    
    def track_completion(
        self,
        user_id: str,
        project_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        provider: Provider = Provider.OPENAI,
        cached: bool = False,
        routed_from: Optional[str] = None,
        request_id: Optional[str] = None,
        endpoint: Optional[str] = None
    ) -> float:
        """
        Track a completion request
        
        Args:
            user_id: User identifier
            project_id: Project identifier
            model: Model name
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            provider: Provider (OpenAI, Anthropic, etc.)
            cached: Was response cached?
            routed_from: Original model if routed
            request_id: Optional request ID
            endpoint: Optional endpoint name
        
        Returns:
            Cost in USD
        """
        # Calculate cost
        cost = self._calculate_completion_cost(
            provider, model, prompt_tokens, completion_tokens
        )
        
        # If cached, cost is zero but we track the saving
        actual_cost = 0.0 if cached else cost
        if cached:
            self.stats["cache_savings"] += cost
        
        # If routed, track saving
        if routed_from:
            original_cost = self._calculate_completion_cost(
                provider, routed_from, prompt_tokens, completion_tokens
            )
            saving = original_cost - actual_cost
            self.stats["router_savings"] += saving
        
        # Create event
        event = CostEvent(
            timestamp=time.time(),
            user_id=user_id,
            project_id=project_id,
            provider=provider,
            service_type=ServiceType.COMPLETION,
            model_name=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=actual_cost,
            request_id=request_id,
            endpoint=endpoint,
            cached=cached,
            routed_from=routed_from
        )
        
        # Store event
        self.events.append(event)
        
        # Update stats
        self._update_stats(event)
        
        # Check budgets
        self._check_budgets(user_id, project_id, actual_cost)
        
        return actual_cost
    
    def track_embedding(
        self,
        user_id: str,
        project_id: str,
        model: str,
        tokens: int,
        provider: Provider = Provider.OPENAI,
        cached: bool = False,
        request_id: Optional[str] = None
    ) -> float:
        """
        Track an embedding request
        
        Args:
            user_id: User identifier
            project_id: Project identifier
            model: Model name
            tokens: Input tokens
            provider: Provider
            cached: Was response cached?
            request_id: Optional request ID
        
        Returns:
            Cost in USD
        """
        # Calculate cost
        cost = self._calculate_embedding_cost(provider, model, tokens)
        
        # If cached, cost is zero
        actual_cost = 0.0 if cached else cost
        if cached:
            self.stats["cache_savings"] += cost
        
        # Create event
        event = CostEvent(
            timestamp=time.time(),
            user_id=user_id,
            project_id=project_id,
            provider=provider,
            service_type=ServiceType.EMBEDDING,
            model_name=model,
            prompt_tokens=tokens,
            total_tokens=tokens,
            cost=actual_cost,
            request_id=request_id,
            cached=cached
        )
        
        # Store and update
        self.events.append(event)
        self._update_stats(event)
        self._check_budgets(user_id, project_id, actual_cost)
        
        return actual_cost
    
    def track_hosting(
        self,
        user_id: str,
        project_id: str,
        instance_type: str,
        hours: float
    ) -> float:
        """
        Track hosting costs (local models)
        
        Args:
            user_id: User identifier
            project_id: Project identifier
            instance_type: Instance type (e.g., "l4-dual")
            hours: Hours of usage
        
        Returns:
            Cost in USD
        """
        cost_per_hour = self.pricing.LOCAL_HOSTING_COSTS.get(instance_type, 1.0)
        cost = cost_per_hour * hours
        
        event = CostEvent(
            timestamp=time.time(),
            user_id=user_id,
            project_id=project_id,
            provider=Provider.LOCAL,
            service_type=ServiceType.HOSTING,
            model_name=instance_type,
            cost=cost
        )
        
        self.events.append(event)
        self._update_stats(event)
        self._check_budgets(user_id, project_id, cost)
        
        return cost
    
    def _calculate_completion_cost(
        self,
        provider: Provider,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Calculate cost for completion"""
        if provider == Provider.OPENAI:
            prices = self.pricing.OPENAI_PRICES.get(model)
            if prices:
                prompt_cost = (prompt_tokens / 1000) * prices.get("prompt", 0)
                completion_cost = (completion_tokens / 1000) * prices.get("completion", 0)
                return prompt_cost + completion_cost
        
        elif provider == Provider.ANTHROPIC:
            prices = self.pricing.ANTHROPIC_PRICES.get(model)
            if prices:
                prompt_cost = (prompt_tokens / 1000) * prices.get("prompt", 0)
                completion_cost = (completion_tokens / 1000) * prices.get("completion", 0)
                return prompt_cost + completion_cost
        
        # Unknown model/provider
        logger.warning(f"Unknown pricing for {provider.value}/{model}")
        return 0.0
    
    def _calculate_embedding_cost(
        self,
        provider: Provider,
        model: str,
        tokens: int
    ) -> float:
        """Calculate cost for embedding"""
        if provider == Provider.OPENAI:
            prices = self.pricing.OPENAI_PRICES.get(model)
            if prices:
                return (tokens / 1000) * prices.get("input", 0)
        
        logger.warning(f"Unknown pricing for {provider.value}/{model}")
        return 0.0
    
    def _update_stats(self, event: CostEvent):
        """Update aggregated statistics"""
        self.stats["total_cost"] += event.cost
        self.stats["total_requests"] += 1
        self.stats["total_tokens"] += event.total_tokens
        
        self.stats["cost_by_user"][event.user_id] += event.cost
        self.stats["cost_by_project"][event.project_id] += event.cost
        self.stats["cost_by_model"][event.model_name] += event.cost
        self.stats["cost_by_provider"][event.provider.value] += event.cost
        
        self.stats["requests_by_model"][event.model_name] += 1
        self.stats["tokens_by_model"][event.model_name] += event.total_tokens
    
    def _check_budgets(self, user_id: str, project_id: str, cost: float):
        """Check if budgets are exceeded and send alerts"""
        for budget in self.config.budgets:
            # Check if budget applies to this user/project
            if budget.user_ids and user_id not in budget.user_ids:
                continue
            if budget.project_ids and project_id not in budget.project_ids:
                continue
            
            # Get budget usage for period
            usage = self._get_budget_usage(budget)
            new_usage = usage + cost
            
            # Update
            self.budget_usage[budget.name] = new_usage
            
            # Check threshold
            usage_pct = new_usage / budget.limit if budget.limit > 0 else 0
            
            if usage_pct >= budget.alert_threshold:
                alert_key = f"{budget.name}_{int(time.time() / 86400)}"  # Daily key
                
                if alert_key not in self.budget_alerts_sent:
                    self._send_budget_alert(budget, new_usage, usage_pct)
                    self.budget_alerts_sent.add(alert_key)
            
            # Hard limit check
            if budget.hard_limit and new_usage >= budget.limit:
                raise RuntimeError(
                    f"Budget '{budget.name}' exceeded: "
                    f"${new_usage:.2f} / ${budget.limit:.2f}"
                )
    
    def _get_budget_usage(self, budget: Budget) -> float:
        """Get current usage for a budget period"""
        # Calculate period start
        now = time.time()
        
        if budget.period == "daily":
            period_start = now - 86400
        elif budget.period == "weekly":
            period_start = now - (86400 * 7)
        elif budget.period == "monthly":
            period_start = now - (86400 * 30)
        else:
            period_start = 0
        
        # Sum costs in period
        total = 0.0
        for event in self.events:
            if event.timestamp < period_start:
                continue
            
            # Check scope
            if budget.user_ids and event.user_id not in budget.user_ids:
                continue
            if budget.project_ids and event.project_id not in budget.project_ids:
                continue
            
            total += event.cost
        
        return total
    
    def _send_budget_alert(self, budget: Budget, usage: float, usage_pct: float):
        """Send budget alert"""
        message = (
            f"⚠️ Budget Alert: '{budget.name}'\n"
            f"Usage: ${usage:.2f} / ${budget.limit:.2f} ({usage_pct*100:.1f}%)\n"
            f"Period: {budget.period}"
        )
        
        logger.warning(message)
        
        # TODO: Send webhook/email
        if self.config.alert_webhook_url:
            # Send to webhook
            pass
    
    def get_costs_by_user(
        self,
        user_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get cost breakdown for a user"""
        events = [
            e for e in self.events
            if e.user_id == user_id
            and (start_time is None or e.timestamp >= start_time)
            and (end_time is None or e.timestamp <= end_time)
        ]
        
        total_cost = sum(e.cost for e in events)
        total_requests = len(events)
        total_tokens = sum(e.total_tokens for e in events)
        
        # By model
        by_model = defaultdict(float)
        for e in events:
            by_model[e.model_name] += e.cost
        
        # By project
        by_project = defaultdict(float)
        for e in events:
            by_project[e.project_id] += e.cost
        
        return {
            "user_id": user_id,
            "total_cost": total_cost,
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "avg_cost_per_request": total_cost / total_requests if total_requests > 0 else 0,
            "by_model": dict(by_model),
            "by_project": dict(by_project)
        }
    
    def get_costs_by_project(
        self,
        project_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get cost breakdown for a project"""
        events = [
            e for e in self.events
            if e.project_id == project_id
            and (start_time is None or e.timestamp >= start_time)
            and (end_time is None or e.timestamp <= end_time)
        ]
        
        total_cost = sum(e.cost for e in events)
        total_requests = len(events)
        total_tokens = sum(e.total_tokens for e in events)
        
        # By user
        by_user = defaultdict(float)
        for e in events:
            by_user[e.user_id] += e.cost
        
        # By model
        by_model = defaultdict(float)
        for e in events:
            by_model[e.model_name] += e.cost
        
        return {
            "project_id": project_id,
            "total_cost": total_cost,
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "avg_cost_per_request": total_cost / total_requests if total_requests > 0 else 0,
            "by_user": dict(by_user),
            "by_model": dict(by_model)
        }
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get cost optimization recommendations
        
        Returns list of actionable recommendations
        """
        recommendations = []
        
        # Check cache hit rate
        if self.stats["total_requests"] > 100:
            cache_rate = self.stats["cache_savings"] / (self.stats["total_cost"] + self.stats["cache_savings"])
            
            if cache_rate < 0.2:
                recommendations.append({
                    "priority": "high",
                    "category": "caching",
                    "title": "Low cache hit rate",
                    "description": f"Current cache rate: {cache_rate*100:.1f}%. Target: 40-60%.",
                    "action": "Tune similarity threshold or increase cache size",
                    "potential_saving": self.stats["total_cost"] * 0.3  # Estimate 30% saving
                })
        
        # Check expensive model usage
        expensive_models = ["gpt-4", "gpt-4-turbo", "claude-3-opus"]
        expensive_usage = sum(
            self.stats["cost_by_model"][model]
            for model in expensive_models
            if model in self.stats["cost_by_model"]
        )
        
        if expensive_usage / self.stats["total_cost"] > 0.5:
            recommendations.append({
                "priority": "high",
                "category": "routing",
                "title": "High expensive model usage",
                "description": f"{expensive_usage/self.stats['total_cost']*100:.1f}% of costs from expensive models.",
                "action": "Implement model routing to use cheaper models for simple queries",
                "potential_saving": expensive_usage * 0.7  # Estimate 70% of expensive can be routed
            })
        
        # Check router effectiveness
        if self.stats["router_savings"] > 0:
            router_effectiveness = self.stats["router_savings"] / (self.stats["total_cost"] + self.stats["router_savings"])
            
            if router_effectiveness < 0.5:
                recommendations.append({
                    "priority": "medium",
                    "category": "routing",
                    "title": "Router underperforming",
                    "description": f"Current routing saving: {router_effectiveness*100:.1f}%. Target: 80-85%.",
                    "action": "Tune routing confidence thresholds",
                    "potential_saving": self.stats["total_cost"] * 0.3
                })
        
        return recommendations
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics"""
        return {
            **self.stats,
            "total_savings": self.stats["cache_savings"] + self.stats["router_savings"],
            "savings_percentage": (
                (self.stats["cache_savings"] + self.stats["router_savings"]) /
                (self.stats["total_cost"] + self.stats["cache_savings"] + self.stats["router_savings"]) * 100
                if (self.stats["total_cost"] + self.stats["cache_savings"] + self.stats["router_savings"]) > 0
                else 0
            ),
            "avg_cost_per_request": (
                self.stats["total_cost"] / self.stats["total_requests"]
                if self.stats["total_requests"] > 0 else 0
            ),
            "top_users_by_cost": sorted(
                self.stats["cost_by_user"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "top_models_by_cost": sorted(
                self.stats["cost_by_model"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


# Factory function
_tracker_instance: Optional[CostTracker] = None

def get_cost_tracker(config: Optional[CostTrackerConfig] = None) -> CostTracker:
    """
    Get cost tracker instance (singleton)
    
    Usage:
        >>> tracker = get_cost_tracker()
        >>> tracker.track_completion(
        >>>     user_id="user123",
        >>>     project_id="proj456",
        >>>     model="gpt-4-turbo",
        >>>     prompt_tokens=100,
        >>>     completion_tokens=200
        >>> )
    """
    global _tracker_instance
    
    if _tracker_instance is None:
        _tracker_instance = CostTracker(config=config)
    
    return _tracker_instance


if __name__ == "__main__":
    # Example usage
    import sys
    
    def main():
        try:
            # Initialize tracker with budgets
            config = CostTrackerConfig(
                budgets=[
                    Budget(
                        name="daily_limit",
                        limit=10.0,  # $10/day
                        period="daily",
                        alert_threshold=0.8
                    ),
                    Budget(
                        name="user123_monthly",
                        limit=100.0,  # $100/month
                        period="monthly",
                        user_ids=["user123"]
                    )
                ]
            )
            
            tracker = CostTracker(config=config)
            
            # Simulate some requests
            print("="*60)
            print("Cost Tracking Tests")
            print("="*60)
            
            # Test 1: GPT-4 completion
            print("\n1. GPT-4 Turbo completion")
            cost1 = tracker.track_completion(
                user_id="user123",
                project_id="proj456",
                model="gpt-4-turbo",
                prompt_tokens=100,
                completion_tokens=200,
                provider=Provider.OPENAI
            )
            print(f"   Cost: ${cost1:.6f}")
            
            # Test 2: Cached response
            print("\n2. Cached GPT-4 response (zero cost)")
            cost2 = tracker.track_completion(
                user_id="user123",
                project_id="proj456",
                model="gpt-4-turbo",
                prompt_tokens=100,
                completion_tokens=200,
                provider=Provider.OPENAI,
                cached=True
            )
            print(f"   Cost: ${cost2:.6f}")
            print(f"   Saved: ${cost1:.6f}")
            
            # Test 3: Routed to cheaper model
            print("\n3. Routed from GPT-4 to GPT-3.5")
            cost3 = tracker.track_completion(
                user_id="user456",
                project_id="proj789",
                model="gpt-3.5-turbo",
                prompt_tokens=50,
                completion_tokens=100,
                provider=Provider.OPENAI,
                routed_from="gpt-4-turbo"
            )
            print(f"   Actual cost: ${cost3:.6f}")
            
            # Test 4: Embedding
            print("\n4. Embedding request")
            cost4 = tracker.track_embedding(
                user_id="user123",
                project_id="proj456",
                model="text-embedding-3-small",
                tokens=500,
                provider=Provider.OPENAI
            )
            print(f"   Cost: ${cost4:.6f}")
            
            # Get stats
            print("\n" + "="*60)
            print("Overall Statistics")
            print("="*60)
            stats = tracker.get_stats()
            
            print(f"Total cost: ${stats['total_cost']:.6f}")
            print(f"Total requests: {stats['total_requests']}")
            print(f"Total savings: ${stats['total_savings']:.6f}")
            print(f"Savings percentage: {stats['savings_percentage']:.1f}%")
            print(f"Cache savings: ${stats['cache_savings']:.6f}")
            print(f"Router savings: ${stats['router_savings']:.6f}")
            
            # User breakdown
            print("\n" + "="*60)
            print("Cost by User")
            print("="*60)
            user_costs = tracker.get_costs_by_user("user123")
            print(f"User: {user_costs['user_id']}")
            print(f"Total: ${user_costs['total_cost']:.6f}")
            print(f"Requests: {user_costs['total_requests']}")
            print(f"Avg per request: ${user_costs['avg_cost_per_request']:.6f}")
            
            # Recommendations
            print("\n" + "="*60)
            print("Optimization Recommendations")
            print("="*60)
            recommendations = tracker.get_optimization_recommendations()
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    print(f"\n{i}. [{rec['priority'].upper()}] {rec['title']}")
                    print(f"   {rec['description']}")
                    print(f"   Action: {rec['action']}")
                    print(f"   Potential saving: ${rec['potential_saving']:.2f}")
            else:
                print("No recommendations - costs are well optimized!")
            
            return 0
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    sys.exit(main())

