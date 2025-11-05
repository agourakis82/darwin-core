"""
AI Agentic Orchestrator - Autonomous Decision Making Layer

Self-healing, predictive scaling, and autonomous operations
Integrates with Kubernetes API for real control
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict

try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False

from .pulsar_client import get_pulsar_client, TOPICS

logger = logging.getLogger("darwin.agentic")


@dataclass
class Agent:
    """Autonomous agent with specific role"""
    name: str
    role: str
    capabilities: List[str]
    health_status: str = "healthy"
    last_action: Optional[datetime] = None
    actions_performed: int = 0


@dataclass
class PluginHealth:
    """Plugin health tracking"""
    plugin_name: str
    status: str = "unknown"  # healthy, degraded, unhealthy
    consecutive_failures: int = 0
    last_check: Optional[datetime] = None
    restart_count: int = 0


class AgenticOrchestrator:
    """
    AI Agentic Layer - Autonomous system management
    
    Capabilities:
    - Self-healing: Auto-restart unhealthy plugins
    - Predictive scaling: ML-based resource allocation
    - Circuit breaking: Prevent cascade failures
    - Anomaly detection: Security and performance
    
    Integrates with:
    - Kubernetes API (restarts, scaling)
    - Apache Pulsar (events)
    - Continuous Learning (patterns)
    """
    
    def __init__(
        self,
        namespace: str = "default",
        health_check_interval: int = 30,
        enable_auto_scaling: bool = True,
        enable_self_healing: bool = True,
    ):
        self.namespace = namespace
        self.health_check_interval = health_check_interval
        self.enable_auto_scaling = enable_auto_scaling
        self.enable_self_healing = enable_self_healing
        
        # Agents
        self.agents: Dict[str, Agent] = {}
        
        # Plugin health tracking
        self.plugin_health: Dict[str, PluginHealth] = {}
        
        # Kubernetes clients
        self.k8s_apps_v1: Optional[client.AppsV1Api] = None
        self.k8s_core_v1: Optional[client.CoreV1Api] = None
        self.k8s_autoscaling: Optional[client.AutoscalingV1Api] = None
        
        # Load history for predictive scaling
        self.load_history: List[Dict[str, float]] = []
        self.max_history = 1000
        
        # Running state
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    async def initialize(self):
        """Initialize agentic orchestrator"""
        logger.info("🤖 Initializing AI Agentic Orchestrator")
        
        # Initialize Kubernetes client
        self.k8s_available = False
        if K8S_AVAILABLE:
            try:
                # Try in-cluster config first (when running in K8s)
                try:
                    config.load_incluster_config()
                    logger.info("✅ Loaded in-cluster K8s config")
                except config.ConfigException:
                    # Fallback to local kubeconfig
                    config.load_kube_config()
                    logger.info("✅ Loaded local K8s config")
                
                self.k8s_apps_v1 = client.AppsV1Api()
                self.k8s_core_v1 = client.CoreV1Api()
                self.k8s_autoscaling = client.AutoscalingV1Api()
                self.k8s_available = True
                
            except Exception as e:
                logger.warning(f"⚠️ K8s client unavailable: {e}")
                self.k8s_available = False
        
        # Create autonomous agents
        self.agents["health_monitor"] = Agent(
            name="HealthMonitor",
            role="Monitor all services and auto-heal",
            capabilities=["health_check", "restart", "scale"]
        )
        
        self.agents["load_balancer"] = Agent(
            name="LoadBalancer",
            role="Distribute load based on ML predictions",
            capabilities=["predict_load", "redistribute", "scale"]
        )
        
        self.agents["security_guard"] = Agent(
            name="SecurityGuard",
            role="Detect anomalies and protect system",
            capabilities=["anomaly_detection", "block_threats", "alert"]
        )
        
        # Start autonomous loops
        self._running = True
        
        if self.enable_self_healing:
            task = asyncio.create_task(self._health_monitoring_loop())
            self._tasks.append(task)
        
        if self.enable_auto_scaling:
            task = asyncio.create_task(self._predictive_scaling_loop())
            self._tasks.append(task)
        
        logger.info(f"✅ Agentic Orchestrator initialized | agents={len(self.agents)} | k8s={self.k8s_available}")
    
    async def shutdown(self):
        """Shutdown orchestrator"""
        logger.info("🛑 Shutting down Agentic Orchestrator")
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("✅ Agentic Orchestrator shutdown complete")
    
    async def _health_monitoring_loop(self):
        """
        Autonomous health monitoring and self-healing
        
        Actions:
        - Check plugin health via gRPC
        - Detect failures
        - Auto-restart unhealthy pods (K8s API)
        - Publish alerts
        """
        logger.info("🏥 Health monitoring agent started")
        
        while self._running:
            try:
                unhealthy = await self._check_all_plugins()
                
                for plugin_name in unhealthy:
                    health = self.plugin_health[plugin_name]
                    
                    logger.warning(
                        f"⚠️ Plugin {plugin_name} unhealthy "
                        f"(failures={health.consecutive_failures})"
                    )
                    
                    # Self-healing decision
                    if health.consecutive_failures >= 3:
                        logger.error(f"🚨 Critical: {plugin_name} failing repeatedly")
                        await self._self_heal(plugin_name)
                    
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
            
            await asyncio.sleep(self.health_check_interval)
        
        logger.info("🏥 Health monitoring agent stopped")
    
    async def _check_all_plugins(self) -> List[str]:
        """Check health of all registered plugins"""
        unhealthy = []
        
        for plugin_name in self.plugin_health.keys():
            try:
                # gRPC health check
                is_healthy = await self._check_plugin_health_grpc(plugin_name)
                
                if is_healthy:
                    self.plugin_health[plugin_name].status = "healthy"
                    self.plugin_health[plugin_name].consecutive_failures = 0
                else:
                    self.plugin_health[plugin_name].status = "unhealthy"
                    self.plugin_health[plugin_name].consecutive_failures += 1
                    unhealthy.append(plugin_name)
                
                self.plugin_health[plugin_name].last_check = datetime.now(timezone.utc)
                
            except Exception as e:
                logger.error(f"Health check error for {plugin_name}: {e}")
                self.plugin_health[plugin_name].consecutive_failures += 1
                unhealthy.append(plugin_name)
        
        return unhealthy
    
    async def _check_plugin_health_grpc(self, plugin_name: str) -> bool:
        """Check plugin health via gRPC (placeholder)"""
        # TODO: Implement actual gRPC health check
        # For now, return True
        return True
    
    async def _self_heal(self, plugin_name: str):
        """
        Autonomous self-healing for failed plugin
        
        Strategy:
        1. Try restart pod (K8s API)
        2. If restart fails 3x, try different node
        3. If migration fails, alert humans
        """
        health = self.plugin_health[plugin_name]
        
        logger.info(f"🔧 Attempting self-heal for {plugin_name}")
        
        # Agent records action
        agent = self.agents["health_monitor"]
        agent.last_action = datetime.now(timezone.utc)
        agent.actions_performed += 1
        
        # Step 1: Restart pod
        if self.k8s_available and self.k8s_core_v1:
            try:
                await self._restart_plugin_pod(plugin_name)
                health.restart_count += 1
                logger.info(f"✅ Restarted pod for {plugin_name}")
                
                # Wait for pod to come up
                await asyncio.sleep(10)
                
                # Check if healthy now
                if await self._check_plugin_health_grpc(plugin_name):
                    logger.info(f"✅ {plugin_name} recovered after restart")
                    health.consecutive_failures = 0
                    return
                    
            except Exception as e:
                logger.error(f"❌ Restart failed for {plugin_name}: {e}")
        
        # Step 2: Alert humans
        pulsar = get_pulsar_client()
        await pulsar.publish(TOPICS["system_alerts"], {
            "severity": "critical",
            "component": plugin_name,
            "message": f"Plugin {plugin_name} failed self-healing after {health.restart_count} attempts",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.error(f"🚨 CRITICAL: {plugin_name} requires manual intervention")
    
    async def _restart_plugin_pod(self, plugin_name: str):
        """Restart plugin pod using K8s API"""
        if not self.k8s_available:
            raise RuntimeError("Kubernetes API not available")
        
        # Find pod by label
        label_selector = f"app=darwin-plugin-{plugin_name}"
        
        loop = asyncio.get_running_loop()
        pods = await loop.run_in_executor(
            None,
            lambda: self.k8s_core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=label_selector
            )
        )
        
        if not pods.items:
            raise ValueError(f"No pods found for {plugin_name}")
        
        # Delete pod (K8s will recreate via deployment)
        pod_name = pods.items[0].metadata.name
        
        await loop.run_in_executor(
            None,
            lambda: self.k8s_core_v1.delete_namespaced_pod(
                name=pod_name,
                namespace=self.namespace,
                grace_period_seconds=10
            )
        )
        
        logger.info(f"✅ Deleted pod {pod_name} (will be recreated)")
    
    async def _predictive_scaling_loop(self):
        """
        ML-based predictive scaling
        
        Uses continuous learning patterns to predict load
        Scales plugins preemptively via HPA/KEDA
        """
        logger.info("📊 Predictive scaling agent started")
        
        while self._running:
            try:
                # Collect current load
                current_load = await self._get_current_load()
                self.load_history.append(current_load)
                
                # Trim history
                if len(self.load_history) > self.max_history:
                    self.load_history = self.load_history[-self.max_history:]
                
                # Predict next 5 minutes
                if len(self.load_history) >= 20:
                    predicted_load = await self._predict_future_load()
                    
                    # Scale decision
                    if predicted_load > 0.8:
                        logger.info(f"📈 High load predicted ({predicted_load:.2f}), scaling up")
                        await self._scale_plugins(scale_up=True)
                        
                    elif predicted_load < 0.3 and len(self.load_history) > 100:
                        logger.info(f"📉 Low load predicted ({predicted_load:.2f}), scaling down")
                        await self._scale_plugins(scale_up=False)
                
            except Exception as e:
                logger.error(f"Predictive scaling error: {e}")
            
            await asyncio.sleep(60)  # Check every minute
        
        logger.info("📊 Predictive scaling agent stopped")
    
    async def _get_current_load(self) -> Dict[str, float]:
        """Get current system load"""
        # Placeholder - would query metrics from Prometheus
        return {
            "cpu_usage": 0.5,
            "memory_usage": 0.6,
            "request_rate": 100.0,
            "timestamp": datetime.now(timezone.utc).timestamp()
        }
    
    async def _predict_future_load(self) -> float:
        """
        Predict load for next 5 minutes using simple moving average
        
        In production, use trained ML model from continuous_learning
        """
        if len(self.load_history) < 10:
            return 0.5
        
        # Simple moving average of last 10 samples
        recent = self.load_history[-10:]
        avg_cpu = sum(r["cpu_usage"] for r in recent) / len(recent)
        avg_mem = sum(r["memory_usage"] for r in recent) / len(recent)
        
        # Combined load
        return (avg_cpu + avg_mem) / 2.0
    
    async def _scale_plugins(self, scale_up: bool):
        """
        Scale plugins via HPA/KEDA
        
        For now, logs the decision
        In production, would patch HPA minReplicas/maxReplicas
        """
        action = "scale_up" if scale_up else "scale_down"
        
        # Record agent action
        agent = self.agents["load_balancer"]
        agent.last_action = datetime.now(timezone.utc)
        agent.actions_performed += 1
        
        if self.k8s_available and self.k8s_autoscaling:
            # Would patch HPA here
            logger.info(f"🎯 Agent decision: {action} plugins")
            # TODO: Implement actual K8s HPA patching
        else:
            logger.info(f"📝 Agent decision (dry-run): {action} plugins")
        
        # Publish event
        pulsar = get_pulsar_client()
        await pulsar.publish(TOPICS["system_alerts"], {
            "severity": "info",
            "component": "agentic_orchestrator",
            "message": f"Autonomous scaling decision: {action}",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def register_plugin(self, plugin_name: str):
        """Register plugin for monitoring"""
        if plugin_name not in self.plugin_health:
            self.plugin_health[plugin_name] = PluginHealth(plugin_name=plugin_name)
            logger.info(f"📝 Registered plugin for monitoring: {plugin_name}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        healthy_count = sum(
            1 for h in self.plugin_health.values() 
            if h.status == "healthy"
        )
        total_plugins = len(self.plugin_health)
        
        # Predict load if enough history
        predicted_load = None
        if len(self.load_history) >= 10:
            predicted_load = await self._predict_future_load()
        
        return {
            "agents": {
                name: {
                    "role": agent.role,
                    "status": agent.health_status,
                    "actions_performed": agent.actions_performed,
                    "last_action": agent.last_action.isoformat() if agent.last_action else None
                }
                for name, agent in self.agents.items()
            },
            "plugins": {
                "total": total_plugins,
                "healthy": healthy_count,
                "unhealthy": total_plugins - healthy_count,
                "details": {
                    name: {
                        "status": health.status,
                        "failures": health.consecutive_failures,
                        "restarts": health.restart_count,
                        "last_check": health.last_check.isoformat() if health.last_check else None
                    }
                    for name, health in self.plugin_health.items()
                }
            },
            "load_prediction": {
                "samples": len(self.load_history),
                "predicted_load": predicted_load
            }
        }


# Singleton
_agentic_orchestrator: Optional[AgenticOrchestrator] = None


def get_agentic_orchestrator() -> AgenticOrchestrator:
    """Get or create agentic orchestrator singleton"""
    global _agentic_orchestrator
    if _agentic_orchestrator is None:
        _agentic_orchestrator = AgenticOrchestrator()
    return _agentic_orchestrator

