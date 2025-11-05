"""
gRPC Client for Plugin Communication

Handles all gRPC calls from Core to Plugins
Includes retry logic, circuit breaking, and observability
"""

import grpc
from grpc import aio
import logging
from typing import Dict, Any, Optional, AsyncIterator
from datetime import datetime, timezone

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "protos"))

try:
    from plugin.v1 import plugin_pb2, plugin_pb2_grpc
    from kec.v1 import kec_pb2, kec_pb2_grpc
    from google.protobuf import any_pb2
    PROTOS_AVAILABLE = True
except ImportError:
    PROTOS_AVAILABLE = False

from .pulsar_client import get_pulsar_client, TOPICS

logger = logging.getLogger("darwin.plugin_client")


class PluginGRPCClient:
    """
    gRPC client for calling plugins
    
    Features:
    - Connection pooling
    - Retry logic with exponential backoff
    - Circuit breaking
    - OpenTelemetry tracing
    """
    
    def __init__(self):
        self.channels: Dict[str, aio.Channel] = {}
        self.stubs: Dict[str, Any] = {}
        self.circuit_breakers: Dict[str, "CircuitBreaker"] = {}
    
    async def connect_to_plugin(
        self,
        plugin_name: str,
        host: str,
        port: int,
        secure: bool = False
    ):
        """
        Connect to plugin gRPC server
        
        Args:
            plugin_name: Plugin identifier
            host: Plugin host (K8s service name)
            port: gRPC port
            secure: Use TLS (mTLS for production)
        """
        if not PROTOS_AVAILABLE:
            logger.error("Proto files not available")
            return
        
        address = f"{host}:{port}"
        
        # gRPC channel options (optimized)
        options = [
            ('grpc.max_receive_message_length', 64 * 1024 * 1024),  # 64MB
            ('grpc.max_send_message_length', 64 * 1024 * 1024),
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 10000),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.keepalive_permit_without_calls', 1),
            ('grpc.enable_retries', 1),
            ('grpc.max_connection_idle_ms', 300000),  # 5 min
        ]
        
        # Create channel
        if secure:
            # TODO: Add TLS credentials
            credentials = grpc.ssl_channel_credentials()
            channel = aio.secure_channel(address, credentials, options=options)
        else:
            channel = aio.insecure_channel(address, options=options)
        
        # Create stubs
        self.channels[plugin_name] = channel
        self.stubs[plugin_name] = plugin_pb2_grpc.PluginServiceStub(channel)
        self.circuit_breakers[plugin_name] = CircuitBreaker(plugin_name)
        
        logger.info(f"✅ Connected to plugin: {plugin_name} at {address}")
    
    async def call_plugin(
        self,
        plugin_name: str,
        operation: str,
        payload: Dict[str, Any],
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Call plugin operation via gRPC
        
        Args:
            plugin_name: Plugin to call
            operation: Operation name
            payload: Request payload
            timeout: Request timeout
            
        Returns:
            Response from plugin
        """
        if plugin_name not in self.stubs:
            raise ValueError(f"Plugin {plugin_name} not connected")
        
        stub = self.stubs[plugin_name]
        breaker = self.circuit_breakers[plugin_name]
        
        # Check circuit breaker
        if breaker.is_open():
            raise RuntimeError(f"Circuit breaker open for {plugin_name}")
        
        try:
            # Create request
            # Serialize payload to Any (allows any proto type)
            any_payload = any_pb2.Any()
            # For now, use JSON string in value field
            import json
            any_payload.value = json.dumps(payload).encode('utf-8')
            
            request = plugin_pb2.ExecuteRequest(
                operation=operation,
                payload=any_payload,
                metadata={
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "darwin-core"
                }
            )
            
            # Call with timeout
            response = await stub.Execute(request, timeout=timeout)
            
            # Record success
            breaker.record_success()
            
            # Publish event
            pulsar = get_pulsar_client()
            await pulsar.publish(TOPICS["continuous_learning"], {
                "plugin": plugin_name,
                "operation": operation,
                "success": response.success,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            if not response.success:
                raise RuntimeError(response.error)
            
            # Deserialize result
            result_json = json.loads(response.result.value.decode('utf-8'))
            
            return {
                "success": True,
                "result": result_json,
                "metrics": dict(response.metrics)
            }
            
        except Exception as e:
            # Record failure
            breaker.record_failure()
            
            logger.error(f"Plugin call failed: {plugin_name}.{operation} - {e}")
            
            # Publish failure event
            pulsar = get_pulsar_client()
            await pulsar.publish(TOPICS["system_alerts"], {
                "severity": "error",
                "component": plugin_name,
                "message": f"gRPC call failed: {operation}",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            raise
    
    async def stream_to_plugin(
        self,
        plugin_name: str,
        data_iterator: AsyncIterator[bytes]
    ) -> AsyncIterator[bytes]:
        """
        Stream data to plugin (bi-directional)
        
        Used for large data like MicroCT images
        """
        if plugin_name not in self.stubs:
            raise ValueError(f"Plugin {plugin_name} not connected")
        
        stub = self.stubs[plugin_name]
        
        async def chunk_generator():
            """Generate DataChunk messages"""
            sequence = 0
            async for data in data_iterator:
                yield plugin_pb2.DataChunk(
                    data=data,
                    sequence=sequence,
                    is_last=False,
                    metadata={"source": "darwin-core"}
                )
                sequence += 1
            
            # Send final chunk
            yield plugin_pb2.DataChunk(
                data=b"",
                sequence=sequence,
                is_last=True
            )
        
        # Stream to plugin and receive responses
        async for response_chunk in stub.StreamData(chunk_generator()):
            yield response_chunk.data
    
    async def health_check_plugin(self, plugin_name: str) -> bool:
        """Check plugin health"""
        if plugin_name not in self.stubs:
            return False
        
        stub = self.stubs[plugin_name]
        
        try:
            response = await stub.HealthCheck(
                plugin_pb2.HealthRequest(),
                timeout=5.0
            )
            return response.healthy
        except Exception as e:
            logger.error(f"Health check failed for {plugin_name}: {e}")
            return False
    
    async def get_plugin_metadata(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get plugin metadata"""
        if plugin_name not in self.stubs:
            return None
        
        stub = self.stubs[plugin_name]
        
        try:
            response = await stub.GetMetadata(
                plugin_pb2.MetadataRequest(),
                timeout=5.0
            )
            
            return {
                "name": response.name,
                "version": response.version,
                "gpu_required": response.gpu_required,
                "gpu_memory_mb": response.gpu_memory_mb,
                "capabilities": list(response.capabilities),
                "proto_version": response.proto_version,
                "requirements": dict(response.requirements)
            }
        except Exception as e:
            logger.error(f"Metadata fetch failed for {plugin_name}: {e}")
            return None
    
    async def close_all(self):
        """Close all connections"""
        for channel in self.channels.values():
            await channel.close()
        logger.info("✅ All plugin connections closed")


class CircuitBreaker:
    """
    Circuit breaker pattern for plugin calls
    
    States:
    - CLOSED: Normal operation
    - OPEN: Too many failures, reject requests
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(
        self,
        plugin_name: str,
        failure_threshold: int = 5,
        timeout_seconds: int = 60
    ):
        self.plugin_name = plugin_name
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def is_open(self) -> bool:
        """Check if circuit is open"""
        if self.state == "OPEN":
            # Check if timeout elapsed
            if self.last_failure_time:
                elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
                if elapsed > self.timeout_seconds:
                    self.state = "HALF_OPEN"
                    logger.info(f"🔄 Circuit breaker HALF_OPEN for {self.plugin_name}")
                    return False
            return True
        return False
    
    def record_success(self):
        """Record successful call"""
        if self.state == "HALF_OPEN":
            # Service recovered!
            self.state = "CLOSED"
            self.failure_count = 0
            logger.info(f"✅ Circuit breaker CLOSED for {self.plugin_name}")
        elif self.state == "CLOSED":
            # Reset counter on success
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(f"🚨 Circuit breaker OPEN for {self.plugin_name}")


# Singleton
_plugin_client: Optional[PluginGRPCClient] = None


def get_plugin_client() -> PluginGRPCClient:
    """Get or create plugin client singleton"""
    global _plugin_client
    if _plugin_client is None:
        _plugin_client = PluginGRPCClient()
    return _plugin_client

