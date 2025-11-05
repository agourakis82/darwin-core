"""
gRPC Server for DARWIN Core

Corrected implementation using grpc.aio (no ThreadPoolExecutor)
Handles plugin communication with bi-directional streaming
"""

import grpc
from grpc import aio
import logging
from typing import Dict, Any, AsyncIterator, Optional
from datetime import datetime

# Import generated proto stubs (will be generated from .proto files)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "protos"))

try:
    from plugin.v1 import plugin_pb2, plugin_pb2_grpc
    from kec.v1 import kec_pb2, kec_pb2_grpc
except ImportError:
    # Fallback if protos not generated yet
    logger = logging.getLogger("darwin.grpc")
    logger.warning("Proto files not generated yet. Run: python -m grpc_tools.protoc ...")
    plugin_pb2 = None
    plugin_pb2_grpc = None
    kec_pb2 = None
    kec_pb2_grpc = None

from ..services.pulsar_client import get_pulsar_client, TOPICS

logger = logging.getLogger("darwin.grpc.core")


class PluginGatewayServicer(plugin_pb2_grpc.PluginServiceServicer if plugin_pb2_grpc else object):
    """
    gRPC Gateway for plugin communication
    
    Routes requests to appropriate plugins via gRPC
    Publishes events to Pulsar for continuous learning
    """
    
    def __init__(self):
        self.plugin_clients: Dict[str, aio.Channel] = {}
        self.plugin_stubs: Dict[str, Any] = {}
        
    async def connect_to_plugin(self, plugin_name: str, host: str, port: int):
        """Connect to a plugin's gRPC server"""
        address = f"{host}:{port}"
        
        # Create async channel
        channel = aio.insecure_channel(
            address,
            options=[
                ('grpc.max_receive_message_length', 64 * 1024 * 1024),  # 64MB
                ('grpc.max_send_message_length', 64 * 1024 * 1024),
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 10000),
            ]
        )
        
        # Create stub
        if plugin_pb2_grpc:
            stub = plugin_pb2_grpc.PluginServiceStub(channel)
            
            self.plugin_clients[plugin_name] = channel
            self.plugin_stubs[plugin_name] = stub
            
            logger.info(f"✅ Connected to plugin {plugin_name} at {address}")
        
    async def GetMetadata(self, request, context):
        """Forward metadata request to plugin"""
        plugin_name = context.peer().split(':')[0]  # Extract from peer info
        
        if plugin_name not in self.plugin_stubs:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Plugin {plugin_name} not registered")
            return plugin_pb2.PluginMetadata()
        
        stub = self.plugin_stubs[plugin_name]
        response = await stub.GetMetadata(request)
        
        return response
    
    async def HealthCheck(self, request, context):
        """Health check for plugin"""
        plugin_name = context.peer().split(':')[0]
        
        if plugin_name not in self.plugin_stubs:
            return plugin_pb2.HealthResponse(
                healthy=False,
                status="unknown",
                details={"error": f"Plugin {plugin_name} not registered"}
            )
        
        stub = self.plugin_stubs[plugin_name]
        try:
            response = await stub.HealthCheck(request, timeout=5.0)
            
            # Publish health event to Pulsar
            pulsar = get_pulsar_client()
            await pulsar.publish(TOPICS["plugin_health"], {
                "plugin": plugin_name,
                "status": response.status,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return response
        except Exception as e:
            logger.error(f"Health check failed for {plugin_name}: {e}")
            return plugin_pb2.HealthResponse(
                healthy=False,
                status="unhealthy",
                details={"error": str(e)}
            )
    
    async def Execute(self, request, context):
        """Execute plugin operation"""
        plugin_name = request.metadata.get("plugin_name", "unknown")
        
        if plugin_name not in self.plugin_stubs:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return plugin_pb2.ExecuteResponse(
                success=False,
                error=f"Plugin {plugin_name} not registered"
            )
        
        stub = self.plugin_stubs[plugin_name]
        
        try:
            # Forward to plugin
            response = await stub.Execute(request, timeout=30.0)
            
            # Publish interaction event
            pulsar = get_pulsar_client()
            await pulsar.publish(TOPICS["continuous_learning"], {
                "plugin": plugin_name,
                "operation": request.operation,
                "success": response.success,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": dict(request.metadata)
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Execute failed for {plugin_name}: {e}")
            return plugin_pb2.ExecuteResponse(
                success=False,
                error=str(e)
            )
    
    async def StreamData(
        self,
        request_iterator: AsyncIterator,
        context
    ) -> AsyncIterator:
        """
        Bi-directional streaming for large data (e.g., MicroCT)
        
        Handles backpressure and chunking automatically
        """
        plugin_name = context.peer().split(':')[0]
        
        if plugin_name not in self.plugin_stubs:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return
        
        stub = self.plugin_stubs[plugin_name]
        
        try:
            # Stream to plugin and stream back
            async for chunk in stub.StreamData(request_iterator):
                yield chunk
                
        except Exception as e:
            logger.error(f"Streaming error for {plugin_name}: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
    
    async def close_all_connections(self):
        """Close all plugin connections"""
        for channel in self.plugin_clients.values():
            await channel.close()
        logger.info("✅ All plugin connections closed")


# Global server instance
_grpc_server: Optional[aio.Server] = None
_servicer: Optional[PluginGatewayServicer] = None


async def start_grpc_server(port: int = 50051) -> aio.Server:
    """
    Start gRPC server (async, no ThreadPoolExecutor)
    
    Corrected implementation following grpc.aio best practices
    """
    global _grpc_server, _servicer
    
    if not plugin_pb2_grpc:
        logger.warning("⚠️ Proto files not generated, skipping gRPC server")
        return None
    
    # Create async server
    server = aio.server(
        options=[
            ('grpc.max_receive_message_length', 64 * 1024 * 1024),
            ('grpc.max_send_message_length', 64 * 1024 * 1024),
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 10000),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.keepalive_permit_without_calls', 1),
        ]
    )
    
    # Create and register servicer
    _servicer = PluginGatewayServicer()
    plugin_pb2_grpc.add_PluginServiceServicer_to_server(_servicer, server)
    
    # Bind port (insecure for now, add TLS later)
    server.add_insecure_port(f'[::]:{port}')
    
    # Start server
    await server.start()
    _grpc_server = server
    
    logger.info(f"🚀 gRPC server started on port {port}")
    logger.info(f"   Using HTTP/2 (gRPC default)")
    logger.info(f"   Max message size: 64MB")
    
    return server


async def stop_grpc_server():
    """Stop gRPC server gracefully"""
    global _grpc_server, _servicer
    
    if _servicer:
        await _servicer.close_all_connections()
    
    if _grpc_server:
        await _grpc_server.stop(grace=5.0)
        logger.info("✅ gRPC server stopped")


def get_grpc_servicer() -> PluginGatewayServicer:
    """Get servicer instance for registering plugins"""
    global _servicer
    return _servicer

