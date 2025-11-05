"""
Test gRPC Communication between Core and Plugin

Tests:
1. Plugin health check
2. Plugin metadata retrieval
3. KEC analysis request
4. Bi-directional streaming
"""

import asyncio
import grpc
from grpc import aio
import sys
from pathlib import Path

# Add protos to path
sys.path.insert(0, str(Path(__file__).parent / "app" / "protos"))

from plugin.v1 import plugin_pb2, plugin_pb2_grpc
from kec.v1 import kec_pb2, kec_pb2_grpc


async def test_plugin_health(stub):
    """Test 1: Health check"""
    print("\n" + "=" * 60)
    print("TEST 1: Plugin Health Check")
    print("=" * 60)
    
    request = plugin_pb2.HealthRequest()
    response = await stub.HealthCheck(request, timeout=5.0)
    
    print(f"✅ Healthy: {response.healthy}")
    print(f"   Status: {response.status}")
    print(f"   Details: {dict(response.details)}")
    
    assert response.healthy, "Plugin should be healthy"
    return response.healthy


async def test_plugin_metadata(stub):
    """Test 2: Get metadata"""
    print("\n" + "=" * 60)
    print("TEST 2: Plugin Metadata")
    print("=" * 60)
    
    request = plugin_pb2.MetadataRequest()
    response = await stub.GetMetadata(request, timeout=5.0)
    
    print(f"✅ Plugin: {response.name}")
    print(f"   Version: {response.version}")
    print(f"   GPU Required: {response.gpu_required}")
    print(f"   GPU Memory: {response.gpu_memory_mb} MB")
    print(f"   Capabilities: {list(response.capabilities)}")
    print(f"   Proto Version: {response.proto_version}")
    
    assert response.name == "biomaterials", "Plugin name should be biomaterials"
    return response


async def test_kec_analysis(kec_stub):
    """Test 3: KEC Analysis"""
    print("\n" + "=" * 60)
    print("TEST 3: KEC 2.0 Analysis")
    print("=" * 60)
    
    request = kec_pb2.ScaffoldAnalysisRequest(
        dataset_id="test_scaffold_001",
        include_ollivier_ricci=True,
        include_quantum_coherence=False,  # Gated feature
        options={"method": "spectral"}
    )
    
    response = await kec_stub.Analyze(request, timeout=30.0)
    
    if response.success:
        print("✅ Analysis successful!")
        print(f"\n📊 KEC Results:")
        print(f"   H_spectral (Entropia): {response.kec.H_spectral:.4f}")
        print(f"   k_forman_mean (Curvatura Forman): {response.kec.k_forman_mean:.4f}")
        print(f"   k_ollivier_mean (Curvatura Ollivier-Ricci): {response.kec.k_ollivier_mean:.4f}")
        print(f"   sigma (Small-worldness): {response.kec.sigma:.4f}")
        print(f"   phi (SWP): {response.kec.phi:.4f}")
        print(f"   d_perc (Percolação): {response.kec.d_perc_um:.2f} μm")
        print(f"\n⏱️  Processing time: {response.kec.processing_time_ms} ms")
        print(f"   Algorithm version: {response.kec.algorithm_version}")
    else:
        print(f"❌ Analysis failed: {response.error_message}")
    
    assert response.success, "Analysis should succeed"
    assert response.kec.H_spectral > 0, "H_spectral should be > 0"
    
    return response


async def test_streaming(kec_stub):
    """Test 4: Bi-directional streaming"""
    print("\n" + "=" * 60)
    print("TEST 4: Bi-directional Streaming")
    print("=" * 60)
    
    async def chunk_generator():
        """Generate test data chunks"""
        for i in range(5):
            chunk = kec_pb2.DataChunk(
                data=f"chunk_{i}".encode('utf-8'),
                sequence=i,
                is_last=(i == 4),
                metadata={"test": "true"}
            )
            print(f"📤 Sending chunk {i}")
            yield chunk
            await asyncio.sleep(0.1)
    
    # Stream to plugin
    chunks_received = 0
    async for response_chunk in kec_stub.StreamData(chunk_generator()):
        chunks_received += 1
        print(f"📥 Received chunk {response_chunk.sequence}")
    
    print(f"✅ Streaming complete: {chunks_received} chunks received")
    
    assert chunks_received == 5, "Should receive 5 chunks"
    return chunks_received


async def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("🧪 DARWIN gRPC Communication Tests")
    print("=" * 60)
    
    # Connect to plugin
    plugin_address = "localhost:50052"  # Adjust if needed
    print(f"\n🔌 Connecting to: {plugin_address}")
    
    # Create channel
    channel = aio.insecure_channel(
        plugin_address,
        options=[
            ('grpc.max_receive_message_length', 64 * 1024 * 1024),
            ('grpc.max_send_message_length', 64 * 1024 * 1024),
        ]
    )
    
    # Create stubs
    plugin_stub = plugin_pb2_grpc.PluginServiceStub(channel)
    kec_stub = kec_pb2_grpc.KECServiceStub(channel)
    
    try:
        # Run tests
        results = {}
        
        results['health'] = await test_plugin_health(plugin_stub)
        results['metadata'] = await test_plugin_metadata(plugin_stub)
        results['kec_analysis'] = await test_kec_analysis(kec_stub)
        results['streaming'] = await test_streaming(kec_stub)
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Health Check: PASS")
        print(f"✅ Metadata: PASS")
        print(f"✅ KEC Analysis: PASS")
        print(f"✅ Streaming: PASS")
        print("")
        print("🎉 All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        await channel.close()
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

