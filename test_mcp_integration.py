"""
Test MCP and Custom GPT Integration

Validates that all endpoints work correctly for:
- Claude Desktop (MCP)
- Custom GPT (OpenAPI Actions)

This ensures operation via AI assistants works after deploy
"""

import asyncio
import httpx
import json

# Test configuration
BASE_URL = "http://localhost:8090"
MCP_BASE = f"{BASE_URL}/api/v1/mcp"

GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
NC = '\033[0m'


async def test_mcp_save_memory():
    """Test 1: MCP darwinSaveMemory"""
    print(f"\n{YELLOW}TEST 1: MCP Save Memory{NC}")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test via MCP execute endpoint
        response = await client.post(
            f"{MCP_BASE}/execute/darwinSaveMemory",
            json={
                "title": "MCP Integration Test",
                "content": "Testing DARWIN 2.0 via MCP after deploy",
                "domain": "research",
                "platform": "claude",
                "tags": ["test", "mcp", "deploy"]
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"{GREEN}✅ MCP Save Memory: OK{NC}")
            print(f"   Content: {result.get('content', [{}])[0].get('text', 'N/A')}")
            return True
        else:
            print(f"{RED}❌ Failed: {response.status_code}{NC}")
            print(f"   {response.text}")
            return False


async def test_mcp_search_memory():
    """Test 2: MCP darwinSearchMemory"""
    print(f"\n{YELLOW}TEST 2: MCP Search Memory{NC}")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{MCP_BASE}/execute/darwinSearchMemory",
            json={
                "query": "deploy test",
                "top_k": 5
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"{GREEN}✅ MCP Search Memory: OK{NC}")
            print(f"   Content: {result.get('content', [{}])[0].get('text', 'N/A')[:100]}")
            return True
        else:
            print(f"{RED}❌ Failed: {response.status_code}{NC}")
            return False


async def test_legacy_endpoints():
    """Test 3: Legacy endpoints (backward compatibility)"""
    print(f"\n{YELLOW}TEST 3: Legacy Endpoints (Custom GPT){NC}")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test old darwinSaveMemory endpoint
        response = await client.post(
            f"{MCP_BASE}/darwinSaveMemory",
            json={
                "title": "Legacy Test",
                "content": "Testing legacy endpoint",
                "domain": "research"
            }
        )
        
        if response.status_code == 200:
            print(f"{GREEN}✅ Legacy darwinSaveMemory: OK{NC}")
            legacy_save = True
        else:
            print(f"{RED}❌ Legacy Save Failed{NC}")
            legacy_save = False
        
        # Test darwinSearchMemory
        response = await client.post(
            f"{MCP_BASE}/darwinSearchMemory",
            json={"query": "test", "top_k": 3}
        )
        
        if response.status_code == 200:
            print(f"{GREEN}✅ Legacy darwinSearchMemory: OK{NC}")
            legacy_search = True
        else:
            print(f"{RED}❌ Legacy Search Failed{NC}")
            legacy_search = False
        
        return legacy_save and legacy_search


async def test_openapi_schema():
    """Test 4: OpenAPI schema (for Custom GPT import)"""
    print(f"\n{YELLOW}TEST 4: OpenAPI Schema{NC}")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{BASE_URL}/openapi.json")
        
        if response.status_code == 200:
            schema = response.json()
            print(f"{GREEN}✅ OpenAPI schema available{NC}")
            print(f"   Title: {schema.get('info', {}).get('title')}")
            print(f"   Version: {schema.get('info', {}).get('version')}")
            
            # Check MCP endpoints exist
            paths = schema.get('paths', {})
            mcp_paths = [p for p in paths if 'mcp' in p or 'darwin' in p.lower()]
            print(f"   MCP endpoints: {len(mcp_paths)}")
            
            return True
        else:
            print(f"{RED}❌ OpenAPI schema not available{NC}")
            return False


async def test_model_management():
    """Test 5: Model Management (list, register)"""
    print(f"\n{YELLOW}TEST 5: Model Management{NC}")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # List models
        response = await client.get(f"{BASE_URL}/api/v1/models/list")
        
        if response.status_code == 200:
            result = response.json()
            total = result.get('total', 0)
            print(f"{GREEN}✅ List Models: {total} models{NC}")
            
            # Show first few
            for model in result.get('models', [])[:3]:
                print(f"   - {model.get('display_name')} ({model.get('provider')})")
            
            return True
        else:
            print(f"{RED}❌ Model list failed{NC}")
            return False


async def test_corpus_ingestion():
    """Test 6: Corpus Ingestion"""
    print(f"\n{YELLOW}TEST 6: Corpus Ingestion{NC}")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Ingest small text
        response = await client.post(
            f"{BASE_URL}/api/v1/corpus/ingest/text",
            json={
                "text": "This is a test scientific paper about biomaterials scaffolds. " * 20,
                "domain": "biomaterials",
                "title": "Test Paper",
                "tags": ["test"]
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"{GREEN}✅ Corpus Ingestion: OK{NC}")
            print(f"   Chunks created: {result.get('chunks_created', 0)}")
            print(f"   Training status: {result.get('training_status', {})}")
            return True
        else:
            print(f"{RED}❌ Corpus ingestion failed{NC}")
            print(f"   {response.text}")
            return False


async def test_training_status():
    """Test 7: Training Status"""
    print(f"\n{YELLOW}TEST 7: Training Status{NC}")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{BASE_URL}/api/v1/corpus/training-status")
        
        if response.status_code == 200:
            result = response.json()
            print(f"{GREEN}✅ Training Status: OK{NC}")
            print(f"   Running: {result.get('running', False)}")
            print(f"   Domains tracked: {len(result.get('domain_counts', {}))}")
            return True
        else:
            print(f"{RED}❌ Training status failed{NC}")
            return False


async def main():
    """Run all MCP/Custom GPT integration tests"""
    print("\n" + "=" * 60)
    print("🧪 DARWIN 2.0 - MCP & Custom GPT Integration Tests")
    print("=" * 60)
    print(f"\nTesting against: {BASE_URL}")
    print("Make sure DARWIN Core is running!")
    print("")
    
    # Check if server is up
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{BASE_URL}/api/v1/health")
            if response.status_code != 200:
                print(f"{RED}❌ Server not responding. Start with:{NC}")
                print(f"   cd darwin-core")
                print(f"   uvicorn app.main:app --host 0.0.0.0 --port 8090")
                return 1
            print(f"{GREEN}✅ Server is up and running{NC}")
    except Exception as e:
        print(f"{RED}❌ Cannot connect to server: {e}{NC}")
        print(f"\nStart server first:")
        print(f"   cd darwin-core")
        print(f"   uvicorn app.main:app --host 0.0.0.0 --port 8090")
        return 1
    
    # Run tests
    results = {}
    
    results['mcp_save'] = await test_mcp_save_memory()
    results['mcp_search'] = await test_mcp_search_memory()
    results['legacy'] = await test_legacy_endpoints()
    results['openapi'] = await test_openapi_schema()
    results['models'] = await test_model_management()
    results['corpus'] = await test_corpus_ingestion()
    results['training'] = await test_training_status()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = f"{GREEN}✅ PASS{NC}" if result else f"{RED}❌ FAIL{NC}"
        print(f"{status} {name}")
    
    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n{GREEN}🎉 All MCP/Custom GPT endpoints working!{NC}")
        print(f"\n✅ DARWIN 2.0 is ready for operation via AI assistants!")
        print(f"\nYou can now:")
        print(f"  1. Configure Claude Desktop MCP")
        print(f"  2. Configure Custom GPT with OpenAPI schema")
        print(f"  3. Use DARWIN through conversations!")
        return 0
    else:
        print(f"\n{YELLOW}⚠️ Some tests failed - fix before deploy{NC}")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

