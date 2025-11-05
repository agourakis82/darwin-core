"""
Darwin Memory API Router
Handles memory storage and retrieval in Qdrant K8s
"""
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import os

logger = logging.getLogger("darwin.memory")

router = APIRouter(prefix="/api/memory", tags=["memory"])

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant.darwin.svc.cluster.local")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "darwin_memories"
VECTOR_SIZE = 384

# Global clients (initialized on startup)
qdrant_client: Optional[QdrantClient] = None
embedding_model: Optional[SentenceTransformer] = None

# Models
class MemoryCreate(BaseModel):
    title: str = Field(..., description="Memory title")
    content: str = Field(..., description="Memory content/text")
    type: str = Field(default="general", description="Memory type: technical, session, decision, general")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class MemorySearch(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Number of results")
    type: Optional[str] = Field(default=None, description="Filter by type")

class MemoryResponse(BaseModel):
    id: str
    title: str
    content: str
    type: str
    score: Optional[float] = None
    metadata: Dict[str, Any]

class MemoryImportBatch(BaseModel):
    memories: List[MemoryCreate] = Field(..., description="Batch of memories to import")

def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client"""
    global qdrant_client
    if qdrant_client is None:
        logger.info(f"Connecting to Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        ensure_collection()
    return qdrant_client

def get_embedding_model() -> SentenceTransformer:
    """Get or create embedding model"""
    global embedding_model
    if embedding_model is None:
        logger.info("Loading SentenceTransformer model: all-MiniLM-L6-v2")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model

def ensure_collection():
    """Ensure Qdrant collection exists"""
    client = get_qdrant_client()
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if COLLECTION_NAME not in collection_names:
        logger.info(f"Creating collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        logger.info(f"✅ Collection '{COLLECTION_NAME}' created")

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text"""
    model = get_embedding_model()
    embedding = model.encode(text, convert_to_tensor=False)
    return embedding.tolist()

@router.post("/create", response_model=MemoryResponse)
async def create_memory(memory: MemoryCreate):
    """Create a new memory"""
    try:
        client = get_qdrant_client()
        
        # Generate embedding
        embedding = generate_embedding(memory.content)
        
        # Create metadata
        metadata = memory.metadata or {}
        metadata.update({
            "title": memory.title,
            "type": memory.type,
            "created_at": datetime.now().isoformat(),
            "source": "api"
        })
        
        # Generate unique ID
        memory_id = hash(f"{memory.title}_{memory.content}_{datetime.now().isoformat()}") % (2**63 - 1)
        
        # Create point
        point = PointStruct(
            id=memory_id,
            vector=embedding,
            payload={
                "content": memory.content,
                **metadata
            }
        )
        
        # Store in Qdrant
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )
        
        logger.info(f"✅ Memory created: {memory.title} (ID: {memory_id})")
        
        return MemoryResponse(
            id=str(memory_id),
            title=memory.title,
            content=memory.content,
            type=memory.type,
            metadata=metadata
        )
    
    except Exception as e:
        logger.error(f"Error creating memory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create memory: {str(e)}")

@router.post("/search", response_model=List[MemoryResponse])
async def search_memories(search: MemorySearch):
    """Search memories by semantic similarity"""
    try:
        client = get_qdrant_client()
        
        # Generate query embedding
        query_embedding = generate_embedding(search.query)
        
        # Build filter
        query_filter = None
        if search.type:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="type",
                        match=MatchValue(value=search.type)
                    )
                ]
            )
        
        # Search
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=search.limit,
            query_filter=query_filter
        )
        
        # Format results
        memories = []
        for result in results:
            memories.append(MemoryResponse(
                id=str(result.id),
                title=result.payload.get("title", "Untitled"),
                content=result.payload.get("content", ""),
                type=result.payload.get("type", "general"),
                score=result.score,
                metadata={k: v for k, v in result.payload.items() if k not in ["content", "title", "type"]}
            ))
        
        logger.info(f"🔍 Memory search: '{search.query}' → {len(memories)} results")
        
        return memories
    
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search memories: {str(e)}")

@router.post("/import", response_model=Dict[str, Any])
async def import_memories(batch: MemoryImportBatch):
    """Import batch of memories"""
    try:
        client = get_qdrant_client()
        
        points = []
        for memory in batch.memories:
            # Generate embedding
            embedding = generate_embedding(memory.content)
            
            # Create metadata
            metadata = memory.metadata or {}
            metadata.update({
                "title": memory.title,
                "type": memory.type,
                "created_at": datetime.now().isoformat(),
                "source": "import"
            })
            
            # Generate unique ID
            memory_id = hash(f"{memory.title}_{memory.content}_{datetime.now().isoformat()}") % (2**63 - 1)
            
            # Create point
            point = PointStruct(
                id=memory_id,
                vector=embedding,
                payload={
                    "content": memory.content,
                    **metadata
                }
            )
            points.append(point)
        
        # Batch upload
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        logger.info(f"✅ Imported {len(points)} memories")
        
        return {
            "success": True,
            "imported": len(points),
            "message": f"Successfully imported {len(points)} memories"
        }
    
    except Exception as e:
        logger.error(f"Error importing memories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to import memories: {str(e)}")

@router.get("/list", response_model=Dict[str, Any])
async def list_memories(limit: int = 100, offset: int = 0, type: Optional[str] = None):
    """List all memories"""
    try:
        client = get_qdrant_client()
        
        # Build filter
        query_filter = None
        if type:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="type",
                        match=MatchValue(value=type)
                    )
                ]
            )
        
        # Scroll through collection
        results, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
            scroll_filter=query_filter
        )
        
        # Format results
        memories = []
        for result in results:
            memories.append({
                "id": str(result.id),
                "title": result.payload.get("title", "Untitled"),
                "content": result.payload.get("content", "")[:200] + "...",  # Truncate
                "type": result.payload.get("type", "general"),
                "created_at": result.payload.get("created_at", "")
            })
        
        # Get total count
        collection_info = client.get_collection(COLLECTION_NAME)
        
        return {
            "memories": memories,
            "total": collection_info.points_count,
            "offset": offset,
            "limit": limit,
            "next_offset": next_offset
        }
    
    except Exception as e:
        logger.error(f"Error listing memories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list memories: {str(e)}")

@router.get("/stats")
async def get_memory_stats():
    """Get memory statistics"""
    try:
        client = get_qdrant_client()
        collection_info = client.get_collection(COLLECTION_NAME)
        
        return {
            "collection": COLLECTION_NAME,
            "total_memories": collection_info.points_count,
            "vector_size": VECTOR_SIZE,
            "status": "operational"
        }
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@router.delete("/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory by ID"""
    try:
        client = get_qdrant_client()
        
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=[int(memory_id)]
        )
        
        logger.info(f"🗑️  Memory deleted: {memory_id}")
        
        return {"success": True, "message": f"Memory {memory_id} deleted"}
    
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete memory: {str(e)}")

# Startup event handler
def init_memory_system():
    """Initialize memory system on startup"""
    logger.info("🧠 Initializing Darwin Memory System")
    try:
        get_qdrant_client()
        get_embedding_model()
        logger.info("✅ Darwin Memory System ready")
    except Exception as e:
        logger.error(f"❌ Failed to initialize memory system: {e}")

