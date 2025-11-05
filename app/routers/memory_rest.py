"""
Memory REST API Router

External REST interface (HTTP/2) for semantic memory
Compatible with MCP and Custom GPT
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from uuid import uuid4

from ..services.semantic_memory import (
    get_semantic_memory_service,
    ConversationMetadata,
    SearchQuery,
    Domain,
    Platform
)
from ..services.continuous_learning import (
    get_continuous_learning_engine,
    UserInteraction
)

router = APIRouter(prefix="/api/v1/memory")


# Models (simplified for core)
class MemorySaveRequest(BaseModel):
    title: Optional[str] = None
    content: str = Field(..., min_length=1)
    domain: str = Field(default="general")
    platform: str = Field(default="other")
    tags: List[str] = Field(default_factory=list)


class MemorySaveResponse(BaseModel):
    conversation_id: str
    status: str
    message: str
    indexed: bool
    stored_at: str


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    domain: Optional[str] = None
    platform: Optional[str] = None


class MemorySearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[dict]


@router.post("/save", response_model=MemorySaveResponse)
async def save_memory(payload: MemorySaveRequest):
    """
    Save memory to semantic store
    
    Records interaction for continuous learning
    Publishes event to Pulsar
    """
    from ..services.pulsar_client import get_pulsar_client, TOPICS
    
    # Get services
    semantic_memory = get_semantic_memory_service()
    if not semantic_memory.initialize():
        raise HTTPException(status_code=503, detail="Semantic memory service unavailable")
    
    conversation_id = f"mem_{uuid4().hex[:12]}"
    
    # Convert to proper enums
    try:
        domain = Domain(payload.domain)
    except ValueError:
        domain = Domain.GENERAL
    
    try:
        platform = Platform(payload.platform)
    except ValueError:
        platform = Platform.OTHER
    
    # Create metadata
    metadata = ConversationMetadata(
        conversation_id=conversation_id,
        title=payload.title or f"Memory {conversation_id}",
        platform=platform,
        domain=domain,
        tags=payload.tags or [],
        message_count=1
    )
    
    # Save to ChromaDB
    success, message = semantic_memory.save_conversation(
        conversation_id=conversation_id,
        content=payload.content,
        metadata=metadata
    )
    
    if not success:
        raise HTTPException(status_code=500, detail=message)
    
    # Continuous Learning: Record interaction
    try:
        learning_engine = get_continuous_learning_engine(semantic_memory)
        interaction = UserInteraction(
            timestamp=datetime.now(),
            platform=payload.platform,
            domain=payload.domain,
            query=payload.content,
            memory_id=conversation_id,
            tags=payload.tags or [],
            memory_saved=True,
            search_performed=False
        )
        await learning_engine.record_interaction(interaction)
    except Exception as e:
        # Don't fail if continuous learning has issues
        import logging
        logging.getLogger(__name__).warning(f"Continuous learning error: {e}")
    
    # Publish event to Pulsar
    try:
        pulsar = get_pulsar_client()
        await pulsar.publish(TOPICS["continuous_learning"], {
            "event_type": "memory_saved",
            "conversation_id": conversation_id,
            "platform": payload.platform,
            "domain": payload.domain,
            "tags": payload.tags or [],
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception:
        # Don't fail if Pulsar is down
        pass
    
    return MemorySaveResponse(
        conversation_id=conversation_id,
        status="success",
        message=message,
        indexed=True,
        stored_at=datetime.utcnow().isoformat()
    )


@router.post("/search", response_model=MemorySearchResponse)
async def search_memory(payload: MemorySearchRequest):
    """
    Search semantic memory
    
    Records search for continuous learning
    Returns personalized results
    """
    from ..services.pulsar_client import get_pulsar_client, TOPICS
    
    # Get service
    semantic_memory = get_semantic_memory_service()
    if not semantic_memory.initialize():
        raise HTTPException(status_code=503, detail="Semantic memory service unavailable")
    
    # Convert domain/platform to enums if provided
    domain = Domain(payload.domain) if payload.domain else None
    platform = Platform(payload.platform) if payload.platform else None
    
    # Create search query
    query = SearchQuery(
        query=payload.query,
        top_k=payload.top_k,
        domain=domain,
        platform=platform
    )
    
    # Search in ChromaDB
    results = semantic_memory.search_conversations(query)
    
    # Continuous Learning: Record search and personalize results
    try:
        learning_engine = get_continuous_learning_engine(semantic_memory)
        
        interaction = UserInteraction(
            timestamp=datetime.now(),
            platform=payload.platform if payload.platform else "other",
            domain=payload.domain if payload.domain else "general",
            query=payload.query,
            search_performed=True,
            memory_saved=False
        )
        await learning_engine.record_interaction(interaction)
        
        # Personalize results
        results = await learning_engine.personalize_search_results(
            results,
            payload.query,
            payload.domain if payload.domain else "general"
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Continuous learning error: {e}")
    
    # Publish search event to Pulsar
    try:
        pulsar = get_pulsar_client()
        await pulsar.publish(TOPICS["continuous_learning"], {
            "event_type": "search_performed",
            "query": payload.query,
            "domain": payload.domain,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception:
        pass
    
    # Convert SearchResult objects to dicts manually
    results_dicts = []
    for r in results:
        if isinstance(r, dict):
            results_dicts.append(r)
        else:
            # Convert Pydantic model to dict
            try:
                results_dicts.append(r.model_dump())
            except AttributeError:
                try:
                    results_dicts.append(r.dict())
                except AttributeError:
                    # Fallback: manual conversion
                    results_dicts.append({
                        "conversation_id": getattr(r, 'conversation_id', ''),
                        "title": getattr(r, 'title', ''),
                        "content": getattr(r, 'content', ''),
                        "score": getattr(r, 'score', 0.0),
                        "platform": getattr(r, 'platform', ''),
                        "domain": getattr(r, 'domain', ''),
                        "tags": getattr(r, 'tags', []),
                        "created_at": getattr(r, 'created_at', ''),
                        "metadata": getattr(r, 'metadata', {})
                    })
    
    return MemorySearchResponse(
        query=payload.query,
        total_results=len(results),
        results=results_dicts
    )


@router.get("/profile")
async def get_profile():
    """Get user learning profile from Continuous Learning Engine"""
    try:
        semantic_memory = get_semantic_memory_service()
        learning_engine = get_continuous_learning_engine(semantic_memory)
        profile = await learning_engine.export_profile()
        return profile
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

