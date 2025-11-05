"""
Semantic Memory Service v2 - Integrated with Embedding Manager 2025

Migração para estado da arte 2025:
- Embedding Manager (nomic/jina/gte-Qwen2)
- Qdrant Hybrid (dense + sparse)
- Late chunking
- Binary quantization
- Cache otimizado
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..services.embedding_manager import (
    get_embedding_manager,
    EmbeddingConfig,
    EmbeddingModel
)
from ..services.qdrant_hybrid_client import (
    get_qdrant_client,
    HybridSearchConfig
)
from ..services.semantic_memory import (
    SemanticMemoryService,
    ConversationMetadata,
    ConversationChunk,
    SearchResult,
    Platform,
    Domain
)

logger = logging.getLogger(__name__)


class SemanticMemoryServiceV2(SemanticMemoryService):
    """
    Semantic Memory Service v2 - Estado da Arte 2025
    
    Melhorias sobre v1:
    - Embedding Manager (modelos SOTA)
    - Qdrant híbrido (dense + sparse)
    - Late chunking automático
    - Binary quantization
    - Cache inteligente
    
    Backward compatible com v1
    """
    
    def __init__(
        self,
        use_qdrant: bool = True,
        embedding_model: EmbeddingModel = EmbeddingModel.JINA_V3,
        use_gpu: bool = True,
        enable_quantization: bool = True,
        **kwargs
    ):
        """
        Initialize v2 semantic memory
        
        Args:
            use_qdrant: Use Qdrant instead of ChromaDB
            embedding_model: Which embedding model to use
            use_gpu: Enable GPU acceleration
            enable_quantization: Enable binary quantization
            **kwargs: Passed to parent class
        """
        # Initialize parent (for backward compatibility)
        super().__init__(**kwargs)
        
        self.use_qdrant = use_qdrant
        self.embedding_model = embedding_model
        
        # Initialize Embedding Manager
        embedding_config = EmbeddingConfig(
            model=embedding_model,
            use_gpu=use_gpu,
            enable_binary_quantization=enable_quantization,
            enable_cache=True,
            cache_ttl=3600
        )
        self.embedding_manager = get_embedding_manager(config=embedding_config)
        
        # Initialize vector database
        if use_qdrant:
            self.qdrant_client = get_qdrant_client()
            self._init_qdrant()
        else:
            self.qdrant_client = None
        
        logger.info(
            f"✅ Semantic Memory v2 initialized "
            f"(model={embedding_model.value}, qdrant={use_qdrant})"
        )
    
    def _init_qdrant(self):
        """Initialize Qdrant collection"""
        if not self.qdrant_client or not self.qdrant_client.initialize():
            logger.warning("Qdrant not available, falling back to ChromaDB")
            self.use_qdrant = False
            return
        
        # Get embedding dimension
        test_emb = self.embedding_manager.encode("test")
        vector_size = test_emb.shape[0]
        
        # Create collection if not exists
        try:
            self.qdrant_client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vector_size=vector_size,
                enable_sparse=True,  # Hybrid search
                enable_quantization=True,
                quantization_type="binary",
                shard_number=1,
                replication_factor=1
            )
            logger.info(f"✅ Qdrant collection ready (dim={vector_size})")
        except Exception as e:
            # Collection might already exist
            logger.info(f"Qdrant collection exists: {e}")
    
    async def save_conversation(
        self,
        conversation_id: str,
        title: str,
        content: str,
        platform: Platform,
        domain: Domain,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        auto_chunk: bool = True,
        chunk_size: int = 512
    ) -> bool:
        """
        Save conversation with v2 features
        
        Args:
            conversation_id: Unique conversation ID
            title: Conversation title
            content: Full conversation content
            platform: AI platform
            domain: Knowledge domain
            tags: Optional tags
            metadata: Additional metadata
            user_id: Optional user ID
            auto_chunk: Enable late chunking for long texts
            chunk_size: Chunk size for late chunking
        
        Returns:
            True if successful
        """
        try:
            # Create metadata
            conv_metadata = ConversationMetadata(
                conversation_id=conversation_id,
                title=title,
                platform=platform,
                domain=domain,
                tags=tags or [],
                user_id=user_id,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Chunking strategy
            if auto_chunk and len(content) > chunk_size:
                # Late chunking para textos longos
                logger.info(f"Applying late chunking (size={len(content)})")
                chunks_embeddings = self.embedding_manager.encode_with_late_chunking(
                    content,
                    chunk_size=chunk_size,
                    chunk_overlap=50
                )
                chunks_text = self._split_text(content, chunk_size, overlap=50)
            else:
                # Single embedding
                embedding = self.embedding_manager.encode(content)
                chunks_embeddings = [embedding]
                chunks_text = [content]
            
            # Save to vector database
            if self.use_qdrant and self.qdrant_client:
                return await self._save_to_qdrant(
                    conversation_id,
                    conv_metadata,
                    chunks_text,
                    chunks_embeddings
                )
            else:
                # Fallback to ChromaDB (v1)
                return await super().save_conversation(
                    conversation_id=conversation_id,
                    title=title,
                    content=content,
                    platform=platform,
                    domain=domain,
                    tags=tags,
                    user_id=user_id
                )
        
        except Exception as e:
            logger.error(f"Failed to save conversation v2: {e}")
            return False
    
    async def _save_to_qdrant(
        self,
        conversation_id: str,
        metadata: ConversationMetadata,
        chunks: List[str],
        embeddings: List[Any]
    ) -> bool:
        """Save to Qdrant with hybrid vectors"""
        try:
            points = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # TODO: Implement sparse embeddings (BM25/SPLADE)
                # For now, use only dense
                
                point = {
                    "id": f"{conversation_id}_{i}",
                    "vector": {
                        "dense": embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                    },
                    "payload": {
                        "conversation_id": conversation_id,
                        "chunk_index": i,
                        "chunk_text": chunk,
                        "title": metadata.title,
                        "platform": metadata.platform.value,
                        "domain": metadata.domain.value,
                        "tags": metadata.tags,
                        "created_at": metadata.created_at.isoformat(),
                        "user_id": metadata.user_id,
                    }
                }
                points.append(point)
            
            # Upsert points
            success = self.qdrant_client.upsert_points(
                self.COLLECTION_NAME,
                points,
                batch_size=100
            )
            
            if success:
                logger.info(
                    f"✅ Saved conversation to Qdrant: {conversation_id} "
                    f"({len(chunks)} chunks)"
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to save to Qdrant: {e}")
            return False
    
    async def search_conversations(
        self,
        query: str,
        limit: int = 10,
        platform: Optional[Platform] = None,
        domain: Optional[Domain] = None,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        use_hybrid: bool = True,
        hybrid_weight_dense: float = 0.7
    ) -> List[SearchResult]:
        """
        Search conversations with v2 features
        
        Args:
            query: Search query
            limit: Max results
            platform: Filter by platform
            domain: Filter by domain
            tags: Filter by tags
            user_id: Filter by user
            use_hybrid: Use hybrid search (dense + sparse)
            hybrid_weight_dense: Weight for dense vector (0-1)
        
        Returns:
            List of search results
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.encode(query)
            
            # Qdrant search
            if self.use_qdrant and self.qdrant_client:
                return await self._search_qdrant(
                    query_embedding,
                    limit,
                    platform,
                    domain,
                    tags,
                    user_id,
                    use_hybrid,
                    hybrid_weight_dense
                )
            else:
                # Fallback to ChromaDB (v1)
                return await super().search_conversations(
                    query=query,
                    limit=limit,
                    platform=platform,
                    domain=domain,
                    tags=tags,
                    user_id=user_id
                )
        
        except Exception as e:
            logger.error(f"Failed to search conversations v2: {e}")
            return []
    
    async def _search_qdrant(
        self,
        query_embedding: Any,
        limit: int,
        platform: Optional[Platform],
        domain: Optional[Domain],
        tags: Optional[List[str]],
        user_id: Optional[str],
        use_hybrid: bool,
        hybrid_weight_dense: float
    ) -> List[SearchResult]:
        """Search Qdrant with hybrid search"""
        try:
            # Build filter
            filter_dict = {}
            if platform:
                filter_dict["platform"] = platform.value
            if domain:
                filter_dict["domain"] = domain.value
            if user_id:
                filter_dict["user_id"] = user_id
            # TODO: Implement tags filter (Qdrant supports arrays)
            
            # Hybrid search config
            config = HybridSearchConfig(
                dense_weight=hybrid_weight_dense,
                sparse_weight=1.0 - hybrid_weight_dense,
                use_rrf=use_hybrid,
                top_k=limit
            )
            
            # Search
            results = self.qdrant_client.hybrid_search(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_embedding,
                config=config,
                filter=filter_dict if filter_dict else None
            )
            
            # Convert to SearchResult
            search_results = []
            for result in results:
                payload = result["payload"]
                
                search_result = SearchResult(
                    conversation_id=payload["conversation_id"],
                    chunk_index=payload.get("chunk_index", 0),
                    content=payload["chunk_text"],
                    title=payload["title"],
                    platform=Platform(payload["platform"]),
                    domain=Domain(payload["domain"]),
                    tags=payload.get("tags", []),
                    score=result["score"],
                    created_at=datetime.fromisoformat(payload["created_at"])
                )
                search_results.append(search_result)
            
            logger.info(f"✅ Found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search Qdrant: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get v2 stats"""
        stats = {}
        
        # Embedding manager stats
        emb_stats = self.embedding_manager.get_cache_stats()
        stats["embedding_manager"] = emb_stats
        
        # Qdrant stats
        if self.use_qdrant and self.qdrant_client:
            try:
                info = self.qdrant_client.get_collection_info(self.COLLECTION_NAME)
                stats["qdrant"] = info
            except:
                stats["qdrant"] = {"error": "Failed to get stats"}
        
        # Parent stats (ChromaDB)
        if not self.use_qdrant:
            parent_stats = super().get_stats()
            stats["chromadb"] = parent_stats
        
        return stats
    
    def _split_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int = 50
    ) -> List[str]:
        """Split text into chunks with overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks


# Factory function
def get_semantic_memory_v2(
    use_qdrant: bool = True,
    embedding_model: EmbeddingModel = EmbeddingModel.JINA_V3,
    **kwargs
) -> SemanticMemoryServiceV2:
    """
    Factory para Semantic Memory v2
    
    Args:
        use_qdrant: Use Qdrant (True) or ChromaDB (False)
        embedding_model: Embedding model to use
        **kwargs: Additional config
    
    Returns:
        SemanticMemoryServiceV2 instance
    """
    return SemanticMemoryServiceV2(
        use_qdrant=use_qdrant,
        embedding_model=embedding_model,
        **kwargs
    )


__all__ = [
    "SemanticMemoryServiceV2",
    "get_semantic_memory_v2",
]

