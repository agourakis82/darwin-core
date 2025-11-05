"""Semantic Memory Service - Unified conversational memory for DARWIN

🧠 SEMANTIC MEMORY SYSTEM
Gerencia memória persistente e semântica para todas as conversas AI:
- Multi-plataforma (ChatGPT, Claude, Gemini, etc)
- Busca semântica via ChromaDB
- Metadados ricos (domain, tags, platform)
- Ranking inteligente de resultados

Technology Stack:
- ChromaDB (vector database)
- Sentence Transformers (embeddings)
- FastAPI (API layer)
"""

import logging
import os
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from ..core.logging import get_logger
from .chroma_client import get_chroma_client, CHROMA_AVAILABLE
# Disable v2 bootstrap due to tenant/database bugs in chromadb 0.5.0
# try:
#     from .chroma_bootstrap import get_chroma_client as get_chroma_client_v2, ensure_tenant_and_database
#     CHROMA_V2_AVAILABLE = True
# except ImportError:
CHROMA_V2_AVAILABLE = False
get_chroma_client_v2 = None
ensure_tenant_and_database = None

logger = get_logger("darwin.semantic_memory")


# ===== ENUMS =====

class Platform(str, Enum):
    """AI platforms supported"""
    CHATGPT = "chatgpt"
    CLAUDE = "claude"
    CLAUDE_CODE = "claude-code"
    GEMINI = "gemini"
    COPILOT = "copilot"
    OTHER = "other"


class Domain(str, Enum):
    """Knowledge domains - all DARWIN plugins"""
    # Core domains
    BIOMATERIALS = "biomaterials"
    CHEMISTRY = "chemistry"
    PBPK = "pbpk"
    
    # Medical domains
    PSYCHIATRY = "psychiatry"
    INTERNAL_MEDICINE = "internal_medicine"
    
    # Discovery domains
    META_DISCOVERY = "meta_discovery"
    Q1_SCHOLAR = "q1_scholar"
    DISCOVERY = "discovery"
    
    # Science domains
    HELIOBIOLOGY = "heliobiology"
    COMPLEX_SYSTEMS = "complex_systems"
    PHYSICS = "physics"
    NEUROSCIENCE = "neuroscience"
    QUANTUM = "quantum"
    
    # Humanities
    PHILOSOPHY = "philosophy"
    
    # Utility domains
    CODE_GENERATION = "code_generation"
    RESEARCH = "research"
    GENERAL = "general"


# ===== DATA MODELS =====

class ConversationMetadata(BaseModel):
    """Metadata for a conversation"""
    conversation_id: str
    title: str
    platform: Platform
    domain: Domain
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    message_count: int = 0
    summary: Optional[str] = None

    # Additional metadata
    language: str = "en"
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class ConversationChunk(BaseModel):
    """A chunk of conversation for indexing"""
    chunk_id: str
    conversation_id: str
    content: str
    metadata: ConversationMetadata

    # Chunk-specific metadata
    chunk_index: int = 0
    total_chunks: int = 1


class SearchResult(BaseModel):
    """A single search result"""
    conversation_id: str
    title: str
    content: str
    score: float
    platform: str
    domain: str
    tags: List[str]
    created_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchQuery(BaseModel):
    """Search query parameters"""
    query: str
    top_k: int = Field(5, ge=1, le=50)
    domain: Optional[Domain] = None
    platform: Optional[Platform] = None
    tags: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


# ===== SEMANTIC MEMORY SERVICE =====

class SemanticMemoryService:
    """
    Unified semantic memory service for DARWIN.

    Manages conversational memory across all AI platforms with:
    - Semantic search via vector embeddings
    - Rich metadata filtering
    - Cross-platform conversation tracking
    """

    COLLECTION_NAME = "darwin_semantic_memory"

    def __init__(self, chroma_host: Optional[str] = None, chroma_port: Optional[int] = None):
        """Initialize semantic memory service"""
        self.chroma_host = chroma_host or os.getenv("CHROMA_HOST", "localhost")
        try:
            self.chroma_port = int(chroma_port if chroma_port is not None else os.getenv("CHROMA_PORT", "8000"))
        except ValueError:
            logger.warning("Invalid CHROMA_PORT value. Falling back to 8000")
            self.chroma_port = 8000
        self.chroma_client = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize ChromaDB connection"""
        if self._initialized:
            return True

        if not CHROMA_AVAILABLE:
            logger.warning("ChromaDB not available - semantic memory disabled")
            return False

        try:
            # Use ChromaDB v2 client with tenant/database if available
            if CHROMA_V2_AVAILABLE and get_chroma_client_v2:
                logger.info("Using ChromaDB v2 client with tenant/database")
                self.chroma_client = get_chroma_client_v2()
            else:
                logger.info("Using legacy ChromaDB client")
                self.chroma_client = get_chroma_client(
                    host=self.chroma_host,
                    port=self.chroma_port
                )

            if self.chroma_client:
                # Ensure collection exists
                self.chroma_client.get_or_create_collection(self.COLLECTION_NAME)
                self._initialized = True
                logger.info("✅ Semantic Memory Service initialized")
                return True
            else:
                logger.warning("Failed to get ChromaDB client")
                return False

        except Exception as e:
            logger.error(f"Failed to initialize semantic memory: {e}")
            return False

    def save_conversation(
        self,
        conversation_id: str,
        content: str,
        metadata: ConversationMetadata
    ) -> Tuple[bool, str]:
        """
        Save a conversation to semantic memory.

        Args:
            conversation_id: Unique conversation ID
            content: Full conversation content
            metadata: Rich metadata

        Returns:
            (success, message)
        """
        if not self._initialized and not self.initialize():
            return False, "Semantic memory not available"

        try:
            # Prepare document metadata for ChromaDB
            doc_metadata = {
                "conversation_id": conversation_id,
                "title": metadata.title,
                "platform": metadata.platform.value,
                "domain": metadata.domain.value,
                "tags": ",".join(metadata.tags),
                "created_at": metadata.created_at.isoformat(),
                "updated_at": metadata.updated_at.isoformat(),
                "message_count": str(metadata.message_count),
                "language": metadata.language,
                "type": "conversation"
            }

            if metadata.summary:
                doc_metadata["summary"] = metadata.summary
            if metadata.user_id:
                doc_metadata["user_id"] = metadata.user_id
            if metadata.session_id:
                doc_metadata["session_id"] = metadata.session_id

            # Store in ChromaDB
            success = self.chroma_client.add_documents(
                collection_name=self.COLLECTION_NAME,
                documents=[content],
                ids=[conversation_id],
                metadatas=[doc_metadata]
            )

            if success:
                logger.info(f"✅ Conversation {conversation_id} saved to semantic memory")
                return True, f"Conversation saved successfully"
            else:
                logger.error(f"Failed to save conversation {conversation_id}")
                return False, "Failed to save to vector database"

        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return False, f"Error: {str(e)}"

    def search_conversations(
        self,
        query: SearchQuery
    ) -> List[SearchResult]:
        """
        Search conversations semantically.

        Args:
            query: Search query with filters

        Returns:
            List of search results
        """
        if not self._initialized and not self.initialize():
            logger.warning("Semantic memory not available for search")
            return []

        try:
            # Build metadata filter
            where_filter = {}

            if query.domain:
                where_filter["domain"] = query.domain.value

            if query.platform:
                where_filter["platform"] = query.platform.value

            # TODO: Add date filtering when ChromaDB supports it better
            # For now, we'll filter in post-processing

            # Perform semantic search
            results = self.chroma_client.query(
                collection_name=self.COLLECTION_NAME,
                query_texts=[query.query],
                n_results=query.top_k * 2,  # Get more results for filtering
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )

            # Process results
            search_results = []

            if results and results.get("ids"):
                ids = results["ids"][0]
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]

                for i in range(len(ids)):
                    metadata = metadatas[i]

                    # Tag filtering (post-process)
                    if query.tags:
                        doc_tags = metadata.get("tags", "").split(",")
                        if not any(tag in doc_tags for tag in query.tags):
                            continue

                    # Date filtering (post-process)
                    created_at_str = metadata.get("created_at", "")
                    if query.date_from or query.date_to:
                        try:
                            created_at = datetime.fromisoformat(created_at_str)
                            if query.date_from and created_at < query.date_from:
                                continue
                            if query.date_to and created_at > query.date_to:
                                continue
                        except:
                            pass

                    # Convert distance to similarity score (lower distance = higher score)
                    score = 1.0 - min(distances[i], 1.0)

                    result = SearchResult(
                        conversation_id=metadata.get("conversation_id", ids[i]),
                        title=metadata.get("title", "Untitled"),
                        content=documents[i][:500] + ("..." if len(documents[i]) > 500 else ""),
                        score=score,
                        platform=metadata.get("platform", "unknown"),
                        domain=metadata.get("domain", "general"),
                        tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
                        created_at=created_at_str,
                        metadata=metadata
                    )

                    search_results.append(result)

                    # Stop when we have enough results
                    if len(search_results) >= query.top_k:
                        break

            logger.info(f"✅ Search returned {len(search_results)} results for: {query.query}")
            return search_results

        except Exception as e:
            logger.error(f"Error searching conversations: {e}")
            return []

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific conversation by ID.

        Args:
            conversation_id: Conversation ID

        Returns:
            Conversation data or None
        """
        if not self._initialized and not self.initialize():
            return None

        try:
            results = self.chroma_client.get_documents(
                collection_name=self.COLLECTION_NAME,
                ids=[conversation_id],
                include=["documents", "metadatas"]
            )

            if results and results.get("ids") and len(results["ids"]) > 0:
                return {
                    "id": results["ids"][0],
                    "content": results["documents"][0],
                    "metadata": results["metadatas"][0]
                }

            return None

        except Exception as e:
            logger.error(f"Error getting conversation {conversation_id}: {e}")
            return None

    def list_recent_conversations(
        self,
        limit: int = 10,
        domain: Optional[Domain] = None,
        platform: Optional[Platform] = None
    ) -> List[Dict[str, Any]]:
        """
        List recent conversations.

        Args:
            limit: Number of conversations to return
            domain: Optional domain filter
            platform: Optional platform filter

        Returns:
            List of conversation metadata
        """
        if not self._initialized and not self.initialize():
            return []

        try:
            # Build filter
            where_filter = {}
            if domain:
                where_filter["domain"] = domain.value
            if platform:
                where_filter["platform"] = platform.value

            # Get recent conversations
            results = self.chroma_client.get_documents(
                collection_name=self.COLLECTION_NAME,
                where=where_filter if where_filter else None,
                limit=limit,
                include=["metadatas"]
            )

            conversations = []
            if results and results.get("ids"):
                for i in range(len(results["ids"])):
                    conversations.append({
                        "id": results["ids"][i],
                        "metadata": results["metadatas"][i]
                    })

            # Sort by created_at (most recent first)
            conversations.sort(
                key=lambda x: x["metadata"].get("created_at", ""),
                reverse=True
            )

            return conversations[:limit]

        except Exception as e:
            logger.error(f"Error listing conversations: {e}")
            return []

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            True if successful
        """
        if not self._initialized and not self.initialize():
            return False

        try:
            success = self.chroma_client.delete_documents(
                collection_name=self.COLLECTION_NAME,
                ids=[conversation_id]
            )

            if success:
                logger.info(f"✅ Conversation {conversation_id} deleted")

            return success

        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get semantic memory statistics.

        Returns:
            Statistics dict
        """
        if not self._initialized and not self.initialize():
            return {
                "available": False,
                "total_conversations": 0
            }

        try:
            count = self.chroma_client.count(self.COLLECTION_NAME)

            return {
                "available": True,
                "total_conversations": count,
                "collection_name": self.COLLECTION_NAME,
                "chroma_host": self.chroma_host,
                "chroma_port": self.chroma_port
            }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "available": False,
                "error": str(e)
            }


# ===== SINGLETON INSTANCE =====

_semantic_memory_service: Optional[SemanticMemoryService] = None


def get_semantic_memory_service() -> SemanticMemoryService:
    """Get singleton semantic memory service instance"""
    global _semantic_memory_service

    if _semantic_memory_service is None:
        _semantic_memory_service = SemanticMemoryService()
        _semantic_memory_service.initialize()

    return _semantic_memory_service
