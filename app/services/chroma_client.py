"""ChromaDB Client - Vector Database for DARWIN RAG++

🌟 CHROMADB CLIENT SYSTEM
Cliente para integração com ChromaDB vector database:
- Embedding storage and retrieval
- Semantic search
- Collection management
- Metadata filtering

Technology: ChromaDB + Vertex AI Embeddings
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from ..core.logging import get_logger

logger = get_logger("darwin.chroma_client")

# ChromaDB imports
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.api.types import Documents, Embeddings, IDs, Metadatas
    CHROMA_AVAILABLE = True
    logger.info("✅ ChromaDB library loaded successfully")
except ImportError as e:
    logger.warning(f"ChromaDB not available: {e}")
    CHROMA_AVAILABLE = False
    chromadb = None
    Settings = None  # Type hint placeholder
    Documents = None
    Embeddings = None
    IDs = None
    Metadatas = None


# Singleton instance
_chroma_client = None


class ChromaClient:
    """ChromaDB client wrapper for DARWIN."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        settings: Optional[Settings] = None
    ):
        """
        Initialize ChromaDB client.

        Args:
            host: ChromaDB host (default: localhost)
            port: ChromaDB port (default: 8000)
            settings: Optional ChromaDB settings
        """
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB not installed. Install with: pip install chromadb")

        self.host = host
        self.port = port

        # Create client (ChromaDB simple mode - tenant/database tem bugs em 0.5.0)
        try:
            self.client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=settings or Settings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
            )
            logger.info(f"✅ ChromaDB client connected to {host}:{port}")

            # Test connection
            heartbeat = self.client.heartbeat()
            logger.info(f"✅ ChromaDB heartbeat: {heartbeat}")

        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise

    def get_or_create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding_function: Optional[Any] = None
    ):
        """
        Get or create a collection.

        Args:
            name: Collection name
            metadata: Optional metadata
            embedding_function: Optional embedding function (uses default if None)

        Returns:
            Collection object
        """
        try:
            # Use default embedding function if none provided
            if embedding_function is None:
                try:
                    from chromadb.utils import embedding_functions
                    embedding_function = embedding_functions.DefaultEmbeddingFunction()
                    logger.info("Using ChromaDB default embedding function")
                except Exception as e:
                    logger.warning(f"Could not load default embedding function: {e}")
                    embedding_function = None

            collection = self.client.get_or_create_collection(
                name=name,
                metadata=metadata,
                embedding_function=embedding_function
            )
            logger.info(f"✅ Collection '{name}' ready (count: {collection.count()})")
            return collection

        except Exception as e:
            logger.error(f"Failed to get/create collection '{name}': {e}")
            raise

    def list_collections(self) -> List[str]:
        """List all collection names."""
        try:
            collections = self.client.list_collections()
            return [c.name for c in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        try:
            self.client.delete_collection(name=name)
            logger.info(f"✅ Collection '{name}' deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection '{name}': {e}")
            return False

    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> bool:
        """
        Add documents to a collection.

        Args:
            collection_name: Name of collection
            documents: List of document texts
            ids: List of unique IDs
            metadatas: Optional list of metadata dicts
            embeddings: Optional pre-computed embeddings

        Returns:
            True if successful
        """
        try:
            collection = self.get_or_create_collection(collection_name)

            if embeddings:
                collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            else:
                collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )

            logger.info(f"✅ Added {len(documents)} documents to '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents to '{collection_name}': {e}")
            return False

    def query(
        self,
        collection_name: str,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: List[str] = ["documents", "metadatas", "distances"]
    ) -> Dict[str, Any]:
        """
        Query a collection.

        Args:
            collection_name: Name of collection
            query_texts: Optional query texts (will be embedded)
            query_embeddings: Optional pre-computed query embeddings
            n_results: Number of results to return
            where: Optional metadata filter
            where_document: Optional document content filter
            include: What to include in results

        Returns:
            Query results dict
        """
        try:
            collection = self.get_or_create_collection(collection_name)

            results = collection.query(
                query_texts=query_texts,
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include
            )

            logger.info(f"✅ Query returned {len(results.get('ids', [[]])[0])} results")
            return results

        except Exception as e:
            logger.error(f"Failed to query collection '{collection_name}': {e}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def get_documents(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: List[str] = ["documents", "metadatas"]
    ) -> Dict[str, Any]:
        """
        Get documents from a collection.

        Args:
            collection_name: Name of collection
            ids: Optional specific IDs to retrieve
            where: Optional metadata filter
            limit: Optional limit
            offset: Optional offset
            include: What to include

        Returns:
            Documents dict
        """
        try:
            collection = self.get_or_create_collection(collection_name)

            results = collection.get(
                ids=ids,
                where=where,
                limit=limit,
                offset=offset,
                include=include
            )

            logger.info(f"✅ Retrieved {len(results.get('ids', []))} documents")
            return results

        except Exception as e:
            logger.error(f"Failed to get documents from '{collection_name}': {e}")
            return {"ids": [], "documents": [], "metadatas": []}

    def delete_documents(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Delete documents from a collection.

        Args:
            collection_name: Name of collection
            ids: Optional IDs to delete
            where: Optional metadata filter

        Returns:
            True if successful
        """
        try:
            collection = self.get_or_create_collection(collection_name)

            collection.delete(
                ids=ids,
                where=where
            )

            logger.info(f"✅ Deleted documents from '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to delete from '{collection_name}': {e}")
            return False

    def update_documents(
        self,
        collection_name: str,
        ids: List[str],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> bool:
        """
        Update documents in a collection.

        Args:
            collection_name: Name of collection
            ids: IDs to update
            documents: Optional new documents
            metadatas: Optional new metadatas
            embeddings: Optional new embeddings

        Returns:
            True if successful
        """
        try:
            collection = self.get_or_create_collection(collection_name)

            collection.update(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )

            logger.info(f"✅ Updated {len(ids)} documents in '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to update documents in '{collection_name}': {e}")
            return False

    def peek(self, collection_name: str, limit: int = 10) -> Dict[str, Any]:
        """
        Peek at first documents in collection.

        Args:
            collection_name: Collection name
            limit: Number of documents to peek

        Returns:
            Documents dict
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            results = collection.peek(limit=limit)
            logger.info(f"✅ Peeked {len(results.get('ids', []))} documents")
            return results

        except Exception as e:
            logger.error(f"Failed to peek collection '{collection_name}': {e}")
            return {"ids": [], "documents": [], "metadatas": []}

    def count(self, collection_name: str) -> int:
        """
        Count documents in collection.

        Args:
            collection_name: Collection name

        Returns:
            Document count
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            count = collection.count()
            logger.info(f"✅ Collection '{collection_name}' has {count} documents")
            return count

        except Exception as e:
            logger.error(f"Failed to count collection '{collection_name}': {e}")
            return 0


def get_chroma_client(
    host: Optional[str] = None,
    port: Optional[int] = None
) -> Optional[ChromaClient]:
    """
    Get singleton ChromaDB client instance.

    Args:
        host: ChromaDB host
        port: ChromaDB port

    Returns:
        ChromaClient instance or None if unavailable
    """
    global _chroma_client

    if not CHROMA_AVAILABLE:
        logger.warning("ChromaDB not available - install with: pip install chromadb")
        return None

    resolved_host = host or os.getenv("CHROMA_HOST", "localhost")
    try:
        resolved_port = int(port if port is not None else os.getenv("CHROMA_PORT", "8000"))
    except ValueError:
        logger.warning("Invalid CHROMA_PORT value. Falling back to 8000")
        resolved_port = 8000

    if _chroma_client is None:
        try:
            _chroma_client = ChromaClient(host=resolved_host, port=resolved_port)
            logger.info("✅ ChromaDB client singleton created")
        except Exception as e:
            logger.error(f"Failed to create ChromaDB client: {e}")
            return None

    return _chroma_client


def reset_chroma_client():
    """Reset singleton client (useful for testing)."""
    global _chroma_client
    _chroma_client = None
    logger.info("🔄 ChromaDB client singleton reset")
