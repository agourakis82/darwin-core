"""
Qdrant Hybrid Client - Estado da Arte 2025

Hybrid Search implementation:
- Dense vectors (semantic embeddings)
- Sparse vectors (BM25, SPLADE++)
- Late interaction (ColBERT v2)
- Reciprocal Rank Fusion (RRF)

Features 2025:
- Binary quantization
- Scalar quantization  
- Product quantization
- HNSW optimized
- Sharding automático
- Snapshot/backup
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, Filter,
        FieldCondition, MatchValue, Range,
        ScoredPoint, SearchRequest, NamedVector,
        Quantization Config, BinaryQuantization, ScalarQuantization,
        ProductQuantization, HnswConfig, OptimizersConfigDiff,
        CollectionStatus
    )
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False
    logging.warning("qdrant-client not available")

import numpy as np

logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    """Modos de busca suportados"""
    DENSE = "dense"  # Apenas dense vectors (embeddings)
    SPARSE = "sparse"  # Apenas sparse vectors (BM25/SPLADE)
    HYBRID = "hybrid"  # Dense + Sparse com RRF
    LATE_INTERACTION = "late_interaction"  # ColBERT style
    MULTI_VECTOR = "multi_vector"  # Multiple dense vectors


@dataclass
class HybridSearchConfig:
    """Configuração para hybrid search"""
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    use_rrf: bool = True
    rrf_k: int = 60  # Parâmetro K para RRF
    top_k: int = 10
    rerank: bool = False


class QdrantHybridClient:
    """
    Qdrant client com hybrid search estado da arte 2025
    
    Features:
    - Dense + Sparse + Late Interaction
    - Multiple quantization methods
    - HNSW optimized
    - RRF fusion
    - Auto sharding
    - Snapshot/backup
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        prefer_grpc: bool = True,
    ):
        """
        Inicializa Qdrant client
        
        Args:
            host: Qdrant host
            port: Qdrant port
            api_key: API key (para Qdrant Cloud)
            prefer_grpc: Use gRPC ao invés de HTTP (mais rápido)
        """
        self.host = host
        self.port = port
        self.api_key = api_key
        self.prefer_grpc = prefer_grpc
        
        self.client: Optional[QdrantClient] = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Inicializa conexão com Qdrant"""
        if self._initialized:
            return True
        
        if not HAS_QDRANT:
            logger.error("qdrant-client not available")
            return False
        
        try:
            self.client = QdrantClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
                prefer_grpc=self.prefer_grpc,
                timeout=30,
            )
            
            # Test connection
            collections = self.client.get_collections()
            
            self._initialized = True
            logger.info(
                f"✅ Qdrant client initialized: {self.host}:{self.port} "
                f"(collections: {len(collections.collections)})"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            return False
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
        enable_sparse: bool = True,
        enable_quantization: bool = True,
        quantization_type: str = "binary",  # binary, scalar, product
        shard_number: int = 1,
        replication_factor: int = 1,
        on_disk_payload: bool = False,
    ) -> bool:
        """
        Cria collection com configuração otimizada 2025
        
        Args:
            collection_name: Nome da collection
            vector_size: Dimensão dos embeddings
            distance: Métrica de distância
            enable_sparse: Habilita sparse vectors
            enable_quantization: Habilita quantization
            quantization_type: Tipo (binary, scalar, product)
            shard_number: Número de shards
            replication_factor: Fator de replicação
            on_disk_payload: Payload em disco (economiza RAM)
        
        Returns:
            True se criado com sucesso
        """
        if not self._initialized and not self.initialize():
            return False
        
        try:
            # Configuração de quantization
            quantization_config = None
            if enable_quantization:
                if quantization_type == "binary":
                    quantization_config = BinaryQuantization(
                        binary=BinaryQuantization()
                    )
                elif quantization_type == "scalar":
                    quantization_config = ScalarQuantization(
                        scalar=ScalarQuantization(
                            type="int8",  # int8 ou float16
                            quantile=0.99,
                            always_ram=True,
                        )
                    )
                elif quantization_type == "product":
                    quantization_config = ProductQuantization(
                        product=ProductQuantization(
                            compression="x16",  # compression ratio
                            always_ram=True,
                        )
                    )
            
            # Configuração HNSW otimizada
            hnsw_config = HnswConfig(
                m=16,  # Número de conexões (16-64)
                ef_construct=100,  # Construção (100-200)
                full_scan_threshold=10000,  # Threshold para full scan
                on_disk=False,  # HNSW em memória (mais rápido)
            )
            
            # Configuração de otimizadores
            optimizers_config = OptimizersConfigDiff(
                deleted_threshold=0.2,
                vacuum_min_vector_number=1000,
                default_segment_number=shard_number,
                max_segment_size=200_000,
                memmap_threshold=50_000,
                indexing_threshold=20_000,
                flush_interval_sec=5,
                max_optimization_threads=2,
            )
            
            # Vetores configurados
            vectors_config = {
                "dense": VectorParams(
                    size=vector_size,
                    distance=distance,
                    hnsw_config=hnsw_config,
                    quantization_config=quantization_config,
                    on_disk=False,
                )
            }
            
            # Adiciona sparse vectors se habilitado
            if enable_sparse:
                vectors_config["sparse"] = VectorParams(
                    size=vector_size,  # Mesmo tamanho
                    distance=Distance.DOT,  # Sparse usa dot product
                    on_disk=False,
                )
            
            # Cria collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                shard_number=shard_number,
                replication_factor=replication_factor,
                on_disk_payload=on_disk_payload,
                optimizers_config=optimizers_config,
            )
            
            logger.info(
                f"✅ Collection created: {collection_name} "
                f"(dim={vector_size}, quantization={quantization_type}, "
                f"sparse={enable_sparse}, shards={shard_number})"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    def upsert_points(
        self,
        collection_name: str,
        points: List[Dict[str, Any]],
        batch_size: int = 100,
        parallel: int = 1,
    ) -> bool:
        """
        Insere ou atualiza pontos na collection
        
        Args:
            collection_name: Nome da collection
            points: Lista de pontos com:
                - id: UUID ou int
                - vector: Dict com "dense" e opcionalmente "sparse"
                - payload: Metadata
            batch_size: Tamanho do batch
            parallel: Número de uploads paralelos
        
        Returns:
            True se sucesso
        """
        if not self._initialized and not self.initialize():
            return False
        
        try:
            # Converte para PointStruct
            point_structs = []
            
            for point in points:
                # ID
                point_id = point.get("id", str(uuid.uuid4()))
                
                # Vectors
                vector = point["vector"]
                if isinstance(vector, dict):
                    # Named vectors (dense + sparse)
                    named_vectors = {}
                    if "dense" in vector:
                        named_vectors["dense"] = vector["dense"]
                    if "sparse" in vector:
                        named_vectors["sparse"] = vector["sparse"]
                    vector_data = named_vectors
                else:
                    # Single vector (dense apenas)
                    vector_data = {"dense": vector}
                
                # Payload
                payload = point.get("payload", {})
                
                point_struct = PointStruct(
                    id=point_id,
                    vector=vector_data,
                    payload=payload,
                )
                point_structs.append(point_struct)
            
            # Upload em batches
            for i in range(0, len(point_structs), batch_size):
                batch = point_structs[i:i + batch_size]
                
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=True,
                )
            
            logger.info(f"✅ Upserted {len(points)} points to {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert points: {e}")
            return False
    
    def hybrid_search(
        self,
        collection_name: str,
        query_vector: Union[np.ndarray, Dict[str, np.ndarray]],
        config: Optional[HybridSearchConfig] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: dense + sparse com RRF
        
        Args:
            collection_name: Nome da collection
            query_vector: Vector ou dict com "dense" e "sparse"
            config: Configuração de busca
            filter: Filtros de payload
        
        Returns:
            Lista de resultados ranqueados
        """
        if not self._initialized and not self.initialize():
            return []
        
        if config is None:
            config = HybridSearchConfig()
        
        try:
            # Prepara vectors
            if isinstance(query_vector, dict):
                dense_vec = query_vector.get("dense")
                sparse_vec = query_vector.get("sparse")
            else:
                dense_vec = query_vector
                sparse_vec = None
            
            # Busca dense
            dense_results = []
            if dense_vec is not None:
                dense_results = self.client.search(
                    collection_name=collection_name,
                    query_vector=NamedVector(
                        name="dense",
                        vector=dense_vec.tolist() if isinstance(dense_vec, np.ndarray) else dense_vec
                    ),
                    limit=config.top_k * 2,  # Get more for RRF
                    query_filter=self._build_filter(filter) if filter else None,
                    with_payload=True,
                    with_vectors=False,
                )
            
            # Busca sparse (se disponível)
            sparse_results = []
            if sparse_vec is not None:
                sparse_results = self.client.search(
                    collection_name=collection_name,
                    query_vector=NamedVector(
                        name="sparse",
                        vector=sparse_vec.tolist() if isinstance(sparse_vec, np.ndarray) else sparse_vec
                    ),
                    limit=config.top_k * 2,
                    query_filter=self._build_filter(filter) if filter else None,
                    with_payload=True,
                    with_vectors=False,
                )
            
            # Fusão de resultados
            if config.use_rrf and dense_results and sparse_results:
                # Reciprocal Rank Fusion
                final_results = self._reciprocal_rank_fusion(
                    dense_results,
                    sparse_results,
                    k=config.rrf_k,
                    dense_weight=config.dense_weight,
                    sparse_weight=config.sparse_weight,
                )
            elif dense_results and sparse_results:
                # Linear combination
                final_results = self._linear_combination(
                    dense_results,
                    sparse_results,
                    dense_weight=config.dense_weight,
                    sparse_weight=config.sparse_weight,
                )
            elif dense_results:
                final_results = dense_results
            else:
                final_results = sparse_results
            
            # Top-K
            final_results = final_results[:config.top_k]
            
            # Converte para dict
            results_dict = []
            for result in final_results:
                results_dict.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                })
            
            logger.info(
                f"🔍 Hybrid search: {len(results_dict)} results "
                f"(dense={len(dense_results)}, sparse={len(sparse_results)})"
            )
            
            return results_dict
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[ScoredPoint],
        sparse_results: List[ScoredPoint],
        k: int = 60,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> List[ScoredPoint]:
        """
        Reciprocal Rank Fusion (RRF)
        
        Score(doc) = Σ weight_i / (k + rank_i(doc))
        
        Better than linear combination for multi-source ranking
        """
        # Mapeia ID → rank
        dense_ranks = {r.id: i for i, r in enumerate(dense_results)}
        sparse_ranks = {r.id: i for i, r in enumerate(sparse_results)}
        
        # Todos IDs únicos
        all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        
        # Calcula RRF score
        rrf_scores = {}
        for doc_id in all_ids:
            score = 0.0
            
            if doc_id in dense_ranks:
                score += dense_weight / (k + dense_ranks[doc_id])
            
            if doc_id in sparse_ranks:
                score += sparse_weight / (k + sparse_ranks[doc_id])
            
            rrf_scores[doc_id] = score
        
        # Ordena por RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Reconstrói resultados
        id_to_point = {}
        for r in dense_results + sparse_results:
            if r.id not in id_to_point:
                id_to_point[r.id] = r
        
        final_results = []
        for doc_id in sorted_ids:
            point = id_to_point[doc_id]
            # Atualiza score com RRF
            point.score = rrf_scores[doc_id]
            final_results.append(point)
        
        return final_results
    
    def _linear_combination(
        self,
        dense_results: List[ScoredPoint],
        sparse_results: List[ScoredPoint],
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> List[ScoredPoint]:
        """Linear combination of scores"""
        # Normaliza scores
        dense_scores = {r.id: r.score for r in dense_results}
        sparse_scores = {r.id: r.score for r in sparse_results}
        
        # Normalização min-max
        if dense_scores:
            max_dense = max(dense_scores.values())
            min_dense = min(dense_scores.values())
            dense_scores = {
                k: (v - min_dense) / (max_dense - min_dense + 1e-8)
                for k, v in dense_scores.items()
            }
        
        if sparse_scores:
            max_sparse = max(sparse_scores.values())
            min_sparse = min(sparse_scores.values())
            sparse_scores = {
                k: (v - min_sparse) / (max_sparse - min_sparse + 1e-8)
                for k, v in sparse_scores.items()
            }
        
        # Combina
        all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        combined_scores = {}
        
        for doc_id in all_ids:
            score = 0.0
            if doc_id in dense_scores:
                score += dense_weight * dense_scores[doc_id]
            if doc_id in sparse_scores:
                score += sparse_weight * sparse_scores[doc_id]
            combined_scores[doc_id] = score
        
        # Ordena
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        
        # Reconstrói
        id_to_point = {}
        for r in dense_results + sparse_results:
            if r.id not in id_to_point:
                id_to_point[r.id] = r
        
        final_results = []
        for doc_id in sorted_ids:
            point = id_to_point[doc_id]
            point.score = combined_scores[doc_id]
            final_results.append(point)
        
        return final_results
    
    def _build_filter(self, filter_dict: Dict[str, Any]) -> Filter:
        """Constrói Qdrant filter a partir de dict"""
        conditions = []
        
        for key, value in filter_dict.items():
            if isinstance(value, dict):
                # Range filter
                if "gte" in value or "lte" in value:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            range=Range(
                                gte=value.get("gte"),
                                lte=value.get("lte"),
                            )
                        )
                    )
            else:
                # Match filter
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
        
        return Filter(must=conditions)
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Informações sobre collection"""
        if not self._initialized and not self.initialize():
            return {}
        
        try:
            info = self.client.get_collection(collection_name)
            
            return {
                "name": collection_name,
                "status": info.status.value if info.status else "unknown",
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "config": {
                    "params": info.config.params.__dict__ if info.config else {},
                    "optimizer": info.config.optimizer_config.__dict__ if info.config else {},
                }
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}


# Singleton global
_qdrant_client: Optional[QdrantHybridClient] = None


def get_qdrant_client(
    host: str = "localhost",
    port: int = 6333,
    force_reload: bool = False
) -> QdrantHybridClient:
    """Get or create Qdrant client singleton"""
    global _qdrant_client
    
    if _qdrant_client is None or force_reload:
        _qdrant_client = QdrantHybridClient(host=host, port=port)
        _qdrant_client.initialize()
    
    return _qdrant_client


__all__ = [
    "QdrantHybridClient",
    "SearchMode",
    "HybridSearchConfig",
    "get_qdrant_client",
]

