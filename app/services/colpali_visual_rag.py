"""
ColPali: Visual RAG for Scientific PDFs

ColPali uses Vision Language Models for end-to-end document retrieval,
treating pages as images. Superior to traditional OCR+text for complex layouts.

Key Features:
- Direct visual embedding of PDF pages
- No OCR required
- Handles tables, figures, equations
- Late interaction retrieval
- State-of-the-art for academic papers

Paper: https://arxiv.org/abs/2407.01449
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import asyncio
from enum import Enum

try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False
    logging.warning("pdf2image not installed. Install with: pip install pdf2image")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logging.warning("Pillow not installed. Install with: pip install Pillow")

try:
    import torch
    from transformers import AutoProcessor, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("transformers not installed. Install with: pip install transformers torch")

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, NamedVector

logger = logging.getLogger(__name__)


class ColPaliModel(str, Enum):
    """Available ColPali model variants"""
    COLPALI_V1 = "vidore/colpali"
    COLPALI_V1_2 = "vidore/colpali-v1.2"
    COLQWEN2 = "vidore/colqwen2-v0.1"  # Qwen2-VL based, very strong


@dataclass
class ColPaliConfig:
    """Configuration for ColPali Visual RAG"""
    model_name: ColPaliModel = ColPaliModel.COLPALI_V1_2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4
    max_length: int = 512
    dpi: int = 150  # PDF rendering DPI
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "colpali_documents"


class ColPaliVisualRAG:
    """
    ColPali Visual RAG implementation.
    
    Uses Vision Language Models for direct visual embedding of PDF pages,
    enabling retrieval without OCR.
    """
    
    def __init__(self, config: ColPaliConfig):
        self.config = config
        
        if not all([HAS_PDF2IMAGE, HAS_PIL, HAS_TRANSFORMERS]):
            raise ImportError(
                "Missing dependencies. Install with:\n"
                "pip install pdf2image Pillow transformers torch"
            )
        
        # Load model and processor
        logger.info(f"Loading ColPali model: {config.model_name}")
        self.processor = AutoProcessor.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name).to(config.device)
        self.model.eval()
        
        # Qdrant client
        self.qdrant = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)
        self._ensure_collection()
        
        logger.info(f"ColPali initialized on {config.device}")
    
    def _ensure_collection(self):
        """Ensure Qdrant collection exists"""
        try:
            self.qdrant.get_collection(self.config.collection_name)
            logger.info(f"Collection '{self.config.collection_name}' exists")
        except Exception:
            # Create collection with multivector support
            # ColPali uses late interaction, so we need multiple vectors per page
            self.qdrant.create_collection(
                collection_name=self.config.collection_name,
                vectors_config={
                    "image": VectorParams(
                        size=128,  # ColPali embedding dimension
                        distance=Distance.COSINE
                    )
                }
            )
            logger.info(f"Created collection '{self.config.collection_name}'")
    
    def _pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        """Convert PDF pages to images"""
        logger.debug(f"Converting PDF to images: {pdf_path}")
        images = convert_from_path(
            str(pdf_path),
            dpi=self.config.dpi,
            fmt='RGB'
        )
        logger.debug(f"Converted {len(images)} pages")
        return images
    
    @torch.no_grad()
    def _embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        """Generate ColPali embeddings for images"""
        embeddings = []
        
        for i in range(0, len(images), self.config.batch_size):
            batch = images[i:i + self.config.batch_size]
            
            # Process images
            inputs = self.processor(
                images=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            ).to(self.config.device)
            
            # Generate embeddings
            outputs = self.model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # Pool
            embeddings.append(batch_embeddings.cpu())
        
        return torch.cat(embeddings, dim=0)
    
    @torch.no_grad()
    def _embed_query(self, query: str) -> torch.Tensor:
        """Generate ColPali embedding for text query"""
        inputs = self.processor(
            text=query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        ).to(self.config.device)
        
        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)  # Pool
        return embedding.cpu()
    
    def ingest_pdf(
        self,
        pdf_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest a PDF into ColPali Visual RAG.
        
        Args:
            pdf_path: Path to PDF file
            metadata: Optional metadata (title, authors, etc)
        
        Returns:
            Ingestion statistics
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Ingesting PDF: {pdf_path}")
        
        # Convert to images
        images = self._pdf_to_images(pdf_path)
        num_pages = len(images)
        
        # Generate embeddings
        embeddings = self._embed_images(images)
        
        # Store in Qdrant
        points = []
        for page_idx, (image, embedding) in enumerate(zip(images, embeddings)):
            point_metadata = {
                "pdf_path": str(pdf_path),
                "page": page_idx + 1,
                "total_pages": num_pages,
                **(metadata or {})
            }
            
            point = PointStruct(
                id=f"{pdf_path.stem}_page_{page_idx + 1}",
                vector={"image": embedding.numpy().tolist()},
                payload=point_metadata
            )
            points.append(point)
        
        self.qdrant.upsert(
            collection_name=self.config.collection_name,
            points=points
        )
        
        stats = {
            "pdf_path": str(pdf_path),
            "pages_ingested": num_pages,
            "embeddings_generated": len(embeddings),
            "points_stored": len(points)
        }
        
        logger.info(f"Ingestion complete: {stats}")
        return stats
    
    def ingest_directory(
        self,
        directory: Path,
        pattern: str = "*.pdf",
        metadata_fn: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Ingest all PDFs in a directory.
        
        Args:
            directory: Directory containing PDFs
            pattern: Glob pattern for PDF files
            metadata_fn: Optional function to extract metadata from filename
        
        Returns:
            Batch ingestion statistics
        """
        pdf_files = list(directory.glob(pattern))
        logger.info(f"Found {len(pdf_files)} PDFs in {directory}")
        
        total_pages = 0
        failed = []
        
        for pdf_path in pdf_files:
            try:
                metadata = metadata_fn(pdf_path) if metadata_fn else None
                stats = self.ingest_pdf(pdf_path, metadata)
                total_pages += stats["pages_ingested"]
            except Exception as e:
                logger.error(f"Failed to ingest {pdf_path}: {e}")
                failed.append(str(pdf_path))
        
        return {
            "total_pdfs": len(pdf_files),
            "successful": len(pdf_files) - len(failed),
            "failed": len(failed),
            "failed_files": failed,
            "total_pages": total_pages
        }
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant PDF pages using visual RAG.
        
        Args:
            query: Text query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
        
        Returns:
            List of search results with scores and metadata
        """
        logger.debug(f"Searching for: {query}")
        
        # Embed query
        query_embedding = self._embed_query(query)
        
        # Search in Qdrant
        search_results = self.qdrant.search(
            collection_name=self.config.collection_name,
            query_vector=("image", query_embedding.numpy().tolist()),
            limit=top_k,
            query_filter=filter_metadata
        )
        
        # Format results
        results = []
        for hit in search_results:
            result = {
                "score": hit.score,
                "pdf_path": hit.payload.get("pdf_path"),
                "page": hit.payload.get("page"),
                "total_pages": hit.payload.get("total_pages"),
                "metadata": {k: v for k, v in hit.payload.items() 
                           if k not in ["pdf_path", "page", "total_pages"]}
            }
            results.append(result)
        
        logger.debug(f"Found {len(results)} results")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        collection_info = self.qdrant.get_collection(self.config.collection_name)
        return {
            "collection_name": self.config.collection_name,
            "total_pages": collection_info.points_count,
            "model": self.config.model_name,
            "device": self.config.device
        }


# Factory function
def get_colpali_rag(config: Optional[ColPaliConfig] = None) -> ColPaliVisualRAG:
    """Factory function to get ColPali Visual RAG instance"""
    if config is None:
        config = ColPaliConfig()
    return ColPaliVisualRAG(config)


# Example usage
if __name__ == "__main__":
    # Example: Ingest scientific papers
    config = ColPaliConfig(
        model_name=ColPaliModel.COLPALI_V1_2,
        device="cuda",
        dpi=150
    )
    
    colpali = get_colpali_rag(config)
    
    # Ingest single PDF
    pdf_path = Path("papers/biomaterials_scaffold_design.pdf")
    if pdf_path.exists():
        stats = colpali.ingest_pdf(
            pdf_path,
            metadata={
                "title": "Advanced Scaffold Design for Tissue Engineering",
                "year": 2024,
                "domain": "biomaterials"
            }
        )
        print(f"Ingested: {stats}")
    
    # Search
    results = colpali.search(
        "What is the optimal pore size for bone scaffolds?",
        top_k=3
    )
    
    print("\n--- Search Results ---")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['pdf_path']} (Page {result['page']}) - Score: {result['score']:.3f}")
    
    # Stats
    print("\n--- Stats ---")
    print(colpali.get_stats())

