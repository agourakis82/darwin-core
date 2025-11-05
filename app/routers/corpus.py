"""
Scientific Corpus Router - Ingest Large Scientific Texts

Permite ingerir:
- PDFs científicos (papers, livros, teses)
- Texto puro (markdown, txt)
- Diretórios completos (bibliotecas de papers)

Sistema automaticamente:
1. Chunka texto inteligentemente
2. Salva no ChromaDB (RAG++)
3. Usa para treinar modelos especializados
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import tempfile
import shutil

from ..services.auto_training_pipeline import get_corpus_ingestion, get_auto_training_pipeline

router = APIRouter(prefix="/api/v1/corpus", tags=["Corpus"])


class TextIngestRequest(BaseModel):
    """Request to ingest plain text"""
    text: str = Field(..., min_length=100)
    domain: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    doi: Optional[str] = None
    tags: List[str] = Field(default=[])


@router.post("/ingest/text")
async def ingest_text(request: TextIngestRequest):
    """
    Ingest large text into system
    
    Example:
    {
      "text": "Long scientific paper text...",
      "domain": "biomaterials",
      "title": "Scaffold Optimization Study",
      "authors": ["Smith J.", "Doe A."],
      "tags": ["scaffold", "PCL", "biocompatibility"]
    }
    
    System will:
    - Chunk text intelligently (~1000 chars per chunk)
    - Save to ChromaDB for RAG++
    - Trigger model fine-tuning when threshold reached
    """
    ingestion = get_corpus_ingestion()
    
    metadata = {
        "title": request.title,
        "authors": request.authors,
        "doi": request.doi,
        "tags": request.tags
    }
    
    result = await ingestion.ingest_text(
        text=request.text,
        domain=request.domain,
        metadata=metadata
    )
    
    # Check if training should be triggered
    pipeline = get_auto_training_pipeline()
    status = await pipeline.get_training_status()
    
    domain_status = status["next_training"].get(request.domain, {})
    
    return {
        **result,
        "training_status": {
            "conversations_for_domain": domain_status.get("conversations", 0),
            "threshold": domain_status.get("threshold", 100),
            "ready_for_training": domain_status.get("ready", False)
        }
    }


@router.post("/ingest/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    domain: str = Form(...),
    extract_metadata: bool = Form(default=True)
):
    """
    Upload and ingest PDF file
    
    Extracts text, metadata, chunks, and saves to memory
    """
    ingestion = get_corpus_ingestion()
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    
    try:
        result = await ingestion.ingest_pdf(
            pdf_path=tmp_path,
            domain=domain,
            extract_metadata=extract_metadata
        )
        
        return result
        
    finally:
        # Cleanup
        tmp_path.unlink()


@router.post("/ingest/batch")
async def ingest_batch_texts(texts: List[TextIngestRequest]):
    """
    Ingest multiple texts in batch
    
    Useful for large corpus ingestion (e.g., 100+ papers)
    """
    ingestion = get_corpus_ingestion()
    
    results = []
    for req in texts:
        try:
            result = await ingestion.ingest_text(
                text=req.text,
                domain=req.domain,
                metadata={
                    "title": req.title,
                    "authors": req.authors,
                    "doi": req.doi,
                    "tags": req.tags
                }
            )
            results.append({"success": True, **result})
        except Exception as e:
            results.append({
                "success": False,
                "error": str(e),
                "domain": req.domain
            })
    
    total_chunks = sum(r.get("chunks_created", 0) for r in results)
    
    return {
        "success": True,
        "texts_processed": len(results),
        "total_chunks": total_chunks,
        "results": results
    }


@router.post("/train-on-corpus")
async def train_on_corpus(domain: str, force: bool = False):
    """
    Trigger training using all corpus data for domain
    
    This creates a specialized model trained on scientific literature!
    
    Example: POST /api/v1/corpus/train-on-corpus?domain=biomaterials
    
    Creates: darwin-biomaterials-local-v2 (or next version)
    """
    pipeline = get_auto_training_pipeline()
    
    result = await pipeline.trigger_manual_training(domain, force=force)
    
    return result


@router.get("/training-status")
async def get_training_status():
    """
    Get status of automatic training pipeline
    
    Shows:
    - How many conversations per domain
    - When next training will happen
    - Training history
    - Model versions
    """
    pipeline = get_auto_training_pipeline()
    
    status = await pipeline.get_training_status()
    
    return status


@router.post("/start-auto-training")
async def start_auto_training():
    """
    Start automatic training pipeline
    
    Monitors conversations and trains models automatically
    """
    pipeline = get_auto_training_pipeline()
    
    if pipeline._running:
        return {"message": "Auto-training already running"}
    
    await pipeline.start()
    
    return {
        "success": True,
        "message": "Auto-training pipeline started",
        "threshold": pipeline.training_threshold,
        "interval_days": pipeline.retrain_interval_days
    }


@router.post("/stop-auto-training")
async def stop_auto_training():
    """Stop automatic training pipeline"""
    pipeline = get_auto_training_pipeline()
    
    await pipeline.stop()
    
    return {"success": True, "message": "Auto-training pipeline stopped"}

