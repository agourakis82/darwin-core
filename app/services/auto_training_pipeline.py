"""
Automatic Training Pipeline - Continuous Model Evolution

Sistema que treina modelos automaticamente baseado em:
1. Suas conversas (MCP, Custom GPT, Claude)
2. Textos científicos ingeridos
3. Feedback de uso (continuous learning)

O sistema:
- Monitora memória semântica
- Detecta quando há dados suficientes para treinar
- Fine-tune modelos automaticamente
- Versiona modelos (v1, v2, v3...)
- Avalia melhoria (A/B testing)
- Deploy automático do melhor modelo
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict
import hashlib

from .semantic_memory import get_semantic_memory_service, Domain
from .continuous_learning import get_continuous_learning_engine
from .model_training import get_model_trainer
from .model_registry import get_model_registry
from .pulsar_client import get_pulsar_client, TOPICS

logger = logging.getLogger("darwin.auto_training")


class AutoTrainingPipeline:
    """
    Pipeline de treinamento automático
    
    Workflow:
    1. Monitora interações acumuladas por domínio
    2. Quando threshold atingido (ex: 100 conversas): dispara fine-tuning
    3. Treina novo modelo com dados do domínio
    4. Avalia modelo (perplexity, user feedback)
    5. Se melhor que anterior: deploy automático
    6. Se não: descarta e mantém modelo atual
    
    Resultado: Modelos evoluem continuamente com SEU uso!
    """
    
    def __init__(
        self,
        training_threshold: int = 100,  # Mínimo de conversas para treinar
        retrain_interval_days: int = 7,  # Frequência de re-treino
        enable_auto_deploy: bool = True,  # Deploy automático se melhor
    ):
        self.training_threshold = training_threshold
        self.retrain_interval_days = retrain_interval_days
        self.enable_auto_deploy = enable_auto_deploy
        
        # Tracking
        self.domain_conversation_count: Dict[str, int] = defaultdict(int)
        self.last_training_time: Dict[str, datetime] = {}
        self.model_versions: Dict[str, int] = defaultdict(int)
        
        # Training history
        self.training_history: List[Dict[str, Any]] = []
        
        # Running state
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start automatic training pipeline"""
        logger.info("🎓 Starting Automatic Training Pipeline")
        
        self._running = True
        
        # Start monitoring loop
        task = asyncio.create_task(self._monitoring_loop())
        self._tasks.append(task)
        
        # Subscribe to conversation events
        pulsar = get_pulsar_client()
        asyncio.create_task(
            pulsar.subscribe(
                TOPICS["continuous_learning"],
                "auto-training-sub",
                self._handle_conversation_event
            )
        )
        
        logger.info("✅ Auto-training pipeline started")
    
    async def stop(self):
        """Stop pipeline"""
        logger.info("🛑 Stopping Auto-training Pipeline")
        self._running = False
        
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("✅ Auto-training pipeline stopped")
    
    async def _handle_conversation_event(self, event: Dict[str, Any]):
        """
        Handle conversation event from Pulsar
        
        Increments count and triggers training if threshold reached
        """
        domain = event.get("domain", "general")
        
        # Increment count
        self.domain_conversation_count[domain] += 1
        
        count = self.domain_conversation_count[domain]
        
        logger.debug(f"📊 Domain {domain}: {count} conversations")
        
        # Check if should trigger training
        if count >= self.training_threshold and count % self.training_threshold == 0:
            logger.info(
                f"🎯 Training threshold reached for {domain}: {count} conversations"
            )
            
            # Trigger async training
            asyncio.create_task(self._trigger_training(domain))
    
    async def _monitoring_loop(self):
        """
        Monitoring loop - checks periodically if retraining is needed
        
        Even if threshold not reached, retrain every N days if there's new data
        """
        logger.info("👁️ Monitoring loop started")
        
        while self._running:
            try:
                # Check each domain
                for domain in Domain:
                    domain_str = domain.value
                    
                    # Check if enough time passed since last training
                    last_training = self.last_training_time.get(domain_str)
                    
                    if last_training:
                        days_since = (datetime.now(timezone.utc) - last_training).days
                        
                        if days_since >= self.retrain_interval_days:
                            count = self.domain_conversation_count.get(domain_str, 0)
                            
                            if count >= 20:  # At least 20 new conversations
                                logger.info(
                                    f"🔄 Time-based retraining for {domain_str}: "
                                    f"{days_since} days, {count} conversations"
                                )
                                asyncio.create_task(self._trigger_training(domain_str))
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
            
            # Check every hour
            await asyncio.sleep(3600)
        
        logger.info("👁️ Monitoring loop stopped")
    
    async def _trigger_training(self, domain: str):
        """
        Trigger training for a domain
        
        This runs in background and doesn't block
        """
        logger.info(f"🚀 Triggering training for domain: {domain}")
        
        try:
            # Determine base model
            base_model = self._select_base_model(domain)
            
            # Increment version
            self.model_versions[domain] += 1
            version = self.model_versions[domain]
            
            new_model_name = f"darwin-{domain}-local-v{version}"
            
            # Fine-tune
            trainer = get_model_trainer()
            
            result = await trainer.fine_tune_from_conversations(
                base_model=base_model,
                domain=domain,
                min_conversations=self.training_threshold,
                new_model_name=new_model_name
            )
            
            # Update last training time
            self.last_training_time[domain] = datetime.now(timezone.utc)
            
            # Reset counter
            self.domain_conversation_count[domain] = 0
            
            # Record in history
            self.training_history.append({
                "domain": domain,
                "model_name": new_model_name,
                "version": version,
                "training_samples": result.get("training_samples", 0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "base_model": base_model
            })
            
            logger.info(f"✅ Training complete for {domain}: {new_model_name}")
            
            # Evaluate and deploy if better
            if self.enable_auto_deploy:
                await self._evaluate_and_deploy(domain, new_model_name, version)
            
        except Exception as e:
            logger.error(f"❌ Training failed for {domain}: {e}")
    
    def _select_base_model(self, domain: str) -> str:
        """
        Select best base model for domain
        
        Strategy:
        - Biomaterials, Medical: Use Qwen 2.5 32B (best for science)
        - Code, Math: Use Qwen Coder
        - Philosophy, General: Use Llama 3.1
        """
        # CORRIGIDO: 32B é inviável com 20GB VRAM
        # Usar 7B/8B/14B com PEFT/QLoRA localmente
        # 32B apenas via job remoto (futuro)
        domain_to_base = {
            "biomaterials": "qwen2.5-coder:7b-instruct-q4_0",  # 7B viável com QLoRA
            "chemistry": "qwen2.5-coder:7b-instruct-q4_0",
            "medical": "llama3.1:8b-instruct-q4_0",
            "pharmacology": "llama3.1:8b-instruct-q4_0",
            "mathematics": "qwen2.5-coder:7b-instruct-q4_0",
            "physics": "llama3.1:8b-instruct-q4_0",
            "quantum": "llama3.1:8b-instruct-q4_0",
            "philosophy": "llama3.1:8b-instruct-q4_0",
            "code_generation": "qwen2.5-coder:7b-instruct-q4_0",
        }
        
        # TODO: Para 32B, enviar para job K8s dedicado com node GPU maior
        return domain_to_base.get(domain, "llama3.1:8b-instruct-q4_0")
    
    async def _evaluate_and_deploy(self, domain: str, new_model_name: str, version: int):
        """
        Evaluate new model vs current model
        
        If new model is better: deploy automatically
        If not: keep current model
        """
        logger.info(f"📊 Evaluating {new_model_name}...")
        
        # TODO: Implement evaluation (perplexity, A/B testing)
        # For now, always deploy new model
        
        # Get registry
        registry = get_model_registry()
        
        # Disable old version if exists
        old_model_id = f"darwin-{domain}-local:latest"
        if old_model_id in registry.models:
            registry.disable_model(old_model_id)
            logger.info(f"⏸️ Disabled old model: {old_model_id}")
        
        # New model already registered by trainer
        # Just log deployment
        logger.info(f"🚀 Auto-deployed: {new_model_name}")
        
        # Publish event
        pulsar = get_pulsar_client()
        await pulsar.publish(TOPICS["system_alerts"], {
            "severity": "info",
            "component": "auto_training",
            "message": f"New model deployed: {new_model_name}",
            "domain": domain,
            "version": version,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def get_training_status(self) -> Dict[str, Any]:
        """Get training pipeline status"""
        return {
            "running": self._running,
            "domain_counts": dict(self.domain_conversation_count),
            "last_training": {
                domain: time.isoformat()
                for domain, time in self.last_training_time.items()
            },
            "model_versions": dict(self.model_versions),
            "training_history": self.training_history[-10:],  # Last 10
            "next_training": {
                domain: {
                    "conversations": count,
                    "threshold": self.training_threshold,
                    "ready": count >= self.training_threshold
                }
                for domain, count in self.domain_conversation_count.items()
            }
        }


class ScientificCorpusIngestion:
    """
    Ingestão de grandes volumes de textos científicos
    
    Features:
    - Upload PDFs, TXTs, papers
    - Chunking inteligente (respeitando parágrafos, seções)
    - Extração de metadados (título, autores, DOI)
    - Indexação no ChromaDB
    - Uso para fine-tuning automático
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    async def ingest_text(
        self,
        text: str,
        domain: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest large text into system
        
        Args:
            text: Full text (pode ser 100+ páginas)
            domain: Domain classification
            metadata: Title, authors, DOI, etc
            
        Returns:
            Ingestion results
        """
        logger.info(f"📚 Ingesting text: {len(text)} characters, domain={domain}")
        
        # Step 1: Chunk text
        chunks = self._chunk_text(text)
        logger.info(f"✂️ Created {len(chunks)} chunks")
        
        # Step 2: Save each chunk to semantic memory
        semantic_memory = get_semantic_memory_service()
        
        saved_ids = []
        for i, chunk in enumerate(chunks):
            # Create metadata for chunk
            chunk_metadata = {
                **(metadata or {}),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "ingestion_type": "scientific_corpus"
            }
            
            # Save to memory
            conversation_id = f"corpus_{domain}_{hashlib.md5(chunk.encode()).hexdigest()[:12]}"
            
            from .semantic_memory import ConversationMetadata, Platform
            metadata_obj = ConversationMetadata(
                conversation_id=conversation_id,
                title=metadata.get("title", f"Corpus {domain} - chunk {i}") if metadata else f"Corpus chunk {i}",
                platform=Platform.OTHER,
                domain=Domain(domain) if domain in [d.value for d in Domain] else Domain.GENERAL,
                tags=metadata.get("tags", []) + ["corpus", "scientific"] if metadata else ["corpus", "scientific"],
                message_count=1
            )
            
            success, msg = semantic_memory.save_conversation(
                conversation_id=conversation_id,
                content=chunk,
                metadata=metadata_obj
            )
            
            if success:
                saved_ids.append(conversation_id)
        
        logger.info(f"✅ Ingested {len(saved_ids)} chunks into memory")
        
        # Step 3: Mark domain for training
        # This will trigger auto-training when threshold reached
        pulsar = get_pulsar_client()
        await pulsar.publish(TOPICS["continuous_learning"], {
            "event_type": "corpus_ingested",
            "domain": domain,
            "chunks": len(chunks),
            "total_chars": len(text),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "success": True,
            "chunks_created": len(chunks),
            "chunks_saved": len(saved_ids),
            "domain": domain,
            "total_characters": len(text)
        }
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Intelligent chunking que respeita parágrafos e seções
        
        Strategy:
        - Split by double newline (paragraphs)
        - Group paragraphs até atingir chunk_size
        - Overlap de chunk_overlap characters
        """
        # Split em parágrafos
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            if current_size + para_size > self.chunk_size and current_chunk:
                # Finalize chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                # Keep last paragraph for overlap
                current_chunk = [current_chunk[-1]] if current_chunk else []
                current_size = len(current_chunk[0]) if current_chunk else 0
            
            current_chunk.append(para)
            current_size += para_size
        
        # Add last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    async def ingest_pdf(
        self,
        pdf_path: Path,
        domain: str,
        extract_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest PDF file
        
        Extracts text, metadata (title, authors), and ingests
        """
        try:
            import PyPDF2
        except ImportError:
            raise RuntimeError("PyPDF2 not installed. Run: pip install PyPDF2")
        
        logger.info(f"📄 Ingesting PDF: {pdf_path}")
        
        # Extract text
        with open(pdf_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n\n"
            
            # Extract metadata
            metadata = {}
            if extract_metadata and pdf.metadata:
                metadata = {
                    "title": pdf.metadata.get("/Title", str(pdf_path.name)),
                    "author": pdf.metadata.get("/Author", "Unknown"),
                    "pages": len(pdf.pages),
                    "source": "pdf"
                }
        
        # Ingest text
        result = await self.ingest_text(text, domain, metadata)
        
        logger.info(f"✅ PDF ingested: {result['chunks_created']} chunks")
        
        return result
    
    async def ingest_directory(
        self,
        directory: Path,
        domain: str,
        file_pattern: str = "*.pdf"
    ) -> Dict[str, Any]:
        """
        Ingest all files in directory
        
        Useful for batch ingestion of paper collections
        """
        files = list(directory.glob(file_pattern))
        
        logger.info(f"📂 Ingesting {len(files)} files from {directory}")
        
        results = []
        for file_path in files:
            try:
                if file_path.suffix.lower() == ".pdf":
                    result = await self.ingest_pdf(file_path, domain)
                    results.append(result)
                elif file_path.suffix.lower() == ".txt":
                    text = file_path.read_text()
                    result = await self.ingest_text(text, domain, {"source": str(file_path)})
                    results.append(result)
                else:
                    logger.warning(f"⚠️ Skipping unsupported file: {file_path}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to ingest {file_path}: {e}")
        
        total_chunks = sum(r["chunks_created"] for r in results)
        
        logger.info(f"✅ Directory ingestion complete: {len(results)} files, {total_chunks} chunks")
        
        return {
            "success": True,
            "files_processed": len(results),
            "total_chunks": total_chunks,
            "results": results
        }
    
    async def trigger_manual_training(
        self,
        domain: str,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Trigger training manually (bypass threshold)
        
        Useful for testing or when you want immediate training
        """
        count = self.domain_conversation_count.get(domain, 0)
        
        if count < self.training_threshold and not force:
            raise ValueError(
                f"Not enough data: {count} conversations "
                f"(need {self.training_threshold}). Use force=True to override."
            )
        
        await self._trigger_training(domain)
        
        return {
            "success": True,
            "domain": domain,
            "message": f"Training triggered for {domain}"
        }


# Singletons
_auto_training_pipeline: Optional[AutoTrainingPipeline] = None
_corpus_ingestion: Optional[ScientificCorpusIngestion] = None


def get_auto_training_pipeline() -> AutoTrainingPipeline:
    """Get or create auto-training pipeline singleton"""
    global _auto_training_pipeline
    if _auto_training_pipeline is None:
        _auto_training_pipeline = AutoTrainingPipeline()
    return _auto_training_pipeline


def get_corpus_ingestion() -> ScientificCorpusIngestion:
    """Get or create corpus ingestion singleton"""
    global _corpus_ingestion
    if _corpus_ingestion is None:
        _corpus_ingestion = ScientificCorpusIngestion()
    return _corpus_ingestion

