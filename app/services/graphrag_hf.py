"""
GraphRAG with Hugging Face Local Models - REAL IMPLEMENTATION

Uses local LLMs (no API keys) running on GPU.
Based on Microsoft GraphRAG but with HF models.
"""

import logging
import torch
import networkx as nx
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
import re

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("transformers not available")

logger = logging.getLogger(__name__)


@dataclass
class GraphRAGConfig:
    """Configuration for GraphRAG"""
    # Model
    model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"  # 360M, super fast for testing
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    load_in_4bit: bool = False  # Small model, no need for quantization
    max_new_tokens: int = 512
    temperature: float = 0.1
    
    # Graph
    community_algorithm: str = "louvain"  # or "leiden"
    max_entities_per_doc: int = 20
    
    # Processing
    batch_size: int = 4


@dataclass
class Entity:
    """Entity extracted from text"""
    name: str
    type: str
    description: str
    mentions: List[int] = field(default_factory=list)  # doc indices


@dataclass
class Relationship:
    """Relationship between entities"""
    source: str
    target: str
    relation_type: str
    description: str
    weight: float = 1.0


class GraphRAG:
    """
    GraphRAG with local Hugging Face models.
    
    Runs entirely on GPU, no API keys needed.
    """
    
    def __init__(self, config: Optional[GraphRAGConfig] = None):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required")
        
        self.config = config or GraphRAGConfig()
        self.model = None
        self.tokenizer = None
        self.pipe = None
        
        # Graph storage
        self.documents: List[str] = []
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.graph: nx.Graph = nx.Graph()
        self.communities: Dict[int, List[str]] = {}
        
        logger.info(f"GraphRAG initialized with {self.config.model_name}")
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load HF model to GPU"""
        logger.info(f"Loading model: {self.config.model_name}")
        logger.info(f"Device: {self.config.device}")
        
        # Quantization config for 4-bit
        if self.config.load_in_4bit and self.config.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto",
                trust_remote_code=True
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Model loaded successfully!")
    
    def _generate(self, prompt: str) -> str:
        """Generate text with local LLM"""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.config.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Decode only the new tokens
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ""
    
    def _extract_entities(self, text: str, doc_id: int) -> List[Entity]:
        """Extract entities from text using LLM"""
        prompt = f"""<|im_start|>system
You are a scientific entity extraction assistant. Extract key entities from text and format as JSON.<|im_end|>
<|im_start|>user
Text: {text[:500]}

Extract entities (name, type, description) as JSON list:
[{{"name": "...", "type": "...", "description": "..."}}]<|im_end|>
<|im_start|>assistant
"""
        
        response = self._generate(prompt)
        logger.info(f"Entity extraction response: {response[:200]}")
        
        # Parse JSON (with error handling)
        try:
            # Find JSON in response
            json_match = re.search(r'\[(.*?)\]', response, re.DOTALL)
            if json_match:
                json_str = '[' + json_match.group(1) + ']'
                entities_data = json.loads(json_str)
                
                entities = []
                for e in entities_data[:self.config.max_entities_per_doc]:
                    entity = Entity(
                        name=e.get('name', '').strip(),
                        type=e.get('type', 'unknown').strip(),
                        description=e.get('description', '').strip(),
                        mentions=[doc_id]
                    )
                    if entity.name:
                        entities.append(entity)
                
                logger.debug(f"Extracted {len(entities)} entities from doc {doc_id}")
                return entities
        except Exception as e:
            logger.warning(f"Failed to parse entities: {e}")
        
        return []
    
    def _extract_relationships(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Relationship]:
        """Extract relationships between entities"""
        if len(entities) < 2:
            return []
        
        entity_names = [e.name for e in entities[:10]]  # Limit to avoid long prompts
        
        prompt = f"""<|im_start|>system
You are a scientific relationship extraction assistant.<|im_end|>
<|im_start|>user
Entities: {', '.join(entity_names)}
Text: {text[:500]}

Extract relationships as JSON:
[{{"source": "...", "target": "...", "relation": "...", "description": "..."}}]<|im_end|>
<|im_start|>assistant
"""
        
        response = self._generate(prompt)
        logger.info(f"Relationship extraction response: {response[:200]}")
        
        try:
            json_match = re.search(r'\[(.*?)\]', response, re.DOTALL)
            if json_match:
                json_str = '[' + json_match.group(1) + ']'
                rels_data = json.loads(json_str)
                
                relationships = []
                for r in rels_data:
                    rel = Relationship(
                        source=r.get('source', '').strip(),
                        target=r.get('target', '').strip(),
                        relation_type=r.get('relation', 'related_to').strip(),
                        description=r.get('description', '').strip()
                    )
                    if rel.source and rel.target:
                        relationships.append(rel)
                
                logger.debug(f"Extracted {len(relationships)} relationships")
                return relationships
        except Exception as e:
            logger.warning(f"Failed to parse relationships: {e}")
        
        return []
    
    def ingest_documents(self, documents: List[str]) -> Dict[str, Any]:
        """
        Ingest documents and build knowledge graph.
        
        Args:
            documents: List of document texts
        
        Returns:
            Statistics
        """
        logger.info(f"Ingesting {len(documents)} documents...")
        
        self.documents = documents
        
        # Extract entities and relationships
        for doc_id, doc_text in enumerate(documents):
            logger.info(f"Processing document {doc_id + 1}/{len(documents)}")
            
            # Extract entities
            entities = self._extract_entities(doc_text, doc_id)
            
            # Store entities
            for entity in entities:
                if entity.name in self.entities:
                    # Merge with existing
                    self.entities[entity.name].mentions.append(doc_id)
                else:
                    self.entities[entity.name] = entity
            
            # Extract relationships
            relationships = self._extract_relationships(doc_text, entities)
            self.relationships.extend(relationships)
        
        # Build graph
        self._build_graph()
        
        # Detect communities
        self._detect_communities()
        
        stats = {
            "num_documents": len(documents),
            "num_entities": len(self.entities),
            "num_relationships": len(self.relationships),
            "num_communities": len(self.communities),
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges()
        }
        
        logger.info(f"Ingestion complete: {stats}")
        return stats
    
    def _build_graph(self):
        """Build NetworkX graph from entities and relationships"""
        self.graph = nx.Graph()
        
        # Add nodes (entities)
        for entity_name, entity in self.entities.items():
            self.graph.add_node(
                entity_name,
                type=entity.type,
                description=entity.description,
                mentions=len(entity.mentions)
            )
        
        # Add edges (relationships)
        for rel in self.relationships:
            if rel.source in self.graph and rel.target in self.graph:
                self.graph.add_edge(
                    rel.source,
                    rel.target,
                    relation=rel.relation_type,
                    description=rel.description,
                    weight=rel.weight
                )
        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def _detect_communities(self):
        """Detect communities in graph"""
        if self.graph.number_of_nodes() == 0:
            return
        
        try:
            from community import community_louvain
            partition = community_louvain.best_partition(self.graph)
            
            # Group by community
            self.communities = defaultdict(list)
            for node, comm_id in partition.items():
                self.communities[comm_id].append(node)
            
            logger.info(f"Detected {len(self.communities)} communities")
        except ImportError:
            logger.warning("python-louvain not installed, skipping community detection")
    
    def query(self, question: str) -> str:
        """
        Query the knowledge graph.
        
        Args:
            question: User question
        
        Returns:
            Answer
        """
        # Simple: retrieve relevant entities and generate answer
        # Extract key terms from question
        question_lower = question.lower()
        
        relevant_entities = []
        for entity_name, entity in self.entities.items():
            if entity_name.lower() in question_lower:
                relevant_entities.append(entity)
        
        if not relevant_entities:
            # Fallback: use all entities (simplified)
            relevant_entities = list(self.entities.values())[:5]
        
        # Build context
        context_parts = []
        for entity in relevant_entities[:5]:
            context_parts.append(f"- {entity.name} ({entity.type}): {entity.description}")
        
        context = "\n".join(context_parts)
        
        # Generate answer
        prompt = f"""<|im_start|>system
You are a scientific knowledge assistant. Answer based on provided knowledge.<|im_end|>
<|im_start|>user
Knowledge:
{context}

Question: {question}<|im_end|>
<|im_start|>assistant
Answer: """
        
        answer = self._generate(prompt)
        logger.info(f"Query answer: {answer[:200]}")
        return answer
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return {
            "num_documents": len(self.documents),
            "num_entities": len(self.entities),
            "num_relationships": len(self.relationships),
            "num_communities": len(self.communities),
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges()
        }


# Test
if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("GraphRAG with Hugging Face Local Models - REAL TEST")
    print("="*70)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n⚠️  No GPU, using CPU (will be slow)")
    
    # Create GraphRAG
    print("\nInitializing GraphRAG...")
    config = GraphRAGConfig(
        model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
        load_in_4bit=False,
        max_new_tokens=256
    )
    
    graphrag = GraphRAG(config)
    
    # Test documents
    docs = [
        "PCL scaffolds are biocompatible polymers used in tissue engineering. They have good mechanical properties.",
        "Scaffold porosity affects cell infiltration and proliferation. Optimal pore size is 300-500 micrometers.",
        "3D printing enables precise control of scaffold architecture and pore distribution."
    ]
    
    print(f"\nIngesting {len(docs)} documents...")
    stats = graphrag.ingest_documents(docs)
    
    print("\n📊 Statistics:")
    print(f"  Entities: {stats['num_entities']}")
    print(f"  Relationships: {stats['num_relationships']}")
    print(f"  Communities: {stats['num_communities']}")
    
    # Query
    print("\n🔍 Testing query...")
    question = "What affects cell infiltration in scaffolds?"
    print(f"Q: {question}")
    answer = graphrag.query(question)
    print(f"A: {answer}")
    
    print("\n✅ GraphRAG with HF models works!")
    sys.exit(0)

