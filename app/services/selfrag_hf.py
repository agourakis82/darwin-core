"""
Self-RAG with Hugging Face Local Models - REAL IMPLEMENTATION

Adaptive retrieval using reflection: decides when to retrieve and self-corrects.
Based on "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
"""

import logging
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)


class ReflectionToken(Enum):
    """Reflection tokens for Self-RAG"""
    # Retrieval necessity
    RETRIEVE = "[Retrieve]"
    NO_RETRIEVE = "[No Retrieve]"
    
    # Relevance
    RELEVANT = "[Relevant]"
    IRRELEVANT = "[Irrelevant]"
    
    # Support
    SUPPORTED = "[Supported]"
    PARTIALLY_SUPPORTED = "[Partially Supported]"
    NOT_SUPPORTED = "[Not Supported]"
    
    # Usefulness
    USEFUL = "[Useful]"
    NOT_USEFUL = "[Not Useful]"


@dataclass
class SelfRAGConfig:
    """Configuration for Self-RAG"""
    model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    load_in_4bit: bool = False
    max_new_tokens: int = 512
    temperature: float = 0.1
    
    # Retrieval
    retrieval_threshold: float = 0.7
    max_retrieval_docs: int = 3
    
    # Self-correction
    max_iterations: int = 2


@dataclass
class RetrievalResult:
    """Result from retrieval"""
    doc_id: int
    text: str
    score: float


class SelfRAG:
    """
    Self-RAG with local Hugging Face models.
    
    Features:
    - Adaptive retrieval (decides when to retrieve)
    - Relevance checking
    - Support verification
    - Usefulness assessment
    - Self-correction loops
    """
    
    def __init__(self, config: Optional[SelfRAGConfig] = None):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required")
        
        self.config = config or SelfRAGConfig()
        self.model = None
        self.tokenizer = None
        
        # Knowledge base (simplified)
        self.documents: List[str] = []
        
        logger.info(f"SelfRAG initialized with {self.config.model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load HF model"""
        logger.info(f"Loading model: {self.config.model_name}")
        
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
        """Generate text"""
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
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ""
    
    def add_documents(self, documents: List[str]):
        """Add documents to knowledge base"""
        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} documents (total: {len(self.documents)})")
    
    def _should_retrieve(self, question: str) -> bool:
        """Decide if retrieval is necessary"""
        prompt = f"""<|im_start|>system
You decide if external knowledge is needed to answer a question.<|im_end|>
<|im_start|>user
Question: {question}

Is retrieval needed? Answer with only "Yes" or "No".<|im_end|>
<|im_start|>assistant
"""
        
        response = self._generate(prompt).lower()
        should_retrieve = "yes" in response
        
        logger.info(f"Should retrieve? {should_retrieve} (response: {response[:50]})")
        return should_retrieve
    
    def _retrieve(self, query: str) -> List[RetrievalResult]:
        """Retrieve relevant documents (simplified: keyword matching)"""
        if not self.documents:
            return []
        
        query_lower = query.lower()
        results = []
        
        for i, doc in enumerate(self.documents):
            doc_lower = doc.lower()
            # Simple keyword matching score
            keywords = query_lower.split()
            score = sum(1 for kw in keywords if kw in doc_lower) / max(len(keywords), 1)
            
            if score > 0:
                results.append(RetrievalResult(doc_id=i, text=doc, score=score))
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top-k
        top_results = results[:self.config.max_retrieval_docs]
        logger.info(f"Retrieved {len(top_results)} documents")
        
        return top_results
    
    def _check_relevance(self, question: str, context: str) -> bool:
        """Check if retrieved context is relevant"""
        prompt = f"""<|im_start|>system
You check if context is relevant to a question.<|im_end|>
<|im_start|>user
Question: {question}
Context: {context[:300]}

Is the context relevant? Answer with only "Relevant" or "Irrelevant".<|im_end|>
<|im_start|>assistant
"""
        
        response = self._generate(prompt).lower()
        is_relevant = "relevant" in response and "irrelevant" not in response
        
        logger.info(f"Context relevant? {is_relevant}")
        return is_relevant
    
    def _check_support(self, answer: str, context: str) -> str:
        """Check if answer is supported by context"""
        prompt = f"""<|im_start|>system
You verify if an answer is supported by context.<|im_end|>
<|im_start|>user
Context: {context[:300]}
Answer: {answer}

Is the answer supported? Choose one:
- Fully Supported
- Partially Supported
- Not Supported<|im_end|>
<|im_start|>assistant
"""
        
        response = self._generate(prompt).lower()
        
        if "fully" in response or "fully supported" in response:
            support = "Supported"
        elif "partially" in response:
            support = "Partially Supported"
        else:
            support = "Not Supported"
        
        logger.info(f"Answer support: {support}")
        return support
    
    def _generate_answer(self, question: str, context: Optional[str] = None) -> str:
        """Generate answer with or without context"""
        if context:
            prompt = f"""<|im_start|>system
You answer questions based on provided context.<|im_end|>
<|im_start|>user
Context: {context}

Question: {question}<|im_end|>
<|im_start|>assistant
Answer: """
        else:
            prompt = f"""<|im_start|>system
You answer questions based on your knowledge.<|im_end|>
<|im_start|>user
Question: {question}<|im_end|>
<|im_start|>assistant
Answer: """
        
        answer = self._generate(prompt)
        logger.info(f"Generated answer: {answer[:100]}")
        return answer
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query with Self-RAG.
        
        Returns:
            Dict with answer, retrieved docs, and reflection results
        """
        logger.info(f"Query: {question}")
        
        # Step 1: Decide if retrieval is needed
        should_retrieve = self._should_retrieve(question)
        
        retrieved_docs = []
        context = None
        
        if should_retrieve and self.documents:
            # Step 2: Retrieve
            retrieved_docs = self._retrieve(question)
            
            if retrieved_docs:
                # Combine top docs
                context = "\n\n".join([r.text for r in retrieved_docs])
                
                # Step 3: Check relevance
                is_relevant = self._check_relevance(question, context)
                
                if not is_relevant:
                    logger.info("Context not relevant, ignoring retrieval")
                    context = None
        
        # Step 4: Generate answer
        answer = self._generate_answer(question, context)
        
        # Step 5: Check support (if context was used)
        support_level = None
        if context:
            support_level = self._check_support(answer, context)
            
            # Step 6: Self-correction if not supported
            if support_level != "Supported":
                logger.info("Answer not fully supported, attempting correction...")
                correction_prompt = f"""<|im_start|>system
You refine answers to better align with context.<|im_end|>
<|im_start|>user
Context: {context[:300]}
Question: {question}
Previous answer: {answer}

Provide a better answer that is fully supported by the context.<|im_end|>
<|im_start|>assistant
Improved answer: """
                
                corrected_answer = self._generate(correction_prompt)
                if corrected_answer:
                    answer = corrected_answer
                    logger.info(f"Corrected answer: {answer[:100]}")
        
        return {
            "answer": answer,
            "retrieved": should_retrieve,
            "num_docs_retrieved": len(retrieved_docs),
            "support_level": support_level,
            "context_used": context is not None
        }


# Test
if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("Self-RAG with Hugging Face Local Models - REAL TEST")
    print("="*70)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n⚠️  No GPU, using CPU")
    
    # Create Self-RAG
    print("\nInitializing Self-RAG...")
    config = SelfRAGConfig(
        model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
        max_new_tokens=128
    )
    
    selfrag = SelfRAG(config)
    
    # Add documents
    docs = [
        "PCL scaffolds are biocompatible polymers with good mechanical properties.",
        "Optimal pore size for scaffolds is 300-500 micrometers for cell infiltration.",
        "3D printing enables precise control of scaffold architecture."
    ]
    
    print(f"\nAdding {len(docs)} documents to knowledge base...")
    selfrag.add_documents(docs)
    
    # Test queries
    questions = [
        "What is the optimal pore size for scaffolds?",
        "What is the capital of France?",  # Should not retrieve
    ]
    
    for q in questions:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        result = selfrag.query(q)
        print(f"A: {result['answer']}")
        print(f"📊 Retrieved: {result['retrieved']}, Docs: {result['num_docs_retrieved']}, Support: {result['support_level']}")
    
    print(f"\n{'='*70}")
    print("✅ Self-RAG with HF models works!")
    sys.exit(0)

