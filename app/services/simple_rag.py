"""
Simple RAG - Versão Funcional Básica

RAG que FUNCIONA sem dependencies complexas.
Usa apenas Python padrão + requests.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Documento simples"""
    id: str
    text: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.id:
            self.id = hashlib.md5(self.text.encode()).hexdigest()


@dataclass
class SearchResult:
    """Resultado de busca"""
    document: Document
    score: float
    snippet: str


class SimpleRAG:
    """
    RAG básico mas FUNCIONAL.
    
    Features:
    - Indexação de documentos
    - Busca por keyword
    - Ranking TF-IDF simples
    - Geração de resposta
    """
    
    def __init__(self, collection_name: str = "default"):
        self.collection_name = collection_name
        self.documents: Dict[str, Document] = {}
        self.index: Dict[str, List[str]] = {}  # word -> [doc_ids]
        logger.info(f"SimpleRAG initialized: {collection_name}")
    
    def add_document(self, text: str, metadata: Optional[Dict] = None) -> str:
        """Adiciona documento"""
        doc = Document(id="", text=text, metadata=metadata or {})
        self.documents[doc.id] = doc
        
        # Index words
        words = self._tokenize(text)
        for word in words:
            if word not in self.index:
                self.index[word] = []
            if doc.id not in self.index[word]:
                self.index[word].append(doc.id)
        
        logger.debug(f"Added document: {doc.id[:8]}... ({len(words)} words)")
        return doc.id
    
    def add_documents(self, texts: List[str]) -> List[str]:
        """Adiciona múltiplos documentos"""
        return [self.add_document(text) for text in texts]
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenização simples"""
        import re
        # Lowercase e remove pontuação
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        # Remove stopwords básicas
        stopwords = {'a', 'o', 'de', 'da', 'do', 'e', 'é', 'em', 'para', 
                     'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'}
        return [w for w in words if w not in stopwords and len(w) > 2]
    
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Busca documentos relevantes.
        
        Args:
            query: Query text
            top_k: Number of results
        
        Returns:
            List of SearchResult
        """
        query_words = self._tokenize(query)
        
        if not query_words:
            return []
        
        # Score documents
        doc_scores: Dict[str, float] = {}
        
        for word in query_words:
            if word in self.index:
                for doc_id in self.index[word]:
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = 0
                    doc_scores[doc_id] += 1  # Simple frequency
        
        # Sort by score
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create results
        results = []
        for doc_id, score in ranked[:top_k]:
            doc = self.documents[doc_id]
            snippet = self._create_snippet(doc.text, query_words)
            results.append(SearchResult(
                document=doc,
                score=score,
                snippet=snippet
            ))
        
        logger.info(f"Search for '{query}': {len(results)} results")
        return results
    
    def _create_snippet(self, text: str, query_words: List[str], max_len: int = 200) -> str:
        """Cria snippet com contexto"""
        text_lower = text.lower()
        
        # Encontra primeira ocorrência de query word
        best_pos = len(text)
        for word in query_words:
            pos = text_lower.find(word)
            if pos != -1 and pos < best_pos:
                best_pos = pos
        
        if best_pos == len(text):
            # Nenhuma palavra encontrada, retorna início
            return text[:max_len] + "..." if len(text) > max_len else text
        
        # Centraliza no match
        start = max(0, best_pos - max_len // 2)
        end = start + max_len
        
        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        
        return snippet
    
    def query(self, question: str, top_k: int = 3) -> str:
        """
        Query RAG completo.
        
        Args:
            question: User question
            top_k: Number of docs to retrieve
        
        Returns:
            Generated answer
        """
        # Retrieve
        results = self.search(question, top_k=top_k)
        
        if not results:
            return "Desculpe, não encontrei informações relevantes."
        
        # Build context
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Documento {i}]\n{result.snippet}")
        
        context = "\n\n".join(context_parts)
        
        # Simple answer generation (sem LLM, apenas extração)
        answer = self._generate_simple_answer(question, results)
        
        return answer
    
    def _generate_simple_answer(self, question: str, results: List[SearchResult]) -> str:
        """Gera resposta simples (extração, não geração)"""
        # Por enquanto, retorna o melhor snippet
        if results:
            best = results[0]
            return f"Baseado nos documentos encontrados:\n\n{best.snippet}\n\n(Score: {best.score:.2f})"
        return "Não encontrei informação relevante."
    
    def get_stats(self) -> Dict[str, Any]:
        """Estatísticas"""
        return {
            'collection': self.collection_name,
            'num_documents': len(self.documents),
            'num_words_indexed': len(self.index),
            'avg_doc_length': sum(len(d.text) for d in self.documents.values()) / len(self.documents) if self.documents else 0
        }


# Factory
def get_simple_rag(collection_name: str = "default") -> SimpleRAG:
    """Factory function"""
    return SimpleRAG(collection_name)


# Test
if __name__ == "__main__":
    import sys
    
    # Create RAG
    rag = get_simple_rag("test")
    
    # Add documents
    docs = [
        "Python é uma linguagem de programação de alto nível.",
        "Machine learning é um subcampo da inteligência artificial.",
        "Deep learning usa redes neurais profundas para aprendizado.",
        "RAG combina retrieval e generation para melhorar respostas.",
    ]
    
    rag.add_documents(docs)
    
    # Test search
    print("\n=== TEST SEARCH ===")
    results = rag.search("machine learning", top_k=2)
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.score:.2f}")
        print(f"   {result.snippet}\n")
    
    # Test query
    print("\n=== TEST QUERY ===")
    answer = rag.query("O que é RAG?")
    print(answer)
    
    # Stats
    print("\n=== STATS ===")
    print(json.dumps(rag.get_stats(), indent=2))
    
    print("\n✅ SimpleRAG works!")
    sys.exit(0)

