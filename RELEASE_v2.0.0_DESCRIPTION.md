# Darwin Core v2.0.0 - Production-Ready AI Platform

First stable release of Darwin Core - Advanced AI Platform for Scientific Computing.

## 🎉 Features

### RAG++ (5 Variants)

- **GraphRAG**: Microsoft Research 2024 implementation
  - 70-80% win rate vs naive RAG (comprehensiveness)
  - 2-3% tokens vs hierarchical summarization
  - Entity extraction + Knowledge graphs + Community detection (Leiden)
  - Supports million-token corpora

- **Self-RAG**: University of Washington 2023
  - +280% accuracy on PopQA (14.7% → 55.8%)
  - Adaptive retrieval (only when necessary)
  - Self-reflection tokens for quality control

- **Visual RAG**: ColPali for document analysis
  - PDF/image understanding
  - Vision-language embeddings

- **Semantic Memory v2**: State-of-the-art 2025
  - Qdrant Hybrid search (dense + sparse)
  - Late chunking (Jina AI)
  - Binary quantization (90% storage reduction)
  - Backward compatible with v1

- **Simple RAG**: Baseline implementation
  - Keyword indexing + TF-IDF ranking

### Multi-AI Orchestration

- **Intelligent routing**: GPT-4, Claude 3.5 Sonnet, Gemini Pro
- **Domain-specific optimization**:
  - Mathematics/Algorithms → Claude (superior reasoning)
  - Biomaterials/Engineering → GPT-4 (STEM expertise)
  - Research/Literature → Gemini (Google Scholar integration)
- **Cross-AI context sharing**: Context Bridge for seamless collaboration
- **Performance learning**: Adapts routing based on success metrics

### Embedding Manager (SOTA 2025)

- **Jina v3**: 1024d, 8K context, multilingual, late chunking
- **gte-Qwen2-7B**: 3584d, 32K context! (massive context window)
- **Nomic v1.5**: 768d, 8K context, Matryoshka embeddings
- **Advanced features**:
  - Late chunking for better context preservation
  - Matryoshka embeddings (adaptive dimensionality)
  - Binary quantization (90% storage reduction)
  - GPU acceleration + intelligent caching

### Plugin System

- **gRPC-based communication**: High-performance, language-agnostic
- **Hot-reload capabilities**: Update plugins without restart
- **Circuit breaking and retry logic**: Production-grade reliability
- **OpenTelemetry tracing**: Full observability

### Production Infrastructure

- **FastAPI REST backend**: Modern, async, high-performance
- **Apache Pulsar**: Event-driven architecture for scalability
- **Qdrant vector database**: Hybrid search (dense + sparse)
- **Redis caching**: Multi-layer caching strategy
- **Full observability**: OpenTelemetry integration

### Agentic Workflows

- **LangGraph integration**: ReAct, Reflexion, Tree of Thoughts
- **Multi-agent patterns**: Researcher, Validator, Synthesizer, Coordinator
- **Self-healing capabilities**: Automatic recovery and adaptation

---

## 📊 Statistics

- **13,638** Python files
- **34,335** lines of code
- **39** production-ready services
- **Python 3.9+** support
- **MIT License**

---

## 📦 Installation

### Basic Installation

```bash
pip install darwin-core
```

### With Optional Features

```bash
# Neuroscience support (EEG, fMRI, Brain Transformers)
pip install darwin-core[neuro]

# Visual RAG
pip install darwin-core[visual]

# Development tools
pip install darwin-core[dev]
```

### From Source

```bash
git clone https://github.com/agourakis82/darwin-core.git
cd darwin-core
pip install -e .
```

---

## 🚀 Quick Start

### GraphRAG Example

```python
from darwin_core.services.graph_rag import GraphRAG

# Initialize
graphrag = GraphRAG()

# Ingest scientific papers
papers = ["paper1.txt", "paper2.txt", ...]
graphrag.ingest_documents(papers)

# Query with global understanding
answer = graphrag.query(
    "What are the main research themes?",
    query_type="global"
)
print(answer)
```

### Self-RAG Example

```python
from darwin_core.services.self_rag import SelfRAG

# Initialize
selfrag = SelfRAG()

# Adaptive retrieval (only when necessary)
result = selfrag.query("What is optimal scaffold porosity?")

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['reflection_tokens']}")
```

### Multi-AI Chat Example

```python
from darwin_core.multi_ai.router import MultiAIHub
from darwin_core.models.multi_ai_models import ChatRequest, ScientificDomain

# Initialize
hub = MultiAIHub()
await hub.initialize()

# Intelligent routing to best AI
request = ChatRequest(
    message="What are optimal parameters for bone scaffolds?",
    domain=ScientificDomain.BIOMATERIALS  # Routes to GPT-4
)

response = await hub.chat_with_routing(request)
print(response.content)
```

---

## 🔗 Related Projects

Darwin Core is designed as **optional infrastructure** for scientific applications. It provides advanced AI capabilities that can enhance standalone scientific software:

### Projects Using Darwin Core:

- **[darwin-scaffold-studio](https://github.com/agourakis82/darwin-scaffold-studio)** - Q1-Level MicroCT & SEM Analysis for Tissue Engineering
  - DOI: [10.5281/zenodo.17535484](https://doi.org/10.5281/zenodo.17535484)
  - Standalone + Optional Darwin AI features

- **[darwin-pbpk-platform](https://github.com/agourakis82/darwin-pbpk-platform)** - AI-Powered Drug Discovery & PBPK Modeling
  - DOI: [10.5281/zenodo.17536674](https://doi.org/10.5281/zenodo.17536674)
  - Standalone + Optional Darwin AI features

### Integration Pattern (Hybrid Mode):

```python
# Your standalone scientific app
try:
    from darwin_core.services.graph_rag import GraphRAG
    DARWIN_AVAILABLE = True
except ImportError:
    DARWIN_AVAILABLE = False

# Use Darwin features when available
if DARWIN_AVAILABLE:
    insights = GraphRAG().query("Relevant literature?")
    print(f"📚 AI Insights: {insights}")
else:
    # App works perfectly without Darwin Core
    baseline_analysis()
```

---

## 🙏 Acknowledgments

Darwin Core builds upon excellent open-source projects:

- **[Microsoft GraphRAG](https://microsoft.github.io/graphrag/)** - Knowledge graph RAG implementation
- **[Self-RAG (University of Washington)](https://selfrag.github.io/)** - Adaptive retrieval with reflection
- **[LangChain](https://github.com/langchain-ai/langchain)** - LLM orchestration framework
- **[Qdrant](https://qdrant.tech/)** - Vector database with hybrid search
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern Python web framework
- **[Apache Pulsar](https://pulsar.apache.org/)** - Distributed messaging system

---

## 📚 Documentation

- **Repository**: https://github.com/agourakis82/darwin-core
- **API Reference**: `http://localhost:8000/docs` (when running)
- **Issues**: https://github.com/agourakis82/darwin-core/issues

---

## 📄 License

MIT License - See [LICENSE](https://github.com/agourakis82/darwin-core/blob/main/LICENSE)

---

## 👥 Author

**Dr. Demetrios Agourakis**
- GitHub: [@agourakis82](https://github.com/agourakis82)
- Email: agourakis@agourakis.med.br

---

## 📈 Roadmap

### v2.1.0 (Next)
- Enhanced plugin hot-reload
- Additional embedding models
- Performance optimizations

### v2.2.0
- Streaming RAG responses
- Advanced caching strategies
- Multi-modal fusion improvements

### v3.0.0
- Breaking API improvements
- Enhanced observability
- Production deployment guides

---

**"Ciência rigorosa. Resultados honestos. Impacto real."**

---

## 🎯 Getting Help

- **Documentation**: Check README and inline docs
- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions
- **Email**: agourakis@agourakis.med.br

