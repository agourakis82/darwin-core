# Darwin Core - AI Platform

**Advanced AI Platform for Scientific Computing**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17537549.svg)](https://doi.org/10.5281/zenodo.17537549)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎯 Overview

Darwin Core is a production-ready AI platform providing state-of-the-art RAG, multi-AI orchestration, and knowledge graph capabilities for scientific applications.

### Key Features

- **RAG++ (5 Variants)**
  - GraphRAG: Microsoft Research 2024 (70-80% win rate vs naive RAG)
  - Self-RAG: University of Washington (+280% accuracy)
  - Visual RAG: ColPali for documents
  - Semantic Memory v2: Qdrant Hybrid (dense + sparse)
  - Simple RAG: Baseline

- **Multi-AI Orchestration**
  - Intelligent routing: GPT-4, Claude, Gemini
  - Domain-specific optimization (Biomaterials, Math, Research)
  - Context sharing across AIs
  - Performance learning

- **Embedding Manager (SOTA 2025)**
  - Jina v3 (1024d, 8K context, multilingual)
  - gte-Qwen2-7B (3584d, 32K context!)
  - Late chunking, Matryoshka, Binary quantization

- **Plugin System**
  - gRPC communication
  - Hot-reload
  - Circuit breaking

- **Production Infrastructure**
  - FastAPI backend
  - Apache Pulsar (event-driven)
  - Qdrant (vector database)
  - Redis (caching)
  - OpenTelemetry (observability)

---

## 📦 Installation

### Basic Installation

```bash
pip install darwin-core
```

### With Optional Features

```bash
# Neuroscience support
pip install darwin-core[neuro]

# Visual RAG
pip install darwin-core[visual]

# Development
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

### GraphRAG

```python
from darwin_core.services.graph_rag import GraphRAG

# Initialize
graphrag = GraphRAG()

# Ingest documents
papers = ["paper1.txt", "paper2.txt", ...]
graphrag.ingest_documents(papers)

# Query
answer = graphrag.query(
    "What are the main themes in these papers?",
    query_type="global"
)
print(answer)
```

### Self-RAG

```python
from darwin_core.services.self_rag import SelfRAG

# Initialize
selfrag = SelfRAG()

# Query (adaptive retrieval)
result = selfrag.query("What is scaffold porosity?")

print(f"Answer: {result['answer']}")
print(f"Reflection: {result['reflection_tokens']}")
```

### Multi-AI Chat

```python
from darwin_core.multi_ai.router import MultiAIHub
from darwin_core.models.multi_ai_models import ChatRequest, ScientificDomain

# Initialize
hub = MultiAIHub()
await hub.initialize()

# Chat with intelligent routing
request = ChatRequest(
    message="What are the optimal parameters for bone scaffolds?",
    domain=ScientificDomain.BIOMATERIALS
)

# Automatically routes to GPT-4 (biomaterials expert)
response = await hub.chat_with_routing(request)
print(response.content)
```

---

## 🏗️ Architecture

```
Darwin Core
├── RAG++ Services
│   ├── GraphRAG (Microsoft Research 2024)
│   ├── Self-RAG (UW 2023)
│   ├── Visual RAG (ColPali)
│   ├── Semantic Memory v2
│   └── Simple RAG
├── Multi-AI Hub
│   ├── Chat Orchestrator (GPT-4/Claude/Gemini)
│   ├── Context Bridge (cross-AI context)
│   └── Conversation Manager
├── Embedding Manager
│   ├── Jina v3 (8K context)
│   ├── gte-Qwen2-7B (32K context!)
│   └── Matryoshka + Binary quantization
├── Infrastructure
│   ├── FastAPI REST API
│   ├── gRPC Plugin System
│   ├── Apache Pulsar (events)
│   ├── Qdrant (vectors)
│   └── Redis (cache)
└── Agentic Workflows
    ├── ReAct, Reflexion
    ├── Tree of Thoughts
    └── Multi-agent debate
```

---

## 📚 Documentation

### Core Services

- **GraphRAG**: [docs/graphrag.md](docs/graphrag.md)
- **Self-RAG**: [docs/selfrag.md](docs/selfrag.md)
- **Multi-AI**: [docs/multi-ai.md](docs/multi-ai.md)
- **Embeddings**: [docs/embeddings.md](docs/embeddings.md)

### API Reference

- FastAPI: `http://localhost:8000/docs`
- gRPC: [protos/](protos/)

---

## 🧪 Testing

```bash
# Install dev dependencies
pip install darwin-core[dev]

# Run tests
pytest

# With coverage
pytest --cov=app --cov-report=html
```

---

## 🛠️ Development

```bash
# Clone repo
git clone https://github.com/agourakis82/darwin-core.git
cd darwin-core

# Install in editable mode
pip install -e ".[dev]"

# Format code
black app/

# Lint
flake8 app/

# Type check
mypy app/
```

---

## 📊 Performance

### GraphRAG Benchmarks

- **Comprehensiveness**: 70-80% win rate vs naive RAG
- **Token efficiency**: 2-3% vs hierarchical summarization
- **Scalability**: Supports million-token corpora

### Self-RAG Benchmarks

- **PopQA**: 55.8% accuracy (+280% vs baseline 14.7%)
- **Adaptive**: Only retrieves when necessary (efficient!)

---

## 🤝 Use Cases

### Scientific Applications

Darwin Core is designed as **optional infrastructure** for scientific software:

```python
# Your standalone app
try:
    from darwin_core.services.graph_rag import GraphRAG
    DARWIN_AVAILABLE = True
except ImportError:
    DARWIN_AVAILABLE = False

# Use Darwin features if available
if DARWIN_AVAILABLE:
    insights = GraphRAG().query(...)
    st.info(f"📚 Literature: {insights}")
```

**Example projects using Darwin Core:**
- [darwin-scaffold-studio](https://github.com/agourakis82/darwin-scaffold-studio) (DOI: 10.5281/zenodo.17535484)
- [darwin-pbpk-platform](https://github.com/agourakis82/darwin-pbpk-platform) (DOI: 10.5281/zenodo.17536674)

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## 👥 Author

**Dr. Demetrios Agourakis**
- GitHub: [@agourakis82](https://github.com/agourakis82)
- Email: agourakis@agourakis.med.br

---

## 🙏 Acknowledgments

Darwin Core builds upon excellent open-source projects:
- [Microsoft GraphRAG](https://microsoft.github.io/graphrag/)
- [Self-RAG (University of Washington)](https://selfrag.github.io/)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Qdrant](https://qdrant.tech/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

## 📈 Status

- **Version**: 2.0.0
- **Status**: Beta (Production-ready)
- **Python**: 3.9+
- **Maintenance**: Active

---

**"Ciência rigorosa. Resultados honestos. Impacto real."**
