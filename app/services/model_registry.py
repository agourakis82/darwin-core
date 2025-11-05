"""
Model Registry - Dynamic Model Management for HuggingFace Integration

Allows adding/removing models dynamically without redeploying
Supports: vLLM, Ollama, Transformers, and custom model providers
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio

logger = logging.getLogger("darwin.model_registry")


class ModelProvider(str, Enum):
    """Model provider types"""
    OLLAMA = "ollama"              # Ollama (llama3, mistral, etc)
    VLLM = "vllm"                  # vLLM (high-performance inference)
    HUGGINGFACE = "huggingface"    # Transformers pipeline
    OPENAI = "openai"              # OpenAI API
    ANTHROPIC = "anthropic"        # Claude API
    GOOGLE = "google"              # Gemini API
    CUSTOM = "custom"              # Custom provider


@dataclass
class ModelConfig:
    """Configuration for a model"""
    model_id: str                           # e.g., "llama3:8b"
    provider: ModelProvider
    display_name: str
    
    # Provider-specific config
    endpoint: Optional[str] = None          # e.g., "http://ollama:11434"
    api_key: Optional[str] = None           # For commercial APIs
    
    # Model capabilities
    capabilities: List[str] = field(default_factory=list)  # ["chat", "embedding", "code"]
    context_length: int = 2048
    max_tokens: int = 1024
    
    # Resource requirements
    gpu_required: bool = False
    gpu_memory_mb: int = 0
    cpu_cores: float = 1.0
    ram_mb: int = 2048
    
    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    enabled: bool = True
    
    # Performance
    temperature: float = 0.7
    top_p: float = 0.9


class ModelRegistry:
    """
    Central registry for all AI models
    
    Features:
    - Add models dynamically
    - Remove models
    - List available models
    - Filter by capability/provider
    - Health check models
    - Load balancing across replicas
    """
    
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default model catalog"""
        
        # === LOCAL MODELS (Ollama) - ATUAL SISTEMA ===
        
        # General Purpose Models
        self.register_model(ModelConfig(
            model_id="llama3.1:8b-instruct-q4_0",
            provider=ModelProvider.OLLAMA,
            display_name="Llama 3.1 8B Instruct",
            endpoint="http://localhost:11434",
            capabilities=["chat", "reasoning", "instruction"],
            context_length=8192,
            max_tokens=2048,
            gpu_required=True,
            gpu_memory_mb=4700,
            description="Meta's Llama 3.1 8B quantized - Fast general purpose",
            tags=["local", "open-source", "fast", "general"],
            temperature=0.8
        ))
        
        self.register_model(ModelConfig(
            model_id="qwen2.5-coder:7b-instruct-q4_0",
            provider=ModelProvider.OLLAMA,
            display_name="Qwen 2.5 Coder 7B",
            endpoint="http://localhost:11434",
            capabilities=["code", "reasoning", "analysis"],
            context_length=32768,
            max_tokens=4096,
            gpu_required=True,
            gpu_memory_mb=4400,
            description="Alibaba Qwen 2.5 Coder - Code expert",
            tags=["local", "code", "specialized", "long-context"],
            temperature=0.7
        ))
        
        self.register_model(ModelConfig(
            model_id="qwen2.5:32b-instruct-q4_K_M",
            provider=ModelProvider.OLLAMA,
            display_name="Qwen 2.5 32B Instruct",
            endpoint="http://localhost:11434",
            capabilities=["chat", "reasoning", "analysis", "research"],
            context_length=32768,
            max_tokens=8192,
            gpu_required=True,
            gpu_memory_mb=19000,
            description="Qwen 2.5 32B - Large, powerful reasoning",
            tags=["local", "large", "high-quality", "research"],
            temperature=0.7
        ))
        
        self.register_model(ModelConfig(
            model_id="llava:13b",
            provider=ModelProvider.OLLAMA,
            display_name="LLaVA 13B",
            endpoint="http://localhost:11434",
            capabilities=["vision", "chat", "multimodal"],
            context_length=4096,
            max_tokens=2048,
            gpu_required=True,
            gpu_memory_mb=8000,
            description="LLaVA - Vision-Language model (image understanding)",
            tags=["local", "multimodal", "vision"],
            temperature=0.8
        ))
        
        # === DARWIN SPECIALIZED MODELS (Fine-tuned) ===
        
        self.register_model(ModelConfig(
            model_id="darwin-biomaterials-local:latest",
            provider=ModelProvider.OLLAMA,
            display_name="DARWIN Biomaterials Expert",
            endpoint="http://localhost:11434",
            capabilities=["chat", "biomaterials", "scaffolds", "kec"],
            context_length=8192,
            max_tokens=2048,
            gpu_required=True,
            gpu_memory_mb=4700,
            description="Fine-tuned for biomaterials research and KEC analysis",
            tags=["local", "darwin", "specialized", "biomaterials"],
            temperature=0.7
        ))
        
        self.register_model(ModelConfig(
            model_id="darwin-medical-local:latest",
            provider=ModelProvider.OLLAMA,
            display_name="DARWIN Medical Expert",
            endpoint="http://localhost:11434",
            capabilities=["chat", "medical", "clinical", "diagnosis"],
            context_length=8192,
            max_tokens=2048,
            gpu_required=True,
            gpu_memory_mb=4700,
            description="Fine-tuned for medical/clinical analysis",
            tags=["local", "darwin", "specialized", "medical"],
            temperature=0.7
        ))
        
        self.register_model(ModelConfig(
            model_id="darwin-pharmacology-local:latest",
            provider=ModelProvider.OLLAMA,
            display_name="DARWIN Pharmacology Expert",
            endpoint="http://localhost:11434",
            capabilities=["chat", "pharmacology", "pbpk", "drug-discovery"],
            context_length=8192,
            max_tokens=2048,
            gpu_required=True,
            gpu_memory_mb=4700,
            description="Fine-tuned for pharmacology and PBPK modeling",
            tags=["local", "darwin", "specialized", "pharmacology"],
            temperature=0.7
        ))
        
        self.register_model(ModelConfig(
            model_id="darwin-mathematics-local:latest",
            provider=ModelProvider.OLLAMA,
            display_name="DARWIN Mathematics Expert",
            endpoint="http://localhost:11434",
            capabilities=["chat", "mathematics", "physics", "analysis"],
            context_length=8192,
            max_tokens=2048,
            gpu_required=True,
            gpu_memory_mb=4700,
            description="Fine-tuned for mathematical and physical analysis",
            tags=["local", "darwin", "specialized", "mathematics"],
            temperature=0.7
        ))
        
        self.register_model(ModelConfig(
            model_id="darwin-quantum-local:latest",
            provider=ModelProvider.OLLAMA,
            display_name="DARWIN Quantum Expert",
            endpoint="http://localhost:11434",
            capabilities=["chat", "quantum", "physics", "advanced"],
            context_length=8192,
            max_tokens=2048,
            gpu_required=True,
            gpu_memory_mb=4700,
            description="Fine-tuned for quantum mechanics and advanced physics",
            tags=["local", "darwin", "specialized", "quantum"],
            temperature=0.8
        ))
        
        self.register_model(ModelConfig(
            model_id="darwin-philosophy-local:latest",
            provider=ModelProvider.OLLAMA,
            display_name="DARWIN Philosophy Expert",
            endpoint="http://localhost:11434",
            capabilities=["chat", "philosophy", "reasoning", "ethics"],
            context_length=8192,
            max_tokens=2048,
            gpu_required=True,
            gpu_memory_mb=4700,
            description="Fine-tuned for philosophical reasoning and ethics",
            tags=["local", "darwin", "specialized", "philosophy"],
            temperature=0.9
        ))
        
        # === EMBEDDINGS ===
        
        self.register_model(ModelConfig(
            model_id="nomic-embed-text:latest",
            provider=ModelProvider.OLLAMA,
            display_name="Nomic Embed Text",
            endpoint="http://localhost:11434",
            capabilities=["embedding"],
            context_length=8192,
            gpu_required=False,
            cpu_cores=2.0,
            ram_mb=512,
            description="Nomic embeddings for semantic search",
            tags=["local", "embedding", "fast"],
            enabled=True
        ))
        
        # === COMMERCIAL MODELS ===
        
        self.register_model(ModelConfig(
            model_id="gpt-4-turbo",
            provider=ModelProvider.OPENAI,
            display_name="GPT-4 Turbo",
            capabilities=["chat", "reasoning", "analysis"],
            context_length=128000,
            max_tokens=4096,
            description="OpenAI GPT-4 Turbo - Generalist, broad connections",
            tags=["commercial", "high-quality", "expensive"],
            temperature=0.9,
            enabled=True  # Enable if API key available
        ))
        
        self.register_model(ModelConfig(
            model_id="claude-3-haiku",
            provider=ModelProvider.ANTHROPIC,
            display_name="Claude 3 Haiku",
            capabilities=["chat", "reasoning", "analysis"],
            context_length=200000,
            max_tokens=4096,
            description="Anthropic Claude Haiku - Specialist, rigorous analysis",
            tags=["commercial", "fast", "affordable"],
            temperature=0.7
        ))
        
        self.register_model(ModelConfig(
            model_id="gemini-2.5-pro",
            provider=ModelProvider.GOOGLE,
            display_name="Gemini 2.5 Pro",
            capabilities=["chat", "reasoning", "medical"],
            context_length=1000000,
            max_tokens=8192,
            description="Google Gemini 2.5 Pro - Medical/biomaterials expert",
            tags=["commercial", "medical", "long-context"],
            temperature=0.7
        ))
        
        logger.info(f"📚 Initialized model registry with {len(self.models)} default models")
    
    def register_model(self, config: ModelConfig) -> bool:
        """
        Register a new model dynamically
        
        Args:
            config: ModelConfig object
            
        Returns:
            True if registered successfully
        """
        if config.model_id in self.models:
            logger.warning(f"⚠️ Model {config.model_id} already registered, updating...")
        
        self.models[config.model_id] = config
        logger.info(f"✅ Registered model: {config.model_id} ({config.provider})")
        
        return True
    
    def unregister_model(self, model_id: str) -> bool:
        """Remove model from registry"""
        if model_id in self.models:
            del self.models[model_id]
            logger.info(f"🗑️ Unregistered model: {model_id}")
            return True
        return False
    
    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration"""
        return self.models.get(model_id)
    
    def list_models(
        self,
        provider: Optional[ModelProvider] = None,
        capability: Optional[str] = None,
        enabled_only: bool = True,
        require_gpu: Optional[bool] = None
    ) -> List[ModelConfig]:
        """
        List models with optional filters
        
        Args:
            provider: Filter by provider (ollama, openai, etc)
            capability: Filter by capability (chat, code, etc)
            enabled_only: Only return enabled models
            require_gpu: Filter by GPU requirement
        """
        models = list(self.models.values())
        
        # Apply filters
        if enabled_only:
            models = [m for m in models if m.enabled]
        
        if provider:
            models = [m for m in models if m.provider == provider]
        
        if capability:
            models = [m for m in models if capability in m.capabilities]
        
        if require_gpu is not None:
            models = [m for m in models if m.gpu_required == require_gpu]
        
        return models
    
    def enable_model(self, model_id: str):
        """Enable a model"""
        if model_id in self.models:
            self.models[model_id].enabled = True
            logger.info(f"✅ Enabled model: {model_id}")
    
    def disable_model(self, model_id: str):
        """Disable a model"""
        if model_id in self.models:
            self.models[model_id].enabled = False
            logger.info(f"⏸️ Disabled model: {model_id}")
    
    async def health_check_model(self, model_id: str) -> bool:
        """
        Health check for a model
        
        Pings the model endpoint to verify availability
        """
        config = self.models.get(model_id)
        if not config:
            return False
        
        try:
            if config.provider == ModelProvider.OLLAMA:
                # Check Ollama endpoint
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{config.endpoint}/api/tags", timeout=5.0)
                    return response.status_code == 200
            
            elif config.provider in [ModelProvider.OPENAI, ModelProvider.ANTHROPIC, ModelProvider.GOOGLE]:
                # Commercial APIs assumed available if key provided
                return config.api_key is not None
            
            else:
                # Unknown provider
                return False
                
        except Exception as e:
            logger.error(f"Health check failed for {model_id}: {e}")
            return False
    
    def get_models_for_debate(self) -> List[ModelConfig]:
        """
        Get models suitable for Multi-AI debate
        
        Returns diverse set of models with different characteristics
        """
        # Get all chat-capable models
        chat_models = self.list_models(capability="chat", enabled_only=True)
        
        # Prioritize diversity
        selected = []
        
        # 1. Try to get one from each provider
        providers_seen = set()
        for model in chat_models:
            if model.provider not in providers_seen:
                selected.append(model)
                providers_seen.add(model.provider)
        
        # 2. If not enough, add more local models
        if len(selected) < 4:
            local_models = [m for m in chat_models if m.provider == ModelProvider.OLLAMA]
            for model in local_models:
                if model not in selected:
                    selected.append(model)
                if len(selected) >= 6:
                    break
        
        logger.info(f"🎭 Selected {len(selected)} models for debate")
        return selected[:6]  # Max 6 models for debate
    
    def export_config(self) -> Dict[str, Any]:
        """Export registry as configuration"""
        return {
            "total_models": len(self.models),
            "enabled_models": len([m for m in self.models.values() if m.enabled]),
            "models": {
                model_id: {
                    "provider": config.provider.value,
                    "display_name": config.display_name,
                    "capabilities": config.capabilities,
                    "enabled": config.enabled,
                    "gpu_required": config.gpu_required,
                    "tags": config.tags
                }
                for model_id, config in self.models.items()
            }
        }


# Singleton
_model_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get or create model registry singleton"""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry

