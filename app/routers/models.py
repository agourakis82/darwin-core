"""
Models Router - Dynamic Model Management

Allows adding/removing/configuring AI models at runtime
Supports HuggingFace, Ollama, vLLM, and commercial APIs
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from ..services.model_registry import get_model_registry, ModelConfig, ModelProvider
from ..services.model_training import get_model_trainer

router = APIRouter(prefix="/api/v1/models", tags=["Models"])


class ModelRegistrationRequest(BaseModel):
    """Request to register a new model"""
    model_id: str = Field(..., description="Unique model identifier")
    provider: str = Field(..., description="Provider: ollama, vllm, huggingface, openai, etc")
    display_name: str
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    capabilities: List[str] = Field(default=["chat"])
    context_length: int = Field(default=2048)
    max_tokens: int = Field(default=1024)
    gpu_required: bool = False
    gpu_memory_mb: int = 0
    description: str = ""
    tags: List[str] = Field(default=[])
    temperature: float = 0.7


@router.get("/list")
async def list_models(
    provider: Optional[str] = None,
    capability: Optional[str] = None,
    enabled_only: bool = True
):
    """
    List all registered models
    
    Query params:
    - provider: Filter by provider (ollama, openai, etc)
    - capability: Filter by capability (chat, code, etc)
    - enabled_only: Only show enabled models
    """
    registry = get_model_registry()
    
    provider_enum = ModelProvider(provider) if provider else None
    
    models = registry.list_models(
        provider=provider_enum,
        capability=capability,
        enabled_only=enabled_only
    )
    
    return {
        "total": len(models),
        "models": [
            {
                "model_id": m.model_id,
                "provider": m.provider.value,
                "display_name": m.display_name,
                "capabilities": m.capabilities,
                "enabled": m.enabled,
                "gpu_required": m.gpu_required,
                "tags": m.tags,
                "description": m.description
            }
            for m in models
        ]
    }


@router.post("/register")
async def register_model(request: ModelRegistrationRequest):
    """
    Register a new model dynamically
    
    Example:
    {
      "model_id": "llama3.1:70b",
      "provider": "ollama",
      "display_name": "Llama 3.1 70B",
      "endpoint": "http://ollama:11434",
      "capabilities": ["chat", "reasoning"],
      "gpu_required": true,
      "gpu_memory_mb": 40960,
      "tags": ["local", "large"]
    }
    """
    registry = get_model_registry()
    
    try:
        provider = ModelProvider(request.provider)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider: {request.provider}"
        )
    
    config = ModelConfig(
        model_id=request.model_id,
        provider=provider,
        display_name=request.display_name,
        endpoint=request.endpoint,
        api_key=request.api_key,
        capabilities=request.capabilities,
        context_length=request.context_length,
        max_tokens=request.max_tokens,
        gpu_required=request.gpu_required,
        gpu_memory_mb=request.gpu_memory_mb,
        description=request.description,
        tags=request.tags,
        temperature=request.temperature
    )
    
    success = registry.register_model(config)
    
    return {
        "success": success,
        "model_id": request.model_id,
        "message": f"Model {request.model_id} registered successfully"
    }


@router.delete("/{model_id}")
async def unregister_model(model_id: str):
    """Remove a model from registry"""
    registry = get_model_registry()
    
    success = registry.unregister_model(model_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    return {
        "success": True,
        "message": f"Model {model_id} unregistered"
    }


@router.post("/{model_id}/enable")
async def enable_model(model_id: str):
    """Enable a model"""
    registry = get_model_registry()
    
    if model_id not in registry.models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    registry.enable_model(model_id)
    
    return {"success": True, "model_id": model_id, "enabled": True}


@router.post("/{model_id}/disable")
async def disable_model(model_id: str):
    """Disable a model"""
    registry = get_model_registry()
    
    if model_id not in registry.models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    registry.disable_model(model_id)
    
    return {"success": True, "model_id": model_id, "enabled": False}


@router.get("/{model_id}")
async def get_model_info(model_id: str):
    """Get detailed model information"""
    registry = get_model_registry()
    
    model = registry.get_model(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    return {
        "model_id": model.model_id,
        "provider": model.provider.value,
        "display_name": model.display_name,
        "endpoint": model.endpoint,
        "capabilities": model.capabilities,
        "context_length": model.context_length,
        "max_tokens": model.max_tokens,
        "gpu_required": model.gpu_required,
        "gpu_memory_mb": model.gpu_memory_mb,
        "cpu_cores": model.cpu_cores,
        "ram_mb": model.ram_mb,
        "description": model.description,
        "tags": model.tags,
        "enabled": model.enabled,
        "temperature": model.temperature,
        "top_p": model.top_p
    }


@router.get("/{model_id}/health")
async def check_model_health(model_id: str):
    """Health check for specific model"""
    registry = get_model_registry()
    
    model = registry.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    healthy = await registry.health_check_model(model_id)
    
    return {
        "model_id": model_id,
        "healthy": healthy,
        "status": "available" if healthy else "unavailable"
    }


@router.get("/debate/candidates")
async def get_debate_candidates():
    """
    Get models selected for Multi-AI debate
    
    Returns diverse set of models for ensemble
    """
    registry = get_model_registry()
    
    candidates = registry.get_models_for_debate()
    
    return {
        "total": len(candidates),
        "models": [
            {
                "model_id": m.model_id,
                "display_name": m.display_name,
                "provider": m.provider.value,
                "role": _infer_debate_role(m),
                "temperature": m.temperature
            }
            for m in candidates
        ]
    }


def _infer_debate_role(model: ModelConfig) -> str:
    """Infer role in debate based on model characteristics"""
    if "gpt-4" in model.model_id.lower():
        return "generalist"
    elif "claude" in model.model_id.lower():
        return "specialist"
    elif "gemini" in model.model_id.lower():
        return "medical_expert"
    elif "llama" in model.model_id.lower():
        return "explorer"
    elif "mistral" in model.model_id.lower():
        return "explorer"
    elif "deepseek" in model.model_id.lower():
        return "code_expert"
    else:
        return "general"


@router.get("/export")
async def export_registry():
    """Export complete registry configuration"""
    registry = get_model_registry()
    return registry.export_config()


# ========== MODEL TRAINING ENDPOINTS ==========

class FineTuneRequest(BaseModel):
    """Request to fine-tune a model"""
    base_model: str = Field(..., description="Base model to fine-tune")
    domain: str = Field(..., description="Domain to specialize in")
    min_conversations: int = Field(default=50, description="Minimum conversations needed")
    new_model_name: Optional[str] = None


@router.post("/fine-tune")
async def fine_tune_model(request: FineTuneRequest):
    """
    Fine-tune a model using conversation history
    
    This creates a specialized DARWIN model trained on your conversations!
    
    Example:
    {
      "base_model": "llama3.1:8b",
      "domain": "biomaterials",
      "min_conversations": 100
    }
    
    Creates: darwin-biomaterials-local:latest
    """
    trainer = get_model_trainer()
    
    try:
        result = await trainer.fine_tune_from_conversations(
            base_model=request.base_model,
            domain=request.domain,
            min_conversations=request.min_conversations,
            new_model_name=request.new_model_name
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync-ollama")
async def sync_ollama():
    """
    Sync registry with Ollama models
    
    Auto-discovers and registers models installed in Ollama
    Useful after manually adding models via 'ollama pull'
    """
    trainer = get_model_trainer()
    
    result = await trainer.sync_with_ollama()
    
    return {
        "success": True,
        "synced_models": result["synced"],
        "message": f"Synced {result['synced']} models from Ollama"
    }


@router.get("/ollama/available")
async def list_ollama_models():
    """List all models available in Ollama (not just registered)"""
    trainer = get_model_trainer()
    
    models = await trainer.list_ollama_models()
    
    return {
        "total": len(models),
        "models": models
    }

