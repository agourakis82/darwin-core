"""
DARWIN Core 2.0 - Settings

Centralized configuration management
"""

import os
from typing import List, Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Core configuration"""
    
    # Service
    app_name: str = "DARWIN Core 2.0"
    app_version: str = "2.0.0"
    environment: str = Field(default="production", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Network
    host: str = Field(default="0.0.0.0", env="HOST")
    http_port: int = Field(default=8090, env="HTTP_PORT")
    grpc_port: int = Field(default=50051, env="GRPC_PORT")
    
    # CORS
    cors_enabled: bool = True
    cors_origins: List[str] = ["*"]
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]
    
    # ChromaDB
    chroma_host: str = Field(default="localhost", env="CHROMA_HOST")
    chroma_port: int = Field(default=8000, env="CHROMA_PORT")
    chroma_collection: str = "darwin_semantic_memory"
    
    # Apache Pulsar
    pulsar_url: str = Field(default="pulsar://localhost:6650", env="PULSAR_URL")
    pulsar_namespace: str = "darwin/default"
    
    # OpenTelemetry
    otel_enabled: bool = Field(default=True, env="OTEL_ENABLED")
    otel_endpoint: str = Field(default="http://jaeger:4317", env="OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_service_name: str = "darwin-core"
    
    # Kubernetes
    k8s_namespace: str = Field(default="darwin", env="K8S_NAMESPACE")
    k8s_in_cluster: bool = Field(default=False, env="K8S_IN_CLUSTER")
    
    # AI Agentic
    agentic_enabled: bool = Field(default=True, env="AGENTIC_ENABLED")
    agentic_health_check_interval: int = 30
    agentic_auto_scaling: bool = True
    agentic_self_healing: bool = True
    
    # Continuous Learning
    continuous_learning_enabled: bool = True
    continuous_learning_min_interactions: int = 50
    continuous_learning_retrain_interval_hours: int = 24
    
    # Plugin Registry
    plugins: Dict[str, Dict[str, Any]] = {
        "biomaterials": {
            "host": "darwin-plugin-biomaterials",
            "port": 50052,
            "enabled": True,
            "gpu_required": True
        },
        "chemistry": {
            "host": "darwin-plugin-chemistry",
            "port": 50053,
            "enabled": False,  # Not deployed yet
            "gpu_required": True
        }
    }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings singleton"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

