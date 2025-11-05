"""Multi-AI Hub - Sistema revolucionário de orchestração de múltiplas IAs."""

from .router import router, hub, initialize_multi_ai_hub, shutdown_multi_ai_hub
from .chat_orchestrator import ChatOrchestrator
from .context_bridge import ContextBridge
from .conversation_manager import ConversationManager

# Export AI clients if available
try:
    from .ai_clients import ChatGPTClient, ClaudeClient, GeminiClient
    AI_CLIENTS_AVAILABLE = True
    __all__ = [
        "router",
        "hub", 
        "initialize_multi_ai_hub",
        "shutdown_multi_ai_hub",
        "ChatOrchestrator",
        "ContextBridge", 
        "ConversationManager",
        "ChatGPTClient",
        "ClaudeClient",
        "GeminiClient"
    ]
except ImportError:
    AI_CLIENTS_AVAILABLE = False
    __all__ = [
        "router",
        "hub",
        "initialize_multi_ai_hub", 
        "shutdown_multi_ai_hub",
        "ChatOrchestrator",
        "ContextBridge",
        "ConversationManager"
    ]