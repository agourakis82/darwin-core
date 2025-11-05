"""
Darwin Core 2025 - API Authentication
Autenticação simples com Bearer token para Custom GPT e APIs externas
"""

import os
import logging
from typing import Optional
from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)

# Token configurável via env var
DARWIN_API_TOKEN = os.getenv("DARWIN_API_TOKEN", "darwin-secure-token-change-in-production")


async def verify_api_token(authorization: Optional[str] = Header(None)) -> bool:
    """
    Verifica Bearer token no header Authorization
    
    Formato esperado: "Authorization: Bearer <token>"
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Parse Bearer token
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = parts[1]
    
    # Verificar token
    if token != DARWIN_API_TOKEN:
        logger.warning(f"Invalid API token attempt: {token[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return True


# Dependência opcional (para usar em endpoints protegidos)
async def get_current_token(authorization: Optional[str] = Header(None)) -> str:
    """Retorna token atual após validação"""
    await verify_api_token(authorization)
    return authorization.split()[1]

