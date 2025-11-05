"""
OAuth 2.0 endpoints for ChatGPT Custom GPT integration
Based on OpenAI's requirements for OAuth authentication
"""

import os
from fastapi import APIRouter, HTTPException, Header, Query, Form
from fastapi.responses import JSONResponse, RedirectResponse
from typing import Optional

router = APIRouter()

# OAuth Configuration
API_TOKEN = os.getenv("DARWIN_API_TOKEN", "darwin-gpt-2024-secure-key")
BASE_URL = os.getenv("BASE_URL", "https://gpt.agourakis.med.br")

# OAuth Client Configuration for ChatGPT
OAUTH_CLIENT_ID = "chatgpt-darwin-client"
OAUTH_CLIENT_SECRET = os.getenv("DARWIN_API_TOKEN", "darwin-gpt-2024-secure-key")


@router.get("/.well-known/oauth-authorization-server")
async def oauth_authorization_server():
    """
    OAuth 2.0 Authorization Server Metadata
    RFC 8414 - Required for ChatGPT OAuth integration
    """
    return {
        "issuer": BASE_URL,
        "authorization_endpoint": f"{BASE_URL}/oauth/authorize",
        "token_endpoint": f"{BASE_URL}/oauth/token",
        "token_endpoint_auth_methods_supported": [
            "client_secret_basic", 
            "client_secret_post"
        ],
        "grant_types_supported": [
            "authorization_code", 
            "client_credentials"
        ],
        "response_types_supported": ["code"],
        "scopes_supported": ["read", "write", "admin"],
        "code_challenge_methods_supported": ["S256", "plain"],
        "revocation_endpoint": f"{BASE_URL}/oauth/revoke",
        "introspection_endpoint": f"{BASE_URL}/oauth/introspect",
    }


@router.get("/oauth/authorize")
async def oauth_authorize(
    response_type: str = Query(..., description="response_type"),
    client_id: str = Query(..., description="client_id"),
    redirect_uri: str = Query(..., description="redirect_uri"),
    scope: Optional[str] = Query(None, description="scope"),
    state: Optional[str] = Query(None, description="state"),
    code_challenge: Optional[str] = Query(None, description="code_challenge"),
    code_challenge_method: Optional[str] = Query("S256", description="code_challenge_method"),
):
    """
    OAuth 2.0 Authorization Endpoint
    Generates authorization code for ChatGPT integration
    """
    
    # Validate client_id
    if client_id != OAUTH_CLIENT_ID:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid client_id: {client_id}"
        )
    
    # Validate response_type
    if response_type != "code":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_type: {response_type}. Only 'code' is supported."
        )
    
    # Generate authorization code (simplified - in production, use proper PKCE flow)
    import hashlib
    import time
    
    code_data = f"{client_id}:{redirect_uri}:{scope or ''}:{int(time.time())}"
    auth_code = hashlib.sha256(code_data.encode()).hexdigest()[:32]
    
    # Build redirect URL with authorization code
    redirect_params = {"code": auth_code}
    if state:
        redirect_params["state"] = state
    
    # Create redirect URL
    from urllib.parse import urlencode
    redirect_url = f"{redirect_uri}?{urlencode(redirect_params)}"
    
    return RedirectResponse(url=redirect_url, status_code=302)


@router.post("/oauth/token")
async def oauth_token(
    grant_type: str = Form(..., description="grant_type"),
    code: Optional[str] = Form(None, description="authorization_code"),
    redirect_uri: Optional[str] = Form(None, description="redirect_uri"),
    client_id: Optional[str] = Form(None, description="client_id"),
    client_secret: Optional[str] = Form(None, description="client_secret"),
    code_verifier: Optional[str] = Form(None, description="PKCE code_verifier"),
):
    """
    OAuth 2.0 Token Endpoint
    Exchanges authorization code for access token
    """
    
    if grant_type == "authorization_code":
        # Validate required parameters
        if not code or not redirect_uri or not client_id or not client_secret:
            raise HTTPException(
                status_code=400,
                detail="Missing required parameters for authorization_code grant"
            )
        
        # Validate client credentials
        if client_id != OAUTH_CLIENT_ID or client_secret != OAUTH_CLIENT_SECRET:
            raise HTTPException(
                status_code=401,
                detail="Invalid client credentials"
            )
        
        # For now, accept any authorization code (in production, validate against stored code)
        # In a real implementation, you'd validate the code and ensure it matches the redirect_uri
        import hashlib
        
        return {
            "access_token": OAUTH_CLIENT_SECRET,  # Use the API token as access token
            "token_type": "Bearer",
            "expires_in": 3600,  # 1 hour
            "scope": "read write admin",
            "refresh_token": f"refresh_{hashlib.sha256(code.encode()).hexdigest()[:16]}",
        }
    
    elif grant_type == "client_credentials":
        # Validate client credentials for client_credentials grant
        if client_id != OAUTH_CLIENT_ID or client_secret != OAUTH_CLIENT_SECRET:
            raise HTTPException(
                status_code=401,
                detail="Invalid client credentials"
            )
        
        return {
            "access_token": OAUTH_CLIENT_SECRET,
            "token_type": "Bearer", 
            "expires_in": 3600,
            "scope": "read write admin",
        }
    
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported grant_type: {grant_type}"
        )


@router.get("/oauth/userinfo")
async def oauth_userinfo(authorization: Optional[str] = Header(None)):
    """
    OAuth 2.0 UserInfo Endpoint
    Returns user information for the authenticated token
    """
    
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, 
            detail="Missing or invalid authorization header"
        )
    
    token = authorization.replace("Bearer ", "")
    
    # Validate token (simplified)
    if token != OAUTH_CLIENT_SECRET:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return {
        "sub": "darwin_chatgpt_user",
        "name": "DARWIN ChatGPT User",
        "preferred_username": "chatgpt",
        "email": "chatgpt@agourakis.med.br",
        "email_verified": True,
    }


@router.post("/oauth/revoke")
async def oauth_revoke(
    token: str = Form(..., description="token to revoke"),
    token_type_hint: Optional[str] = Form(None, description="access_token or refresh_token"),
):
    """
    OAuth 2.0 Token Revocation Endpoint
    RFC 7009
    """
    # For now, just return success (in production, implement actual revocation)
    return {"message": "Token revoked successfully"}


@router.post("/oauth/introspect")
async def oauth_introspect(
    token: str = Form(..., description="token to introspect"),
    token_type_hint: Optional[str] = Form(None, description="access_token or refresh_token"),
):
    """
    OAuth 2.0 Token Introspection Endpoint  
    RFC 7662
    """
    
    # Validate token
    if token == OAUTH_CLIENT_SECRET:
        return {
            "active": True,
            "scope": "read write admin",
            "client_id": OAUTH_CLIENT_ID,
            "token_type": "Bearer",
            "exp": 3600,  # expires in 1 hour
        }
    else:
        return {"active": False}
