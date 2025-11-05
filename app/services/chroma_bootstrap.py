"""
ChromaDB Bootstrap - Tenant and Database Setup

Ensures tenant and database exist in ChromaDB v2
"""

import os
import logging
from typing import Optional
import chromadb
from chromadb.config import Settings

logger = logging.getLogger("darwin.chroma_bootstrap")

TENANT = os.getenv("CHROMA_TENANT", "darwin")
DATABASE = os.getenv("CHROMA_DATABASE", "memory")


def get_chroma_settings() -> Settings:
    """Get ChromaDB settings from environment"""
    return Settings(
        chroma_server_host=os.getenv("CHROMA_HOST", "172.17.0.1"),
        chroma_server_http_port=int(os.getenv("CHROMA_PORT", "8003")),
        chroma_server_ssl_enabled=os.getenv("CHROMA_SSL", "false").lower() == "true",
    )


def ensure_tenant_and_database() -> bool:
    """
    Ensures tenant and database exist in ChromaDB v2
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        settings = get_chroma_settings()
        admin = chromadb.AdminClient(settings=settings)
        
        # Create/get tenant (idempotent)
        try:
            admin.get_tenant(TENANT)
            logger.info(f"✅ Tenant '{TENANT}' already exists")
        except Exception:
            admin.create_tenant(TENANT)
            logger.info(f"✅ Tenant '{TENANT}' created")
        
        # Create/get database (idempotent)
        try:
            admin.get_database(name=DATABASE, tenant=TENANT)
            logger.info(f"✅ Database '{DATABASE}' in tenant '{TENANT}' already exists")
        except Exception:
            admin.create_database(name=DATABASE, tenant=TENANT)
            logger.info(f"✅ Database '{DATABASE}' created in tenant '{TENANT}'")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to bootstrap ChromaDB: {e}")
        return False


def get_chroma_client() -> Optional[chromadb.HttpClient]:
    """
    Get ChromaDB HttpClient configured for DARWIN tenant/database
    
    Returns:
        HttpClient or None if connection fails
    """
    try:
        settings = get_chroma_settings()
        
        client = chromadb.HttpClient(
            host=settings.chroma_server_host,
            port=settings.chroma_server_http_port,
            ssl=settings.chroma_server_ssl_enabled,
            tenant=TENANT,
            database=DATABASE,
        )
        
        # Test connection
        client.heartbeat()
        logger.info(f"✅ ChromaDB client connected to {TENANT}/{DATABASE}")
        
        return client
        
    except Exception as e:
        logger.error(f"❌ Failed to create ChromaDB client: {e}")
        return None


# Initialize on import
_bootstrap_success = ensure_tenant_and_database()

if __name__ == "__main__":
    # Test
    ensure_tenant_and_database()
    client = get_chroma_client()
    if client:
        print("✅ ChromaDB bootstrap successful!")
        print(f"Tenant: {TENANT}, Database: {DATABASE}")
    else:
        print("❌ ChromaDB bootstrap failed!")

