"""Logging configuration for DARWIN META-RESEARCH BRAIN."""

import logging
import sys
from typing import Optional

# Basic logging setup
def setup_logging(level: str = "INFO"):
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(f"darwin.{name}")

def get_domain_logger(domain: str, component: str) -> logging.Logger:
    """Get a domain-specific logger."""
    return logging.getLogger(f"darwin.{domain}.{component}")

__all__ = ["setup_logging", "get_logger", "get_domain_logger"]