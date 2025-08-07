"""
Data models and schemas for the Intelligent Query-Retrieval System

This module contains all Pydantic models used for request/response
validation and internal data structures.
"""

from .schemas import (
    QueryRequest,
    QueryResponse,
    DocumentChunk,
    RetrievalResult,
    DecisionResult,
    ParsedQuery,
)

__all__ = [
    "QueryRequest",
    "QueryResponse", 
    "DocumentChunk",
    "RetrievalResult",
    "DecisionResult",
    "ParsedQuery",
]

# Model version for API compatibility
SCHEMA_VERSION = "1.0.0"

def get_schema_info():
    """Get information about available schemas"""
    return {
        "version": SCHEMA_VERSION,
        "models": __all__,
        "description": "Data models for intelligent query-retrieval system"
    }
