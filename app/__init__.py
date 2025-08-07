"""
Intelligent Query-Retrieval System

A FastAPI-based system for processing documents and answering queries
using LLM capabilities with Groq API and vector search.
"""

__version__ = "1.0.0"
__title__ = "Intelligent Query-Retrieval System"
__description__ = "LLM-Powered document query system with Groq API"
__author__ = "Development Team"

# Import core components for easy access
from .core import (
    settings,
    DocumentProcessor,
    VectorStore,
    QueryProcessor,
    IntelligentRetriever,
    DecisionEngine,
)

from .models import (
    QueryRequest,
    QueryResponse,
    DocumentChunk,
    RetrievalResult,
    DecisionResult,
    ParsedQuery,
)

__all__ = [
    # Core components
    "settings",
    "DocumentProcessor",
    "VectorStore",
    "QueryProcessor", 
    "IntelligentRetriever",
    "DecisionEngine",
    
    # Data models
    "QueryRequest",
    "QueryResponse",
    "DocumentChunk",
    "RetrievalResult", 
    "DecisionResult",
    "ParsedQuery",
]

def get_app_info():
    """Get application information"""
    return {
        "title": __title__,
        "version": __version__,
        "description": __description__,
        "author": __author__,
    }
