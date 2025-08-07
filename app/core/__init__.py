"""
Core module for the Intelligent Query-Retrieval System

This module contains the core components for document processing,
vector storage, query processing, retrieval, and decision making.
"""

from .config import settings
from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .query_processor import QueryProcessor
from .retriever import IntelligentRetriever
from .decision_engine import DecisionEngine

__all__ = [
    "settings",
    "DocumentProcessor",
    "VectorStore", 
    "QueryProcessor",
    "IntelligentRetriever",
    "DecisionEngine",
]

# Version information
__version__ = "1.0.0"
__author__ = "Intelligent Query System Team"
__description__ = "LLM-Powered Intelligent Query-Retrieval System with Groq API"

# Module-level configuration
def get_system_info():
    """Get system information and configuration"""
    return {
        "version": __version__,
        "groq_model": settings.GROQ_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "max_tokens": settings.MAX_TOKENS,
        "chunk_size": settings.CHUNK_SIZE,
        "max_retrieval_results": settings.MAX_RETRIEVAL_RESULTS
    }

def validate_configuration():
    """Validate core configuration settings"""
    errors = []
    
    if not settings.GROQ_API_KEY:
        errors.append("GROQ_API_KEY is not set")
    
    if not settings.GROQ_API_KEY.startswith('gsk_'):
        errors.append("Invalid GROQ_API_KEY format")
    
    if settings.CHUNK_SIZE <= 0:
        errors.append("CHUNK_SIZE must be greater than 0")
    
    if settings.MAX_TOKENS <= 0:
        errors.append("MAX_TOKENS must be greater than 0")
    
    if settings.MAX_RETRIEVAL_RESULTS <= 0:
        errors.append("MAX_RETRIEVAL_RESULTS must be greater than 0")
    
    return errors
