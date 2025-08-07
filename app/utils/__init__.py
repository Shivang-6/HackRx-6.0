"""
Utility functions for the Intelligent Query-Retrieval System

This module contains helper functions used across the application.
"""

from .helpers import (
    clean_text,
    calculate_similarity,
    extract_key_phrases,
    format_confidence_score,
    truncate_text,
    validate_groq_api_key,
)

__all__ = [
    "clean_text",
    "calculate_similarity", 
    "extract_key_phrases",
    "format_confidence_score",
    "truncate_text",
    "validate_groq_api_key",
]

# Utility version
UTILS_VERSION = "1.0.0"

def get_utils_info():
    """Get utility module information"""
    return {
        "version": UTILS_VERSION,
        "functions": __all__,
        "description": "Helper utilities for document processing and text analysis"
    }
