import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\-.,;:!?()\[\]{}]', '', text)
    return text.strip()

def calculate_similarity(text1: str, text2: str) -> float:
    """Simple word-based similarity calculation"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def extract_key_phrases(text: str) -> List[str]:
    """Extract key phrases from text"""
    phrases = []
    pattern = r'\b[A-Z][a-z]*\s+[a-z]+(?:\s+[a-z]+)*\b'
    matches = re.findall(pattern, text)
    
    for match in matches:
        if len(match) > 3:
            phrases.append(match.strip())
    
    return list(set(phrases))

def format_confidence_score(confidence: float) -> str:
    """Format confidence score as percentage"""
    return f"{confidence * 100:.1f}%"

def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > 0:
        truncated = truncated[:last_space]
    
    return truncated + "..."

def validate_groq_api_key(api_key: str) -> bool:
    """Validate Groq API key format"""
    # Basic validation - Groq API keys typically start with 'gsk_'
    return api_key.startswith('gsk_') and len(api_key) > 20
