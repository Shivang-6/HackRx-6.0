from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class DocumentChunk(BaseModel):
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class RetrievalResult(BaseModel):
    text: str
    metadata: Dict[str, Any]
    score: float
    chunk_id: str

class DecisionResult(BaseModel):
    answer: str
    confidence: float
    reasoning: str
    supporting_clauses: List[str]
    decision_type: str

class ParsedQuery(BaseModel):
    intent: str
    entities: List[str]
    query_type: str
    expanded_terms: List[str]
