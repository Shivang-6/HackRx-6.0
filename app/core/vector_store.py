import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from app.core.config import settings
from app.models.schemas import DocumentChunk, RetrievalResult

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.index = faiss.IndexFlatL2(settings.EMBEDDING_DIMENSION)
        self.chunks: List[DocumentChunk] = []
        self.is_trained = False
    
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add documents to the vector store"""
        if not chunks:
            return
        
        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks with embeddings
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i].tolist()
            self.chunks.append(chunk)
        
        self.is_trained = True
    
    def search(self, query: str, k: int = None) -> List[RetrievalResult]:
        """Search for similar chunks"""
        if not self.is_trained:
            return []
        
        k = k or settings.MAX_RETRIEVAL_RESULTS
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search in FAISS
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            min(k, len(self.chunks))
        )
        
        # Convert to results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                similarity_score = 1 / (1 + distance)
                
                results.append(RetrievalResult(
                    text=chunk.text,
                    metadata=chunk.metadata,
                    score=float(similarity_score),
                    chunk_id=chunk.metadata.get('chunk_id', f'chunk_{idx}')
                ))
        
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def hybrid_search(self, query: str, keywords: List[str], k: int = None) -> List[RetrievalResult]:
        """Perform hybrid search combining semantic and keyword search"""
        semantic_results = self.search(query, k)
        
        # Boost scores for chunks containing keywords
        for result in semantic_results:
            keyword_boost = 0
            text_lower = result.text.lower()
            
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    keyword_boost += 0.1
            
            result.score = min(1.0, result.score + keyword_boost)
        
        return sorted(semantic_results, key=lambda x: x.score, reverse=True)
    
    def clear(self) -> None:
        """Clear the vector store"""
        self.index.reset()
        self.chunks.clear()
        self.is_trained = False
