from typing import List
from app.core.vector_store import VectorStore
from app.core.query_processor import QueryProcessor
from app.models.schemas import ParsedQuery, RetrievalResult

class IntelligentRetriever:
    def __init__(self, vector_store: VectorStore, query_processor: QueryProcessor):
        self.vector_store = vector_store
        self.query_processor = query_processor
    
    def retrieve_relevant_clauses(self, query: str, parsed_info: ParsedQuery) -> List[RetrievalResult]:
        """Retrieve relevant clauses using multi-strategy approach"""
        
        # Primary semantic search
        primary_results = self.vector_store.search(query, k=15)
        
        # Entity-specific searches
        entity_results = []
        for entity in parsed_info.entities:
            entity_results.extend(self.vector_store.search(entity, k=5))
        
        # Expanded term searches
        expanded_results = []
        for term in parsed_info.expanded_terms:
            expanded_results.extend(self.vector_store.search(term, k=3))
        
        # Combine and deduplicate
        all_results = primary_results + entity_results + expanded_results
        unique_results = self._deduplicate_results(all_results)
        
        # Re-rank based on query intent
        reranked_results = self._rerank_by_intent(unique_results, parsed_info)
        
        return reranked_results[:10]
    
    def _deduplicate_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Remove duplicate results based on chunk_id"""
        seen = set()
        unique_results = []
        
        for result in results:
            if result.chunk_id not in seen:
                seen.add(result.chunk_id)
                unique_results.append(result)
        
        return unique_results
    
    def _rerank_by_intent(self, results: List[RetrievalResult], parsed_info: ParsedQuery) -> List[RetrievalResult]:
        """Re-rank results based on query intent"""
        
        intent_keywords = {
            'coverage_check': ['cover', 'covered', 'include', 'benefit', 'eligible'],
            'conditions': ['condition', 'requirement', 'provided', 'subject to', 'if'],
            'exclusions': ['exclude', 'not covered', 'except', 'limitation', 'restriction'],
            'definitions': ['mean', 'define', 'refer', 'include', 'definition'],
            'waiting_periods': ['waiting', 'period', 'months', 'years', 'after', 'before'],
            'benefits': ['benefit', 'amount', 'limit', 'maximum', 'coverage'],
            'limitations': ['limit', 'maximum', 'cap', 'restrict', 'subject to']
        }
        
        keywords = intent_keywords.get(parsed_info.intent, [])
        
        for result in results:
            intent_boost = 0
            text_lower = result.text.lower()
            
            for keyword in keywords:
                if keyword in text_lower:
                    intent_boost += 0.05
            
            result.score = min(1.0, result.score + intent_boost)
        
        return sorted(results, key=lambda x: x.score, reverse=True)
