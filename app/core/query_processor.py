from groq import Groq
import json
import os
from typing import List
from app.core.config import settings
from app.models.schemas import ParsedQuery

class QueryProcessor:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
    
    def parse_query(self, query: str) -> ParsedQuery:
        """Parse and analyze the input query using Groq API"""
        
        prompt = f"""
        Analyze this insurance/legal query and return a JSON response:
        
        Query: "{query}"
        
        Extract:
        1. intent: Main purpose (coverage_check, conditions, exclusions, definitions, waiting_periods, benefits, limitations)
        2. entities: Key entities mentioned (medical procedures, conditions, time periods, amounts)
        3. query_type: Type of question (yes_no, conditional, definitional, quantitative)
        4. expanded_terms: Related synonyms and terms that might appear in policy documents
        
        Return only valid JSON in this format:
        {{
            "intent": "coverage_check",
            "entities": ["knee surgery", "conditions"],
            "query_type": "conditional", 
            "expanded_terms": ["knee operation", "knee procedure", "orthopedic surgery"]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            
            return ParsedQuery(
                intent=result.get('intent', 'coverage_check'),
                entities=result.get('entities', []),
                query_type=result.get('query_type', 'yes_no'),
                expanded_terms=result.get('expanded_terms', [])
            )
            
        except Exception as e:
            # Fallback parsing
            return ParsedQuery(
                intent="coverage_check",
                entities=self._extract_entities_fallback(query),
                query_type="yes_no",
                expanded_terms=[]
            )
    
    def _extract_entities_fallback(self, query: str) -> List[str]:
        """Simple fallback entity extraction"""
        medical_terms = ['surgery', 'treatment', 'procedure', 'condition', 'disease']
        entities = []
        
        words = query.lower().split()
        for i, word in enumerate(words):
            if word in medical_terms and i > 0:
                entities.append(f"{words[i-1]} {word}")
        
        return entities
