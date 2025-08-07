from groq import Groq
import json
from typing import List
from app.core.config import settings
from app.models.schemas import RetrievalResult, DecisionResult, ParsedQuery


class DecisionEngine:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
    
    def evaluate_coverage(self, query: str, relevant_clauses: List[RetrievalResult], parsed_info: ParsedQuery) -> DecisionResult:
        """Evaluate coverage based on relevant clauses using Groq API"""
        
        context = self._prepare_context(relevant_clauses)
        
        prompt = f"""
        You are an expert insurance policy analyst. Based on the following policy clauses, provide a comprehensive answer to this question.
        
        Question: {query}
        
        Policy Clauses:
        {context}
        
        Provide a clear, direct answer. Do not use JSON format in your response.
        Include:
        1. Direct answer to the question
        2. Specific conditions or limitations  
        3. Relevant policy references
        4. Your reasoning
        
        Be specific about conditions, limitations, and requirements.
        Include relevant waiting periods, exclusions, or special conditions.
        Quote specific policy terms when relevant.
        If information is not available in the clauses, state that clearly.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=settings.MAX_TOKENS,
                temperature=0.1
            )
            
            answer_text = response.choices[0].message.content.strip()
            
            return DecisionResult(
                answer=answer_text,
                confidence=0.8,  # Set a reasonable default
                reasoning=f"Analysis based on available policy clauses for: {query}",
                supporting_clauses=[clause.chunk_id for clause in relevant_clauses[:3]],
                decision_type=parsed_info.intent
            )
            
        except Exception as e:
            return DecisionResult(
                answer=f"Based on the available policy information, I need more specific details to provide a complete answer to: {query}",
                confidence=0.3,
                reasoning=f"Error in processing, but attempted analysis: {str(e)}",
                supporting_clauses=[],
                decision_type=parsed_info.intent
            )

    def _prepare_context(self, clauses: List[RetrievalResult]) -> str:
        """Prepare context string from retrieval results"""
        context_parts = []
        
        for i, clause in enumerate(clauses[:5]):
            context_parts.append(f"Clause {i+1} (Relevance: {clause.score:.2f}):\n{clause.text}\n")
        
        return "\n".join(context_parts)
    
    def generate_explanation(self, decision: DecisionResult, relevant_clauses: List[RetrievalResult]) -> str:
        """Generate a clear explanation for the decision"""
        
        explanation = f"{decision.answer}\n\n"
        
        if decision.reasoning:
            explanation += f"Reasoning: {decision.reasoning}\n\n"
        
        if relevant_clauses:
            explanation += "This answer is based on the following policy sections:\n"
            for i, clause in enumerate(relevant_clauses[:3]):
                explanation += f"- Section {i+1}: {clause.text[:200]}...\n"
        
        return explanation.strip()
