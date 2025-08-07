import time
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.core.config import settings
from app.models.schemas import QueryRequest, QueryResponse
from app.core.document_processor import DocumentProcessor
from app.core.vector_store import VectorStore
from app.core.query_processor import QueryProcessor
from app.core.retriever import IntelligentRetriever
from app.core.decision_engine import DecisionEngine

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="LLM-Powered Intelligent Query-Retrieval System with Groq"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != settings.BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
    return credentials.credentials

# Initialize components
doc_processor = DocumentProcessor()
query_processor = QueryProcessor()
decision_engine = DecisionEngine()

@app.get("/")
async def root():
    return {
        "message": "Intelligent Query-Retrieval System with Groq is running",
        "model": settings.GROQ_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL
    }

@app.post("/hackrx/run", response_model=QueryResponse)
async def process_queries(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """Process queries against the document using Groq API"""
    
    try:
        start_time = time.time()
        
        # Initialize vector store for this request
        vector_store = VectorStore()
        retriever = IntelligentRetriever(vector_store, query_processor)
        
        # Process document
        print(f"Processing document: {request.documents}")
        chunks = doc_processor.process_document(request.documents)
        print(f"Created {len(chunks)} document chunks")
        
        # Build vector store
        vector_store.add_documents(chunks)
        print("Vector store built successfully")
        
        # Process each question
        answers = []
        for i, question in enumerate(request.questions):
            print(f"Processing question {i+1}/{len(request.questions)}: {question}")
            
            try:
                # Parse query using Groq
                parsed_info = query_processor.parse_query(question)
                
                # Retrieve relevant clauses
                relevant_clauses = retriever.retrieve_relevant_clauses(
                    question, parsed_info
                )
                
                # Make decision using Groq
                decision = decision_engine.evaluate_coverage(
                    question, relevant_clauses, parsed_info
                )
                
                # Generate final answer
                final_answer = decision_engine.generate_explanation(
                    decision, relevant_clauses
                )
                
                answers.append(final_answer)
                
            except Exception as e:
                print(f"Error processing question {i+1}: {str(e)}")
                answers.append(f"Unable to process this question: {str(e)}")
        
        processing_time = time.time() - start_time
        print(f"Total processing time: {processing_time:.2f} seconds")
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        print(f"Error in main processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "groq_model": settings.GROQ_MODEL
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

