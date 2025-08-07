import os
from pydantic_settings import BaseSettings  # ← This is the important change

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Intelligent Query Retrieval System"
    
    # Groq Configuration
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    MAX_TOKENS: int = 1000
    
    # Vector Store Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    FAISS_INDEX_TYPE: str = "IndexFlatL2"
    
    # Processing Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_RETRIEVAL_RESULTS: int = 10
    
    # Authentication
    BEARER_TOKEN: str = "18163f3e6126c61340065e4e89c23f11e090a36427c400d76a702c7897cbfeb5"
    
    class Config:
        env_file = ".env"

settings = Settings()
