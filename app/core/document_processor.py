import requests
import PyPDF2
import docx
import re
from io import BytesIO
from typing import List, Tuple, Dict, Any
from app.core.config import settings
from app.models.schemas import DocumentChunk

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
    
    def process_document(self, doc_url: str) -> List[DocumentChunk]:
        """Process document from URL and return chunks with metadata"""
        try:
            # Download document
            response = requests.get(doc_url, timeout=30)
            response.raise_for_status()
            
            # Detect document type
            doc_type = self._detect_document_type(doc_url)
            
            # Extract text
            raw_text = self._extract_text(response.content, doc_type)
            
            # Create semantic chunks
            chunks = self._create_semantic_chunks(raw_text)
            
            # Create DocumentChunk objects
            document_chunks = []
            for i, chunk in enumerate(chunks):
                document_chunks.append(DocumentChunk(
                    text=chunk,
                    metadata={
                        "chunk_id": f"chunk_{i}",
                        "document_url": doc_url,
                        "document_type": doc_type,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                ))
            
            return document_chunks
            
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")
    
    def _detect_document_type(self, url: str) -> str:
        """Detect document type from URL"""
        if url.lower().endswith('.pdf'):
            return 'pdf'
        elif url.lower().endswith(('.docx', '.doc')):
            return 'docx'
        else:
            return 'pdf'
    
    def _extract_text(self, content: bytes, doc_type: str) -> str:
        """Extract text based on document type"""
        if doc_type == 'pdf':
            return self._extract_pdf_text(content)
        elif doc_type == 'docx':
            return self._extract_docx_text(content)
        else:
            raise ValueError(f"Unsupported document type: {doc_type}")
    
    def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF content"""
        pdf_reader = PyPDF2.PdfReader(BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def _extract_docx_text(self, content: bytes) -> str:
        """Extract text from DOCX content"""
        doc = docx.Document(BytesIO(content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def _create_semantic_chunks(self, text: str) -> List[str]:
        """Create semantic chunks from text"""
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + " " + sentence
                else:
                    chunks.extend(self._split_long_sentence(sentence))
                    current_chunk = ""
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk"""
        words = text.split()
        overlap_words = words[-self.chunk_overlap//10:]
        return " ".join(overlap_words)
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split a long sentence into smaller chunks"""
        words = sentence.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size//10):
            chunk_words = words[i:i + self.chunk_size//10]
            chunks.append(" ".join(chunk_words))
        
        return chunks
