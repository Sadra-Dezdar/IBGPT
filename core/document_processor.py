"""Document processing functionality."""

import os
import pathlib
from typing import List, Dict, Any, Optional

from utils.chromadb_utils import get_chroma_client, get_or_create_collection, add_documents_to_collection
from core.chunker import SmartIBChunker

class IBDocumentProcessor:
    def __init__(self, db_dir: str = "./chroma_db"):
        self.client = get_chroma_client(db_dir)
        self.chunker = SmartIBChunker()
    
    def process_document(self, filepath: str, doc_type: str, 
                        subject: str, level: str = None, year: str = None,
                        topic: str = None, section: str = None) -> None:
        """Process a single document and add to appropriate collection."""
        
        # Determine collection based on doc_type
        collection_name = self._get_collection_name(doc_type)
        
        # Extract text based on file type
        if filepath.endswith('.pdf'):
            text = self._extract_pdf_text(filepath)
        else:
            text = self._extract_text_file(filepath)
        
        # Smart chunking based on document type
        chunks = self.chunker.chunk_by_type(text, doc_type)
        
        # Prepare for insertion
        ids, documents, metadatas = [], [], []
        
        for i, chunk in enumerate(chunks):
            # Extract section info from chunk
            section_info = self._extract_section_info(chunk, doc_type)
            
            metadata = {
                "source": filepath,
                "doc_type": doc_type,
                "subject": subject,
                "level": level,
                "year": year,
                "topic": topic,
                "section": section,
                "chunk_index": i,
                **section_info
            }
            
            # Remove None values from metadata
            metadata = {k: v for k, v in metadata.items() if v is not None}
            
            ids.append(f"{filepath}-chunk-{i}")
            documents.append(chunk)
            metadatas.append(metadata)
        
        # Add to collection
        collection = get_or_create_collection(
            self.client, 
            collection_name,
            embedding_model_name="all-MiniLM-L6-v2"
        )
        
        add_documents_to_collection(
            collection, ids, documents, metadatas
        )
    
    def _get_collection_name(self, doc_type: str) -> str:
        """Map document type to collection name."""
        from config.collections import DOC_TYPE_TO_COLLECTION
        return DOC_TYPE_TO_COLLECTION.get(doc_type, "ib_general")
    
    def _extract_pdf_text(self, filepath: str) -> str:
        """Extract text from PDF file."""
        try:
            import PyPDF2
            
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            # Fallback to using pypdf
            from pypdf import PdfReader
            
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    
    def _extract_text_file(self, filepath: str) -> str:
        """Extract text from text file."""
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _extract_section_info(self, chunk: str, doc_type: str) -> Dict[str, Any]:
        """Extract additional metadata from chunk content."""
        import re
        
        headers = re.findall(r'^#+\s+(.+)$', chunk, re.MULTILINE)
        header_str = '; '.join(headers) if headers else ''
        
        return {
            "headers": header_str,
            "char_count": len(chunk),
            "word_count": len(chunk.split())
        }