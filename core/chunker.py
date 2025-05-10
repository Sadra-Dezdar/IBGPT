"""Smart chunking implementation for IB documents."""

import re
from typing import List
from abc import ABC, abstractmethod

class BaseChunker(ABC):
    """Base class for document chunkers."""
    
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        pass

class SmartIBChunker:
    """Smart chunker that handles different IB document types."""
    
    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size
        self.chunkers = {
            "ia_guide": IAGuideChunker(max_chunk_size),
            "ia_example": GeneralChunker(max_chunk_size),  # Use general for now
            "mark_scheme": MarkSchemeChunker(max_chunk_size),
            "syllabus": SyllabusChunker(max_chunk_size),
            "general_info": GeneralChunker(max_chunk_size)
        }
    
    def chunk_by_type(self, text: str, doc_type: str) -> List[str]:
        """Chunk text based on document type."""
        chunker = self.chunkers.get(doc_type, self.chunkers["general_info"])
        return chunker.chunk(text)

class IAGuideChunker(BaseChunker):
    """Chunker for IA guide documents."""
    
    def __init__(self, max_chunk_size: int):
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, text: str) -> List[str]:
        """Chunk by assessment criteria."""
        criteria_pattern = r"Criterion [A-Z]:|Assessment Criteria [A-Z]:"
        sections = re.split(criteria_pattern, text)
        
        chunks = []
        for section in sections:
            if len(section.strip()) > self.max_chunk_size:
                # Further split large sections
                sub_chunks = self._split_by_size(section)
                chunks.extend(sub_chunks)
            else:
                chunks.append(section.strip())
        
        return [chunk for chunk in chunks if chunk]
    
    def _split_by_size(self, text: str) -> List[str]:
        """Split text by size while preserving sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.max_chunk_size:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

class MarkSchemeChunker(BaseChunker):
    """Chunker for mark schemes."""
    
    def __init__(self, max_chunk_size: int):
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, text: str) -> List[str]:
        """Keep question-answer pairs together."""
        qa_pattern = r"Question \d+.*?(?=Question \d+|$)"
        qa_pairs = re.findall(qa_pattern, text, flags=re.DOTALL)
        
        # If no pattern found, fall back to general chunking
        if not qa_pairs:
            return GeneralChunker(self.max_chunk_size).chunk(text)
        
        return [pair.strip() for pair in qa_pairs if pair.strip()]

class SyllabusChunker(BaseChunker):
    """Chunker for syllabus documents."""
    
    def __init__(self, max_chunk_size: int):
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, text: str) -> List[str]:
        """Chunk by topic and subtopic."""
        topic_pattern = r"Topic \d+:|Unit \d+:|Chapter \d+:"
        topics = re.split(topic_pattern, text)
        
        chunks = []
        for topic in topics:
            if topic.strip():
                # Check for subtopics
                subtopic_pattern = r"\d+\.\d+|[A-Z]\.\d+"
                subtopics = re.split(subtopic_pattern, topic)
                
                if len(subtopics) > 1:
                    chunks.extend([st.strip() for st in subtopics if st.strip()])
                else:
                    chunks.append(topic.strip())
        
        # If no pattern found, fall back to general chunking
        if not chunks:
            return GeneralChunker(self.max_chunk_size).chunk(text)
        
        return chunks

class GeneralChunker(BaseChunker):
    """Default chunker for general documents."""
    
    def __init__(self, max_chunk_size: int):
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, text: str) -> List[str]:
        """Simple chunking by size."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # +1 for space
            if current_size + word_size > self.max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks