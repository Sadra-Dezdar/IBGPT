"""Metadata schema configuration for IB documents."""

from typing import Dict, Any, Literal

# Base metadata schema
METADATA_SCHEMA = {
    "source": str,  # document filename
    "doc_type": str,  # document type
    "subject": str,  # IB subject
    "level": str,  # HL or SL
    "year": str,  # academic year
    "topic": str,  # specific topic
    "section": str,  # document section
    "chunk_index": int,  # chunk number
    "headers": str,  # hierarchical headers
}

# Valid values for metadata fields
VALID_SUBJECTS = [
    "General",  # Add this for general IB documents
    "Mathematics AA",  # Analysis and Approaches
    "Mathematics AI",  # Applications and Interpretation
    "Physics", 
    "Chemistry",
    "Biology",
    "English",
    "Spanish",
    "French",
    "Economics",
    "History",
    "Psychology",
    "Computer Science",
    "Business Management",
    "Environmental Systems and Societies",
    "Theory of Knowledge",
    "Extended Essay"
]

VALID_LEVELS = ["HL", "SL", "Core"]

VALID_DOC_TYPES = [
    "ia_guide",
    "ia_example",
    "mark_scheme",
    "syllabus",
    "general_info"
]

VALID_SECTIONS = [
    "Paper 1",
    "Paper 2",
    "Paper 3",
    "IA",
    "TOK",
    "EE",
    "CAS"
]

def create_metadata(
    source: str,
    doc_type: str,
    subject: str,
    level: str = None,
    year: str = None,
    topic: str = None,
    section: str = None,
    chunk_index: int = 0,
    headers: str = ""
) -> Dict[str, Any]:
    """Create standardized metadata dictionary."""
    metadata = {
        "source": source,
        "doc_type": doc_type,
        "subject": subject,
        "chunk_index": chunk_index,
        "headers": headers
    }
    
    # Add optional fields if provided
    if level:
        metadata["level"] = level
    if year:
        metadata["year"] = year
    if topic:
        metadata["topic"] = topic
    if section:
        metadata["section"] = section
    
    return metadata