"""
ChromaDB collection configuration for IB documents.

This module defines the structure and organization of document collections
used by the IB Student Assistant. Each collection is optimized for specific
types of IB-related content.

Collections:
- ib_general: Core IB programme information and policies
- ia_guides: Subject-specific Internal Assessment guides
- ia_examples: Exemplar IAs with scores and examiner feedback  
- mark_schemes: Detailed marking criteria for past papers
- syllabus: Official curriculum guides and subject specifications

Author: IB Student Assistant Team
Version: 2.0.0
"""


# Collection definitions with descriptions
COLLECTIONS = {
    "ib_general": "General IBDP programme information and policies",
    "ia_guides": "Internal Assessment guides organized by subject", 
    "ia_examples": "Example IAs with scores and detailed feedback",
    "mark_schemes": "Comprehensive mark schemes for past papers",
    "syllabus": "Official subject syllabi and curriculum guides"
}


# Collection distance metrics for similarity search
COLLECTION_METRICS = {
    "ib_general": "cosine",
    "ia_guides": "cosine", 
    "ia_examples": "cosine",
    "mark_schemes": "cosine",
    "syllabus": "cosine"
}


# Document type to collection mapping for automatic routing
DOC_TYPE_TO_COLLECTION = {
    "ia_guide": "ia_guides",
    "ia_example": "ia_examples",
    "mark_scheme": "mark_schemes",
    "syllabus": "syllabus",
    "general_info": "ib_general"
}