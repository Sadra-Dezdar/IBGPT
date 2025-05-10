"""ChromaDB collection configuration for IB documents."""

# Collection definitions
COLLECTIONS = {
    "ib_general": "General IBDP programme information",
    "ia_guides": "Internal Assessment guides by subject", 
    "ia_examples": "Example IAs with scores and feedback",
    "mark_schemes": "Mark schemes for past papers",
    "syllabus": "Subject syllabi and curriculum guides"
}

# Collection distance metrics
COLLECTION_METRICS = {
    "ib_general": "cosine",
    "ia_guides": "cosine",
    "ia_examples": "cosine",
    "mark_schemes": "cosine",
    "syllabus": "cosine"
}

# Document type to collection mapping
DOC_TYPE_TO_COLLECTION = {
    "ia_guide": "ia_guides",
    "ia_example": "ia_examples",
    "mark_scheme": "mark_schemes",
    "syllabus": "syllabus",
    "general_info": "ib_general"
}