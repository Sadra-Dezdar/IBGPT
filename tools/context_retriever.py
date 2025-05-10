"""Context retrieval tool for the RAG system."""

from typing import Dict, Any, List
from pydantic_ai import RunContext, Tool

from core.retrieval import retrieve_documents, multi_collection_search
from config.collections import DOC_TYPE_TO_COLLECTION

@Tool 
async def retrieve_context(
    context: RunContext,
    query: str,
    doc_type: str = None,
    subject: str = None,
    level: str = None,
    n_results: int = 5
) -> str:
    """Retrieve relevant documents from ChromaDB based on a search query."""
    
    # Get ChromaDB client from context
    client = context.deps.chroma_client
    
    # Build metadata filter
    metadata_filter = {}
    if subject:
        metadata_filter["subject"] = subject
    if level:
        metadata_filter["level"] = level
    if doc_type:
        metadata_filter["doc_type"] = doc_type
    
    # Determine which collections to search
    if doc_type and doc_type in DOC_TYPE_TO_COLLECTION:
        # Search specific collection
        collection_name = DOC_TYPE_TO_COLLECTION[doc_type]
        results = await retrieve_documents(
            client, 
            collection_name, 
            query, 
            metadata_filter, 
            n_results
        )
    else:
        # Search across multiple collections
        collections_to_search = list(DOC_TYPE_TO_COLLECTION.values())
        results = await multi_collection_search(
            client,
            collections_to_search,
            query,
            metadata_filter,
            n_results_per_collection=2
        )
    
    # Format results as context
    if not results:
        return "No relevant documents found for the query."
    
    context_text = "RELEVANT CONTEXT:\n\n"
    
    for i, doc in enumerate(results):
        context_text += f"[Document {i+1}]\n"
        
        # Add metadata
        if doc.get("metadata"):
            metadata = doc["metadata"]
            if metadata.get("source"):
                # Extract just the filename
                source = metadata["source"].split("/")[-1]
                context_text += f"Source: {source}\n"
            if metadata.get("subject"):
                context_text += f"Subject: {metadata['subject']}\n"
            if metadata.get("section"):
                context_text += f"Section: {metadata['section']}\n"
        
        # Add content
        content = doc.get("content", "")
        # Truncate very long content
        if len(content) > 500:
            content = content[:500] + "..."
        
        context_text += f"Content: {content}\n"
        context_text += f"Relevance: {doc.get('relevance', 0):.2f}\n\n"
    
    return context_text