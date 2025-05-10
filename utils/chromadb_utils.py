"""ChromaDB utility functions."""

import os
import chromadb
from chromadb.utils import embedding_functions
from typing import Dict, Any, List, Optional, Iterator
from itertools import islice

def get_chroma_client(persist_directory: str) -> chromadb.PersistentClient:
    """Get ChromaDB client."""
    os.makedirs(persist_directory, exist_ok=True)
    return chromadb.PersistentClient(path=persist_directory)

def get_embedding_function(model_name: str = "all-MiniLM-L6-v2"):
    """Get embedding function."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )

def get_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    embedding_model: str = "all-MiniLM-L6-v2"
) -> chromadb.Collection:
    """Get a collection by name."""
    embedding_func = get_embedding_function(embedding_model)
    return client.get_collection(
        name=collection_name,
        embedding_function=embedding_func
    )

def query_collection(
    collection: chromadb.Collection,
    query_text: str,
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Query a collection."""
    # Only include where parameter if it has actual filters
    query_params = {
        "query_texts": [query_text],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"]
    }
    
    # Only add where clause if it's not empty
    if where and len(where) > 0:
        query_params["where"] = where
        
    return collection.query(**query_params)


def format_results_as_context(query_results: Dict[str, Any]) -> str:
    """Format query results as context string."""
    if not query_results.get("documents") or not query_results["documents"][0]:
        return "No relevant documents found."
    
    context = "RELEVANT CONTEXT:\n\n"
    
    for i, (doc, metadata, distance) in enumerate(zip(
        query_results["documents"][0],
        query_results["metadatas"][0],
        query_results["distances"][0]
    )):
        context += f"[Document {i+1}]\n"
        
        # Add metadata
        if metadata:
            if metadata.get("source"):
                context += f"Source: {metadata['source']}\n"
            if metadata.get("subject"):
                context += f"Subject: {metadata['subject']}\n"
            if metadata.get("section"):
                context += f"Section: {metadata['section']}\n"
        
        # Add content
        context += f"Content: {doc}\n"
        context += f"Relevance: {1 - distance:.2f}\n\n"
    
    return context