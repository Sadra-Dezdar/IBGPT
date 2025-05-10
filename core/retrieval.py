"""Document retrieval functionality."""

from typing import Dict, Any, List, Optional
import chromadb

from utils.chromadb_utils import (
    get_collection,
    query_collection,
    format_results_as_context
)
from config.collections import DOC_TYPE_TO_COLLECTION

async def retrieve_documents(
    client: chromadb.PersistentClient,
    collection_name: str,
    query: str,
    metadata_filter: Optional[Dict[str, Any]] = None,
    n_results: int = 5
) -> List[Dict[str, Any]]:
    """Retrieve documents from a specific collection."""
    
    collection = get_collection(client, collection_name)
    
    # Convert metadata filter to ChromaDB format
    where_filter = None
    if metadata_filter and len(metadata_filter) > 0:
        # If we have multiple filters, combine them with $and
        if len(metadata_filter) > 1:
            where_filter = {
                "$and": [
                    {key: {"$eq": value}} 
                    for key, value in metadata_filter.items()
                ]
            }
        else:
            # Single filter
            key, value = next(iter(metadata_filter.items()))
            where_filter = {key: {"$eq": value}}
    
    results = query_collection(
        collection,
        query,
        n_results=n_results,
        where=where_filter
    )
    
    # Format results
    documents = []
    if results["documents"] and results["documents"][0]:
        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            documents.append({
                "content": doc,
                "metadata": metadata,
                "distance": distance,
                "relevance": 1 - distance
            })
    
    return documents

async def retrieve_by_type(
    client: chromadb.PersistentClient,
    doc_type: str,
    query: str,
    metadata_filter: Optional[Dict[str, Any]] = None,
    n_results: int = 5
) -> List[Dict[str, Any]]:
    """Retrieve documents by document type."""
    
    collection_name = DOC_TYPE_TO_COLLECTION.get(doc_type)
    if not collection_name:
        raise ValueError(f"Unknown document type: {doc_type}")
    
    return await retrieve_documents(
        client,
        collection_name,
        query,
        metadata_filter,
        n_results
    )

async def multi_collection_search(
    client: chromadb.PersistentClient,
    collections: List[str],
    query: str,
    metadata_filter: Optional[Dict[str, Any]] = None,
    n_results_per_collection: int = 3
) -> List[Dict[str, Any]]:
    """Search across multiple collections."""
    
    all_results = []
    
    for collection_name in collections:
        try:
            results = await retrieve_documents(
                client,
                collection_name,
                query,
                metadata_filter,
                n_results_per_collection
            )
            all_results.extend(results)
        except Exception as e:
            print(f"Error retrieving from {collection_name}: {e}")
    
    # Sort by relevance
    all_results.sort(key=lambda x: x["relevance"], reverse=True)
    
    return all_results