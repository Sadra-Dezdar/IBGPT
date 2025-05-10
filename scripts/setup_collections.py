"""Script to set up ChromaDB collections with proper configuration."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict
import chromadb

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.collections import COLLECTIONS, COLLECTION_METRICS
from utils.chromadb_utils import get_chroma_client, get_embedding_function

def setup_collections(db_dir: str = "./chroma_db", reset: bool = False):
    """Create all necessary ChromaDB collections."""
    
    # Ensure directory exists
    os.makedirs(db_dir, exist_ok=True)
    
    # Get ChromaDB client
    client = get_chroma_client(db_dir)
    
    # Get embedding function
    embedding_func = get_embedding_function()
    
    for collection_name, description in COLLECTIONS.items():
        try:
            # Check if collection exists
            existing_collection = client.get_collection(collection_name)
            
            if reset:
                print(f"Deleting existing collection '{collection_name}'")
                client.delete_collection(collection_name)
                raise Exception("Collection deleted for recreation")
            else:
                print(f"Collection '{collection_name}' already exists")
                continue
                
        except Exception:
            # Create collection with proper metadata
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_func,
                metadata={
                    "description": description,
                    "hnsw:space": COLLECTION_METRICS.get(collection_name, "cosine")
                }
            )
            print(f"Created collection '{collection_name}'")
    
    print("\nCollection setup complete!")
    
    # List all collections
    collections = client.list_collections()
    print(f"\nAvailable collections: {[c.name for c in collections]}")

def main():
    parser = argparse.ArgumentParser(description="Set up ChromaDB collections")
    parser.add_argument("--db-dir", default="./chroma_db", help="ChromaDB directory")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate existing collections")
    
    args = parser.parse_args()
    setup_collections(args.db_dir, args.reset)

if __name__ == "__main__":
    main()