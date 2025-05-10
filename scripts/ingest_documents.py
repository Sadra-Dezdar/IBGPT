"""Enhanced document ingestion script for IB documents."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.collections import DOC_TYPE_TO_COLLECTION
from config.metadata import create_metadata, VALID_SUBJECTS, VALID_LEVELS, VALID_DOC_TYPES
from core.document_processor import IBDocumentProcessor

def main():
    parser = argparse.ArgumentParser(description="Ingest IB documents into ChromaDB")
    parser.add_argument("filepath", help="Path to document file or directory")
    parser.add_argument("--doc-type", required=True, help="Document type", 
                       choices=VALID_DOC_TYPES)
    parser.add_argument("--subject", required=True, help="IB subject",
                       choices=VALID_SUBJECTS)
    parser.add_argument("--level", help="HL/SL/Core", choices=VALID_LEVELS)
    parser.add_argument("--year", help="Academic year")
    parser.add_argument("--topic", help="Specific topic")
    parser.add_argument("--section", help="Document section")
    parser.add_argument("--batch", action="store_true", 
                       help="Process all PDFs in directory")
    parser.add_argument("--db-dir", default="./chroma_db", help="ChromaDB directory")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = IBDocumentProcessor(db_dir=args.db_dir)
    
    if args.batch and os.path.isdir(args.filepath):
        # Process all PDFs in directory
        pdf_files = list(Path(args.filepath).glob("**/*.pdf"))
        print(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                print(f"Processing {pdf_file}")
                processor.process_document(
                    filepath=str(pdf_file),
                    doc_type=args.doc_type,
                    subject=args.subject,
                    level=args.level,
                    year=args.year,
                    topic=args.topic,
                    section=args.section
                )
                print(f"✓ Successfully processed {pdf_file}")
            except Exception as e:
                print(f"✗ Error processing {pdf_file}: {e}")
    else:
        # Process single file
        processor.process_document(
            filepath=args.filepath,
            doc_type=args.doc_type,
            subject=args.subject,
            level=args.level,
            year=args.year,
            topic=args.topic,
            section=args.section
        )
        print(f"Successfully processed {args.filepath}")

if __name__ == "__main__":
    main()