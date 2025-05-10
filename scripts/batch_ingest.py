"""Batch ingestion script for organizing and processing IB documents."""

import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Document type mappings based on filename patterns
FILENAME_PATTERNS = {
    "ia_guide": ["ia_guide", "internal_assessment", "ia_criteria"],
    "mark_scheme": ["mark_scheme", "markscheme", "ms_", "marking_scheme"],
    "syllabus": ["syllabus", "curriculum", "_guide_"],
    "ia_example": ["ia_example", "sample_ia", "exemplar"]
}

# Subject detection patterns - IMPROVED
SUBJECT_PATTERNS = {
    "Mathematics AA": [
        "mathematics_analysis_and_approaches", 
        "math_aa", 
        "mathematics_aa", 
        "analysis_approaches", 
        "maths_aa",
        "mathematics aa",
        "math aa",
        "analysis and approaches"
    ],
    "Mathematics AI": [
        "mathematics_applications_and_interpretation",
        "math_ai", 
        "mathematics_ai", 
        "applications_interpretation", 
        "maths_ai",
        "mathematics ai",
        "math ai",
        "applications and interpretation"
    ],
    "Physics": ["physics"],
    "Chemistry": ["chemistry", "chem"],
    "Biology": ["biology", "bio"],
    "English": ["english", "language_a"],
    "Economics": ["economics", "econ"],
    "Psychology": ["psychology", "psych"],
    "Computer Science": ["computer_science", "cs", "comp_sci"],
    "History": ["history"],
    "Business Management": ["business", "management"]
}

def detect_document_type(filename: str) -> str:
    """Detect document type from filename."""
    filename_lower = filename.lower()
    
    # Check for past papers (which we're treating as general_info since we removed past_paper)
    if "paper_" in filename_lower and "markscheme" not in filename_lower:
        return "general_info"
    
    # Check for mark schemes
    if "markscheme" in filename_lower or "mark_scheme" in filename_lower:
        return "mark_scheme"
    
    # Check for guides
    if "guide" in filename_lower and "ia" not in filename_lower:
        return "syllabus"
    
    # Check for formula booklets
    if "formula" in filename_lower and "booklet" in filename_lower:
        return "general_info"
    
    # Check patterns
    for doc_type, patterns in FILENAME_PATTERNS.items():
        if any(pattern in filename_lower for pattern in patterns):
            return doc_type
    
    return "general_info"

def detect_subject(filename: str) -> str:
    """Detect subject from filename."""
    filename_lower = filename.lower()
    
    # Special case for Mathematics documents
    if "mathematics_analysis_and_approaches" in filename_lower or "math_aa" in filename_lower:
        return "Mathematics AA"
    
    if "mathematics_applications_and_interpretation" in filename_lower or "math_ai" in filename_lower:
        return "Mathematics AI"
    
    # General mathematics
    if "mathematics" in filename_lower or "math" in filename_lower:
        # Try to determine AA or AI
        if any(term in filename_lower for term in ["analysis", "aa"]):
            return "Mathematics AA"
        elif any(term in filename_lower for term in ["applications", "ai"]):
            return "Mathematics AI"
        else:
            # Default to AA for general math documents
            return "Mathematics AA"
    
    # Check other subjects
    for subject, patterns in SUBJECT_PATTERNS.items():
        if any(pattern in filename_lower for pattern in patterns):
            return subject
    
    return "General"

def detect_level(filename: str) -> str:
    """Detect level from filename."""
    filename_lower = filename.lower()
    
    if "hl" in filename_lower or "higher" in filename_lower:
        return "HL"
    elif "sl" in filename_lower or "standard" in filename_lower:
        return "SL"
    
    return None

def detect_year(filename: str) -> str:
    """Extract year from filename."""
    import re
    
    # Look for patterns like M21, N21, M20, etc.
    session_pattern = r'[MN]\d{2}'
    match = re.search(session_pattern, filename)
    if match:
        session = match.group(0)
        year_suffix = session[1:]
        # Convert to full year (e.g., 21 -> 2021)
        return f"20{year_suffix}"
    
    # Look for 4-digit years
    year_match = re.search(r"20\d{2}", filename)
    if year_match:
        return year_match.group(0)
    
    return None

def process_directory(directory: str):
    """Process all PDFs in directory structure."""
    base_path = Path(directory)
    
    # Find all PDFs
    pdf_files = list(base_path.glob("**/*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        filename = pdf_file.name
        
        # Auto-detect properties
        doc_type = detect_document_type(filename)
        subject = detect_subject(filename)
        level = detect_level(filename)
        year = detect_year(filename)
        
        print(f"\nProcessing: {filename}")
        print(f"  Type: {doc_type}")
        print(f"  Subject: {subject}")
        print(f"  Level: {level}")
        print(f"  Year: {year}")
        
        # Construct command
        cmd = [
            sys.executable,  # Use current Python executable
            "scripts/ingest_documents.py",
            str(pdf_file),
            "--doc-type", doc_type,
            "--subject", subject
        ]
        
        if level:
            cmd.extend(["--level", level])
        if year:
            cmd.extend(["--year", year])
        
        # Run ingestion
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Success")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process IB documents")
    parser.add_argument("directory", help="Directory containing PDFs")
    
    args = parser.parse_args()
    process_directory(args.directory)