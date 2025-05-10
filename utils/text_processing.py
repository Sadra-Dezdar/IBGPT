"""Text processing utilities for IB documents."""

import re
from typing import List, Tuple

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might interfere with search
    text = re.sub(r'[^\w\s\-.,!?;:()"\']', '', text)
    return text.strip()

def extract_sections(text: str) -> List[Tuple[str, str]]:
    """Extract sections with headers from text."""
    # Pattern for common IB document headers
    header_pattern = r'^(?:#{1,3}|\d+\.?\d*)\s*(.+?)$'
    
    sections = []
    lines = text.split('\n')
    current_header = ""
    current_content = []
    
    for line in lines:
        header_match = re.match(header_pattern, line, re.MULTILINE)
        if header_match:
            # Save previous section if exists
            if current_header and current_content:
                sections.append((current_header, '\n'.join(current_content)))
            # Start new section
            current_header = header_match.group(1)
            current_content = []
        else:
            current_content.append(line)
    
    # Add final section
    if current_header and current_content:
        sections.append((current_header, '\n'.join(current_content)))
    
    return sections

def extract_ib_keywords(text: str) -> List[str]:
    """Extract IB-specific keywords from text."""
    ib_keywords = [
        'criterion', 'criteria', 'assessment', 'ia', 'internal assessment',
        'tok', 'theory of knowledge', 'ee', 'extended essay', 'cas',
        'hl', 'sl', 'higher level', 'standard level', 'mark scheme',
        'command term', 'analyze', 'evaluate', 'discuss', 'explain'
    ]
    
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in ib_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)
    
    return list(set(found_keywords))