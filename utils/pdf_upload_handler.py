"""PDF upload and processing utilities for student submissions."""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import streamlit as st
from pypdf import PdfReader

from core.document_processor import IBDocumentProcessor
from utils.text_processing import clean_text
from config.metadata import VALID_SUBJECTS, VALID_LEVELS

class PDFUploadHandler:
    """Handles PDF uploads for student IA submissions and other documents."""
    
    def __init__(self, temp_dir: str = "./temp_uploads"):
        """Initialize the upload handler."""
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.processor = IBDocumentProcessor()
    
    def save_uploaded_file(self, uploaded_file, submission_type: str = "ia") -> str:
        """Save uploaded file to temporary directory and return file path."""
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{submission_type}_{timestamp}_{unique_id}_{uploaded_file.name}"
        
        # Save to temp directory
        file_path = self.temp_dir / filename
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return str(file_path)
    
    def save_uploaded_pdf(self, uploaded_file, submission_type: str = "ia") -> str:
        """Save uploaded PDF to temporary directory and return file path (alias for compatibility)."""
        return self.save_uploaded_file(uploaded_file, submission_type)
    
    def extract_pdf_text(self, file_path: str) -> str:
        """Extract text from uploaded PDF."""
        try:
            reader = PdfReader(file_path)
            text = ""
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text.strip():  # Only add non-empty pages
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page_text + "\n"
            
            return clean_text(text)
        
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def analyze_ia_structure(self, text: str) -> Dict[str, Any]:
        """Analyze IA structure and extract key sections."""
        # Common IA section headers
        section_patterns = {
            "title": r"(?i)(title|research question|aim)",
            "introduction": r"(?i)(introduction|background|rationale)",
            "methodology": r"(?i)(methodology|method|approach|data collection)",
            "analysis": r"(?i)(analysis|results|findings|data analysis)",
            "evaluation": r"(?i)(evaluation|reflection|limitations|validity)",
            "conclusion": r"(?i)(conclusion|summary)",
            "bibliography": r"(?i)(bibliography|references|sources|works cited)"
        }
        
        structure = {
            "word_count": len(text.split()),
            "sections_found": [],
            "missing_sections": [],
            "structure_score": 0
        }
        
        import re
        
        # Check for each section
        for section, pattern in section_patterns.items():
            if re.search(pattern, text):
                structure["sections_found"].append(section)
            else:
                structure["missing_sections"].append(section)
        
        # Calculate structure score (percentage of sections found)
        structure["structure_score"] = len(structure["sections_found"]) / len(section_patterns) * 100
        
        return structure
    
    def validate_ia_requirements(self, text: str, subject: str, level: str = "HL") -> Dict[str, Any]:
        """Validate IA against basic requirements."""
        validation = {
            "word_count_check": False,
            "word_count": len(text.split()),
            "recommended_range": (3000, 4000) if level == "HL" else (2500, 3500),
            "issues": [],
            "recommendations": []
        }
        
        word_count = validation["word_count"]
        min_words, max_words = validation["recommended_range"]
        
        # Word count validation
        if word_count < min_words * 0.8:  # 20% under minimum
            validation["issues"].append(f"Word count ({word_count}) is significantly below recommended minimum ({min_words})")
            validation["recommendations"].append(f"Expand your analysis and discussion to reach {min_words}-{max_words} words")
        elif word_count > max_words * 1.2:  # 20% over maximum
            validation["issues"].append(f"Word count ({word_count}) exceeds recommended maximum ({max_words})")
            validation["recommendations"].append(f"Consider condensing your work to {min_words}-{max_words} words")
        else:
            validation["word_count_check"] = True
        
        # Subject-specific checks
        if subject.lower() in ["mathematics aa", "mathematics ai"]:
            if "graph" not in text.lower() and "equation" not in text.lower():
                validation["issues"].append("Mathematical IA should include graphs, equations, or mathematical notation")
                validation["recommendations"].append("Ensure your IA includes appropriate mathematical content and notation")
        
        return validation
    
    def cleanup_temp_file(self, file_path: str):
        """Remove temporary uploaded file."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            st.warning(f"Could not clean up temporary file: {str(e)}")
    
    def get_upload_session_key(self) -> str:
        """Generate a unique session key for uploads."""
        if "upload_session_id" not in st.session_state:
            st.session_state.upload_session_id = str(uuid.uuid4())
        return st.session_state.upload_session_id

def create_ia_upload_interface() -> Optional[Tuple[str, str, str, str]]:
    """Create Streamlit interface for IA uploads. Returns (file_path, text, subject, level) or None."""
    st.subheader("📝 Upload Your IA for Feedback")
    
    # Subject and level selection
    col1, col2 = st.columns(2)
    
    with col1:
        subject = st.selectbox(
            "Select your subject:",
            ["Mathematics AA", "Mathematics AI", "Physics", "Chemistry", "Biology", 
             "Economics", "Psychology", "Computer Science", "English", "History"],
            key="ia_subject"
        )
    
    with col2:
        level = st.selectbox(
            "Select your level:",
            ["HL", "SL"],
            key="ia_level"
        )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose your IA PDF file",
        type="pdf",
        help="Upload your Internal Assessment PDF for detailed feedback based on IB criteria",
        key="ia_pdf_upload"
    )
    
    if uploaded_file is not None:
        # Show file info
        st.success(f"✅ Uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        # Process the file
        with st.spinner("📄 Processing your IA..."):
            handler = PDFUploadHandler()
            
            # Save file temporarily
            file_path = handler.save_uploaded_pdf(uploaded_file, "ia")
            
            # Extract text
            text = handler.extract_pdf_text(file_path)
            
            if text:
                # Show basic analysis
                with st.expander("📊 Quick Analysis", expanded=False):
                    structure = handler.analyze_ia_structure(text)
                    validation = handler.validate_ia_requirements(text, subject, level)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Word Count", validation["word_count"])
                    
                    with col2:
                        st.metric("Structure Score", f"{structure['structure_score']:.0f}%")
                    
                    with col3:
                        st.metric("Sections Found", f"{len(structure['sections_found'])}/7")
                    
                    if structure["missing_sections"]:
                        st.warning(f"**Missing sections:** {', '.join(structure['missing_sections'])}")
                    
                    if validation["issues"]:
                        st.error("**Issues found:**")
                        for issue in validation["issues"]:
                            st.write(f"• {issue}")
                
                return file_path, text, subject, level
            else:
                st.error("Could not extract text from the PDF. Please ensure it's a text-based PDF.")
                handler.cleanup_temp_file(file_path)
                return None
    
    return None

def create_document_upload_interface() -> Optional[Tuple[str, str]]:
    """Create interface for general document uploads. Returns (file_path, text) or None."""
    st.subheader("📚 Upload Any IB Document")
    st.caption("Upload PDF documents like past papers, guides, or reference materials")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload any IB-related PDF document",
        key="general_pdf_upload"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        
        with st.spinner("📄 Processing document..."):
            handler = PDFUploadHandler()
            
            # Save file temporarily
            file_path = handler.save_uploaded_pdf(uploaded_file, "document")
            
            # Extract text
            text = handler.extract_pdf_text(file_path)
            
            if text:
                # Show basic info
                word_count = len(text.split())
                st.info(f"📄 Document processed: {word_count} words extracted")
                
                return file_path, text
            else:
                st.error("Could not extract text from the PDF.")
                handler.cleanup_temp_file(file_path)
                return None
    
    return None
