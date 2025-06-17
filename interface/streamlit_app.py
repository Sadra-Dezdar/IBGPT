"""
Streamlit Web Interface for IB Student Assistant

This is the main web application interface that provides an interactive
platform for IB students to:
- Ask questions about the IB programme
- Upload and analyze IA documents
- Get personalized feedback and guidance
- Access comprehensive IB resources

Features:
- Real-time chat interface
- PDF/DOCX file upload support
- IA assessment and feedback
- Multi-agent AI processing
- Responsive design

Usage:
    streamlit run interface/streamlit_app.py

Author: IB Student Assistant Team
Version: 2.0.0
"""

import os
import sys
import warnings
import streamlit as st
import asyncio
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress PyTorch warnings before importing ML libraries
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
os.environ["TORCH_DISABLE_AUTOGRAD"] = "1"

# Import core system components
from agents.multi_agent_system_no_tools import MultiAgentSystem, MultiAgentDeps
from utils.chromadb_utils import get_chroma_client
from utils.ia_assessor import IAAssessor, format_assessment_for_display

# Initialize system
@st.cache_resource
def init_system():
    """Initialize the multi-agent system."""
    return MultiAgentSystem()

@st.cache_resource
def init_deps():
    """Initialize dependencies."""
    return MultiAgentDeps(
        chroma_client=get_chroma_client("./chroma_db")
    )

@st.cache_resource
def init_ia_assessor():
    """Initialize IA assessor."""
    return IAAssessor("./chroma_db")

async def handle_query(query: str, uploaded_file=None, file_description=None):
    """Handle user query asynchronously with optional file upload."""
    system = init_system()
    deps = init_deps()
    
    # If file is uploaded, process it and enhance the query
    if uploaded_file is not None:
        try:
            from utils.pdf_upload_handler import PDFUploadHandler
            from utils.ia_assessor import IAAssessor
            
            handler = PDFUploadHandler()
            
            # Save uploaded file temporarily
            file_path = handler.save_uploaded_file(uploaded_file)
            
            # Extract text from file
            if file_path.endswith('.pdf'):
                file_text = handler.extract_pdf_text(file_path)
            else:
                # For other file types, read as text
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_text = f.read()
                except UnicodeDecodeError:
                    # Try with different encoding
                    with open(file_path, 'r', encoding='latin-1') as f:
                        file_text = f.read()
            
            if not file_text or len(file_text.strip()) < 100:
                handler.cleanup_temp_file(file_path)
                return {
                    "response": "I couldn't extract meaningful text from the uploaded file. Please ensure it's a text-based PDF or document.",
                    "thinking": "File text extraction failed or insufficient content",
                    "classification": {"query_type": "error", "subject": None, "level": None}
                }
            
            # Check if this seems like an IA that needs assessment
            is_ia_assessment = (
                any(term in query.lower() for term in ['assess', 'feedback', 'grade', 'criteria', 'score']) and
                any(term in query.lower() for term in ['ia', 'internal assessment']) and
                len(file_text) > 1000  # Reasonable length for an IA
            )
            
            if is_ia_assessment:
                # Extract subject and level from description or guess from content
                subject = "Mathematics AA"  # Default
                level = "HL"  # Default
                
                # Try to detect subject and level from file description or content
                if file_description:
                    desc_lower = file_description.lower()
                    # Detect subject
                    if "math" in desc_lower:
                        if "ai" in desc_lower:
                            subject = "Mathematics AI"
                        else:
                            subject = "Mathematics AA"
                    elif "physics" in desc_lower:
                        subject = "Physics"
                    elif "chemistry" in desc_lower:
                        subject = "Chemistry"
                    elif "biology" in desc_lower:
                        subject = "Biology"
                    elif "economics" in desc_lower:
                        subject = "Economics"
                    elif "psychology" in desc_lower:
                        subject = "Psychology"
                    
                    # Detect level
                    if "sl" in desc_lower or "standard" in desc_lower:
                        level = "SL"
                    elif "hl" in desc_lower or "higher" in desc_lower:
                        level = "HL"
                
                # Also check first part of document for subject/level clues
                file_start = file_text[:500].lower()
                if "mathematics ai" in file_start or "math ai" in file_start:
                    subject = "Mathematics AI"
                elif "mathematics aa" in file_start or "math aa" in file_start:
                    subject = "Mathematics AA"
                elif "physics" in file_start:
                    subject = "Physics"
                elif "chemistry" in file_start:
                    subject = "Chemistry"
                elif "biology" in file_start:
                    subject = "Biology"
                
                if "standard level" in file_start or " sl " in file_start:
                    level = "SL"
                elif "higher level" in file_start or " hl " in file_start:
                    level = "HL"
                
                # Use IA Assessor for comprehensive assessment
                try:
                    assessor = init_ia_assessor()
                    assessment = await assessor.assess_ia(file_text, subject, level)
                    
                    # Format assessment response
                    from utils.ia_assessor import format_assessment_for_display
                    assessment_response = format_assessment_for_display(assessment)
                    
                    # Clean up temporary file
                    handler.cleanup_temp_file(file_path)
                    
                    return {
                        "response": assessment_response,
                        "thinking": f"Processed IA assessment for {subject} {level} document with {len(file_text)} characters",
                        "classification": {"query_type": "ia_assessment", "subject": subject, "level": level}
                    }
                except Exception as assessment_error:
                    # Fallback to multi-agent system for IA assessment
                    enhanced_query = f"""Please assess this {subject} {level} Internal Assessment:

User's question: {query}

IA Content (first 3000 characters):
{file_text[:3000]}{"..." if len(file_text) > 3000 else ""}

Please provide detailed feedback based on IB assessment criteria including:
- Overall grade prediction
- Strengths and areas for improvement
- Specific suggestions for enhancement
"""
                    result = await system.process_query(enhanced_query, deps)
                    handler.cleanup_temp_file(file_path)
                    return result
            else:
                # For general document analysis, enhance query with file content
                enhanced_query = f"""User uploaded a document and asked: {query}

{f"File description: {file_description}" if file_description else ""}

Document content (first 2000 characters):
{file_text[:2000]}{"..." if len(file_text) > 2000 else ""}

Please analyze the document content and answer the user's question based on both the document and your IB knowledge."""
                
                # Clean up temporary file
                handler.cleanup_temp_file(file_path)
                
                # Process enhanced query
                result = await system.process_query(enhanced_query, deps)
                return result
                
        except Exception as e:
            return {
                "response": f"I encountered an error processing your uploaded file: {str(e)}. Please try uploading the file again or ask your question without the file.",
                "thinking": f"File processing error: {str(e)}",
                "classification": {"query_type": "error", "subject": None, "level": None}
            }
    else:
        # Normal query without file
        result = await system.process_query(query, deps)
        return result

# Streamlit UI
st.set_page_config(
    page_title="IB Student Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 IB Student Assistant")

# Initialize session state for current conversation only
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_thinking" not in st.session_state:
    st.session_state.show_thinking = True

# Initialize session state for file upload
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

# Main chat interface with integrated file upload
st.markdown("### 🎓 Ask questions, upload IAs for assessment, or analyze any IB document!")

# File upload section above chat
with st.container():
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📎 Upload PDF or Word document (optional)",
            type=['pdf', 'docx', 'txt'],
            help="Upload your IA for assessment or any IB document for analysis",
            key="file_uploader"
        )
    
    with col2:
        if uploaded_file:
            file_description = st.text_input(
                "File description (optional):",
                placeholder="e.g., Math AA HL IA on calculus",
                help="Briefly describe what this document is"
            )
        else:
            file_description = None

# Display upload status
if uploaded_file:
    if uploaded_file != st.session_state.uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.file_processed = False
    
    if not st.session_state.file_processed:
        st.info(f"📄 **File ready:** {uploaded_file.name} ({uploaded_file.size:,} bytes)")
        st.markdown("💡 **Tip:** Ask questions like *'Assess my IA'*, *'What are the key points in this document?'*, or *'How can I improve this?'*")
        st.session_state.file_processed = True

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show thinking in expandable section if available and enabled
        if (message["role"] == "assistant" and 
            message.get("thinking") and 
            message["thinking"].strip() and
            st.session_state.show_thinking):
            with st.expander("🤔 Model Thinking Process", expanded=False):
                st.markdown(f"```\n{message['thinking']}\n```")

# Chat input with file processing
if prompt := st.chat_input("Ask your IB question or upload a file above for analysis..."):
    # Prepare user message content
    user_content = prompt
    if uploaded_file:
        user_content += f"\n\n📎 *Attached: {uploaded_file.name}*"
        if file_description:
            user_content += f"\n*Description: {file_description}*"
    
    # Add user message
    user_message = {"role": "user", "content": user_content, "timestamp": datetime.now().isoformat()}
    st.session_state.messages.append(user_message)
    
    with st.chat_message("user"):
        st.markdown(user_content)
    
    # Generate response
    with st.chat_message("assistant"):
        try:
            # Show appropriate spinner based on whether file is uploaded
            spinner_text = "🧠 Analyzing your document and question..." if uploaded_file else "🧠 Analyzing your question and consulting IB materials..."
            
            with st.spinner(spinner_text):
                result = asyncio.run(handle_query(prompt, uploaded_file, file_description))
                response = result.get("response", "I encountered an issue processing your question.")
                thinking = result.get("thinking", "")
                
                # Ensure response doesn't contain thinking tags
                if "<think>" in response.lower() or "</think>" in response.lower():
                    import re
                    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
                    response = response.strip()
                
                if not response:
                    response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                
                st.markdown(response)
                
                # Show thinking in expandable section if available and enabled
                if thinking and thinking.strip() and st.session_state.show_thinking:
                    with st.expander("🤔 Model Thinking Process", expanded=False):
                        st.markdown(f"```\n{thinking}\n```")
        except Exception as e:
            st.error(f"❌ Sorry, I encountered an error: {str(e)}")
            response = f"Error processing your question: {str(e)}"
            thinking = f"Error details: {str(e)}"
    
    # Add assistant message
    assistant_message = {
        "role": "assistant", 
        "content": response,
        "timestamp": datetime.now().isoformat()
    }
    if thinking:
        assistant_message["thinking"] = thinking
    
    st.session_state.messages.append(assistant_message)
    
    # Clear uploaded file after processing
    if uploaded_file:
        st.session_state.uploaded_file = None
        st.session_state.file_processed = False
        st.rerun()

# Simplified sidebar
with st.sidebar:
    st.header("🚀 Getting Started")
    
    st.markdown("""
    **How to use the IB Student Assistant:**
    
    💬 **Ask Questions:** Type any IB-related question in the chat
    📎 **Upload Files:** Use the file uploader above the chat for:
    - **IA Assessment:** Upload your IA and ask "Assess my IA" 
    - **Document Analysis:** Upload any IB document and ask questions
    - **Study Help:** Upload notes or papers for explanation
    
    💡 **Example prompts with files:**
    - "Assess my Mathematics AA HL IA"
    - "What are the main points in this document?"
    - "How can I improve this work?"
    - "Explain this concept from my uploaded notes"
    """)
    
    st.header("📎 File Upload Tips")
    
    st.markdown("""
    **Supported formats:** PDF, DOCX, TXT
    
    **For IA Assessment:**
    - Upload your complete IA PDF
    - Ask "Assess my IA" or "Give me feedback on my IA"
    - Get detailed criterion-based scoring
    
    **For General Analysis:**
    - Upload any IB-related document
    - Ask specific questions about the content
    - Get AI-powered insights and explanations
    """)
    
    st.header("⚙️ Options")
    
    # Toggle for showing thinking
    st.session_state.show_thinking = st.checkbox(
        "Show model thinking process", 
        value=st.session_state.show_thinking,
        help="Toggle to see how the AI reasons through your questions"
    )
    
    # Clear current conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
