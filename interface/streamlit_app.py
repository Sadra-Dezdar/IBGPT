"""Streamlit interface for IB Student Assistant."""

import streamlit as st
import asyncio
from datetime import datetime

# Use the new system without tools
from agents.multi_agent_system_no_tools import MultiAgentSystem, MultiAgentDeps
from utils.chromadb_utils import get_chroma_client

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

async def handle_query(query: str):
    """Handle user query asynchronously."""
    system = init_system()
    deps = init_deps()
    
    result = await system.process_query(query, deps)
    return result

# Streamlit UI
st.set_page_config(
    page_title="IB Student Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 IB Student Assistant")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_thinking" not in st.session_state:
    st.session_state.show_thinking = False

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show thinking in expandable section if available
        if message["role"] == "assistant" and message.get("thinking"):
            with st.expander("🤔 Model Thinking Process", expanded=False):
                st.markdown(f"```\n{message['thinking']}\n```")

# Chat input
if prompt := st.chat_input("Ask your IB question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = asyncio.run(handle_query(prompt))
            response = result["response"]
            thinking = result.get("thinking", "")
            
            st.markdown(response)
            
            # Show thinking in expandable section if available
            if thinking:
                with st.expander("🤔 Model Thinking Process", expanded=False):
                    st.markdown(f"```\n{thinking}\n```")
    
    # Add assistant message
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "thinking": thinking
    })

# Sidebar
with st.sidebar:
    st.header("Options")
    
    # Toggle for showing thinking
    st.session_state.show_thinking = st.checkbox(
        "Show model thinking process", 
        value=st.session_state.show_thinking
    )
    
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()
    
    st.header("Quick Actions")
    
    if st.button("Upload Documents"):
        st.info("Use scripts/ingest_documents.py to add documents")
    
    st.header("About")
    st.markdown("""
    This assistant helps IB students with:
    - General programme questions
    - IA feedback and guidance
    - Exam practice and solutions
    
    Powered by Ollama models:
    - Fast routing: qwen3
    - Deep reasoning: deepseek-r1
    """)