"""
Cownselor - AI Academic Advisor Chatbot
Streamlit UI for Cownselor
"""

import streamlit as st
import logging
from datetime import datetime
from pathlib import Path
import time
from typing import List, Dict, Any
import pandas as pd

from rag_chain import Cownselor
from ingest import DocumentIngestor

import os
import torch

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# configuration
st.set_page_config(
    page_title="🐄 Cownselor - Your AI Academic Advisor",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1f4e79, #4a90e2);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f4e79;
    }
    .user-message {
        background-color: #f0f8ff;
        border-left-color: #4a90e2;
    }
    .assistant-message {
        background-color: #f8f9fa;
        border-left-color: #1f4e79;
    }
    .sidebar-info {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .status-good {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'counselor' not in st.session_state:
        st.session_state.counselor = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'last_status_check' not in st.session_state:
        st.session_state.last_status_check = None

@st.cache_resource
def load_counselor():
    """load RAG system"""
    try:
        with st.spinner("🧠 Initializing Cownselor AI system..."):
            counselor = Cownselor()
            return counselor, True
    except Exception as e:
        logger.error(f"Failed to initialize Cownselor: {str(e)}")
        return None, False

def display_header():
    st.markdown("""
    <div class="main-header">
        <h1>🐄 Cownselor</h1>
        <h3>Your AI Academic Advisor</h3>
        <p>Get help with class selection, graduation requirements, schedule planning, and electives!</p>
    </div>
    """, unsafe_allow_html=True)

def display_system_status(counselor):
    st.sidebar.markdown("## 📊 System Status")
    
    if counselor is None:
        st.sidebar.markdown('<div class="status-error">❌ System Not Initialized</div>', unsafe_allow_html=True)
        return
    
    try:
        status = counselor.get_system_status()
        
        if status['vectorstore_available']:
            st.sidebar.markdown('<div class="status-good">✅ Knowledge Base: Ready</div>', unsafe_allow_html=True)
            if 'documents_in_vectorstore' in status:
                st.sidebar.write(f"📚 Documents: {status['documents_in_vectorstore']}")
        else:
            st.sidebar.markdown('<div class="status-warning">⚠️ Knowledge Base: Not Found</div>', unsafe_allow_html=True)
            st.sidebar.warning("Run `python ingest.py` to create knowledge base")
        
        st.sidebar.markdown(f'<div class="status-good">🤖 LLM: {status["llm_model"]}</div>', unsafe_allow_html=True)
        
        st.sidebar.write(f"🔤 Embeddings: {status['embedding_model']}")
        
        st.sidebar.write(f"💬 Chat Messages: {status['chat_history_length']}")
        
    except Exception as e:
        st.sidebar.markdown('<div class="status-error">❌ Status Check Failed</div>', unsafe_allow_html=True)
        st.sidebar.error(f"Error: {str(e)}")

def display_quick_actions():
    st.sidebar.markdown("## 🚀 Quick Actions")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.counselor:
                st.session_state.counselor.clear_history()
            st.rerun()
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_resource.clear()
            st.session_state.counselor = None
            st.session_state.system_initialized = False
            st.rerun()

def display_sample_questions():
    """Display sample questions for users."""
    st.sidebar.markdown("## 💡 Sample Questions")
    
    sample_questions = [
        "How many credits do I need to graduate?",
        "What are the prerequisites for Calculus II?",
        "Can you recommend electives for data science?",
        "What general education requirements do I have left?"
    ]
    
    for question in sample_questions:
        if st.sidebar.button(f"💬 {question}", key=f"sample_{hash(question)}", use_container_width=True):
            process_user_question(question)

def display_document_management():
    st.sidebar.markdown("## 📁 Document Management")
    
    docs_path = Path("docs")
    if docs_path.exists():
        doc_files = [f for f in docs_path.rglob("*") if f.is_file() and not f.name.startswith('.')]
        
        if doc_files:
            st.sidebar.write(f"📚 {len(doc_files)} documents in knowledge base")
            
            with st.sidebar.expander("View Documents"):
                for doc_file in doc_files:
                    file_type = doc_file.suffix.upper()
                    st.write(f"{file_type} {doc_file.name}")
        else:
            st.sidebar.warning("No documents found in docs/ folder")
    else:
        st.sidebar.error("docs/ folder not found")
    
    if st.sidebar.button("🔄 Re-index Documents", use_container_width=True):
        with st.spinner("Re-indexing documents..."):
            try:
                ingestor = DocumentIngestor()
                vectorstore = ingestor.process_and_store()
                if vectorstore:
                    st.sidebar.success("Documents re-indexed successfully!")
                    st.cache_resource.clear()
                    st.session_state.counselor = None
                    st.session_state.system_initialized = False
                    # state clear wait
                    time.sleep(1)
                    st.rerun()
                else:
                    st.sidebar.error("Failed to re-index documents")
            except Exception as e:
                st.sidebar.error(f"Error re-indexing: {str(e)}")

def display_chat_message(message: Dict[str, str], sources: List = None):
    if message["role"] == "user":
        with st.container():
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🎓 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    else:
        with st.container():
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🐄 Cownselor:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # show sources referenced
            if sources and len(sources) > 0:
                with st.expander(f"📚 Sources ({len(sources)} documents)", expanded=False):
                    for i, doc in enumerate(sources, 1):
                        source = doc.metadata.get('filename', 'Unknown')
                        category = doc.metadata.get('category', 'general')
                        content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        
                        st.markdown(f"""
                        **Source {i}: {source}** ({category})
                        
                        {content_preview}
                        """)

def process_user_question(user_input: str):
    if not user_input.strip():
        return
    
    st.session_state.chat_history.append({"role": "user", "content": user_input, "timestamp": datetime.now()})
    
    with st.spinner("🧠 Cownselor is thinking..."):
        try:
            if st.session_state.counselor:
                response, sources = st.session_state.counselor.generate_response(user_input)
                
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": sources,
                    "timestamp": datetime.now()
                })
            else:
                st.error("❌ Cownselor system not initialized. Please refresh the page.")
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            error_msg = "I apologize, but I encountered an error. Please try again or check the system status."
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_msg,
                "timestamp": datetime.now()
            })
    
    st.rerun()

def main():
    initialize_session_state()
    display_header()
    
    # init Cownselor system
    if not st.session_state.system_initialized:
        counselor, success = load_counselor()
        st.session_state.counselor = counselor
        st.session_state.system_initialized = success
        
        if not success:
            st.error("❌ Failed to initialize Cownselor. Please check the system requirements and try again.")
            st.info("🔧 Make sure you have:")
            st.markdown("""
            - Ollama installed and running
            - deepseek-r1:7b model available in Ollama
            - Documents in the `docs/` folder
            - Run `python ingest.py` to create the knowledge base
            """)
            return
    
    # sidebar controls
    with st.sidebar:
        st.markdown("# 🛠️ Controls")
        display_system_status(st.session_state.counselor)
        display_quick_actions()
        display_sample_questions()
        display_document_management()
    
    st.markdown("## 💬 Chat with Cownselor")
    
    # chat container
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="sidebar-info">
                <h4>👋 Welcome to Cownselor!</h4>
                <p>I'm here to help you with:</p>
                <ul>
                    <li>🎯 Class selection and course planning</li>
                    <li>📋 Graduation requirement verification</li>
                    <li>⚖️ Schedule difficulty assessment</li>
                    <li>🌟 Elective recommendations</li>
                </ul>
                <p><strong>Ask me anything about your academic journey!</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # chat messages
            for message in st.session_state.chat_history:
                sources = message.get("sources", [])
                display_chat_message(message, sources)
    
    # input
    with st.container():
        st.markdown("---")
        col1, col2 = st.columns([6, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask Cownselor a question:",
                placeholder="e.g., What courses do I need for my major?",
                key="user_input",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.button("Send 🚀", use_container_width=True)
        
        # process input
        if send_button or (user_input and st.session_state.get('last_input') != user_input):
            if user_input:
                st.session_state.last_input = user_input
                process_user_question(user_input)
    
    st.markdown("---")
    st.markdown("""    <div style="text-align: center; color: #666; font-size: 0.8em;">
        🐄 Cownselor v1.0 | Built with Streamlit, LangChain, FAISS, and Ollama<br>
        Remember to verify important academic information with your official academic advisor!
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
