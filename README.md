# 🐄 Cownselor - AI Academic Advisor

A complete local Retrieval-Augmented Generation (RAG) chatbot designed to help university students with academic planning and course selection.

## Technologies

- **LLM**: Ollama with deepseek-r1:7b (completely local)
- **Embeddings**: HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- **Vector Database**: FAISS (but can be easily replaced with Chroma, Qdrant, Pinecone, etc)
- **Framework**: LangChain for RAG
- **UI**: Streamlit
- **Document Processing**: Supports PDF, DOCX, CSV, TXT, MD files

## Quick Start

### Prerequisites

`python, ollama, conda/venv`

### Installation

1. **Setup**:
   ```bash
   # Create conda environment with Python 3.9
   conda create -n cownselor python=3.9 -y
   conda activate cownselor
   
   # Install other dependencies
   pip install -r requirements.txt
   ```

2. **Add your academic documents** to the `docs/` folder:
   - Course catalogs (PDF)
   - Degree requirements (CSV/PDF)
   - Academic policies (TXT/DOCX)
   - Any other relevant academic documents

3. **Process the documents**:
   ```bash
   python ingest.py
   ```

4. **Start app**:
   ```bash
   streamlit run app.py
   ```
