"""
Document Ingestion Pipeline for Cownselor RAG Chatbot

This module handles loading, processing, and embedding academic documents
into ChromaDB for retrieval-augmented generation.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
# may need to add other loader types, depending on added documents
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentIngestor:
    """Handles document loading, chunking, and vector storage."""
    
    def __init__(self, docs_path: str = "docs", vectorstore_path: str = "vectorstore"):
        self.docs_path = Path(docs_path)
        self.vectorstore_path = vectorstore_path
        
        # using hugging face all-MiniLM-L6-v2 model for embeddings
        logger.info("Loading embedding model: all-MiniLM-L6-v2")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def load_documents(self) -> List[Document]:
        """Load all supported documents from the docs folder."""
        documents = []
        
        if not self.docs_path.exists():
            logger.warning(f"Documents path {self.docs_path} does not exist")
            return documents
            
        for file_path in self.docs_path.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                try:
                    docs = self._load_single_file(file_path)
                    documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} chunks from {file_path.name}")
                except Exception as e:
                    logger.error(f"Error loading {file_path.name}: {str(e)}")
                    
        return documents
    
    def _load_single_file(self, file_path: Path) -> List[Document]:
        """Load a single file"""
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_extension == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif file_extension == '.csv':
                loader = CSVLoader(str(file_path))
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(str(file_path))
            elif file_extension == '.md':
                loader = TextLoader(str(file_path), encoding='utf-8')
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return []
                
            documents = loader.load()
            
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'filename': file_path.name,
                    'file_type': file_extension,
                    'category': self._categorize_document(file_path.name)
                })
                
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {str(e)}")
            return []
    
    def _categorize_document(self, filename: str) -> str:
        """Categorize documents based on filename for better retrieval."""
        filename_lower = filename.lower()
        
        if any(keyword in filename_lower for keyword in ['course', 'catalog', 'schedule']):
            return 'courses'
        elif any(keyword in filename_lower for keyword in ['requirement', 'degree', 'major']):
            return 'requirements'
        elif any(keyword in filename_lower for keyword in ['elective', 'recommendation']):
            return 'electives'
        elif any(keyword in filename_lower for keyword in ['policy', 'rule', 'guideline']):
            return 'policies'
        else:
            return 'general'
    
    def process_and_store(self) -> FAISS:
        """Load, chunk, and store documents in ChromaDB."""
        logger.info("Starting document ingestion process...")
        
        documents = self.load_documents()
        if not documents:
            logger.warning("No documents found to process")
            return None
            
        logger.info(f"Loaded {len(documents)} documents")
        
        # spltting docs
        logger.info("Splitting documents into chunks...")
        texts = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(texts)} text chunks")
        
        # updating vector store
        logger.info("Creating embeddings and storing in FAISS...")
        vectorstore = FAISS.from_documents(
            documents=texts,
            embedding=self.embeddings
        )
        
        vectorstore.save_local(self.vectorstore_path)
        logger.info(f"Vector store created and persisted to {self.vectorstore_path}")
        
        return vectorstore
    
    def update_vectorstore(self, new_documents: List[Document]) -> None:
        """Add new documents to existing vector store."""
        if not new_documents:
            return
            
        logger.info(f"Adding {len(new_documents)} new documents to vector store...")
        
        try:
            vectorstore = FAISS.load_local(
                self.vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except:
            logger.info("No existing vector store found, creating new one...")
            self.process_and_store()
            return
        
        # spltting and adding new docs

        texts = self.text_splitter.split_documents(new_documents)
        
        vectorstore.add_documents(texts)
        vectorstore.save_local(self.vectorstore_path)
        
        logger.info(f"Added {len(texts)} new chunks to vector store")

def main():
    """Main function to run document ingestion."""
    print("🐄 Cownselor Document Ingestion Pipeline")
    print("=" * 50)
    
    docs_path = Path("docs")
    if not docs_path.exists():
        print("'docs' folder not found!")
        return
    
    files = list(docs_path.rglob("*"))
    doc_files = [f for f in files if f.is_file() and not f.name.startswith('.') and f.suffix.lower() in ['.pdf', '.txt', '.csv', '.docx', '.doc', '.md']]
    
    if not doc_files:
        print("Please add some academic documents and run this script again.")
        return
    
    print(f"Found {len(doc_files)} documents to process:")
    for doc_file in doc_files:
        print(f"  • {doc_file.name}")
    
    ingestor = DocumentIngestor()
    
    try:
        vectorstore = ingestor.process_and_store()
        if vectorstore:
            print("\nDocument ingestion completed successfully!")
            print(f"Vector store saved to: {ingestor.vectorstore_path}")
            print("\nTo run: streamlit run app.py")
        else:
            print("\nDocument ingestion failed!")
    except Exception as e:
        print(f"\nError during ingestion: {str(e)}")
        logger.exception("Ingestion failed")

if __name__ == "__main__":
    main()
