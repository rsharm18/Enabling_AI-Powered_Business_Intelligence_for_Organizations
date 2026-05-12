"""Document processing module for RAG analysis."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import sys
import os

from app.rag_analysis.embedding_generator import EmbeddingGenerator
from app.config import Config
from app.database import db

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document loading, splitting, and storage."""
    
    def __init__(self, data_directory: Optional[str] = None):
        """
        Initialize document processor.
        
        Args:
            data_directory: Directory containing PDF files
        """
        self.data_directory = Path(data_directory) if data_directory else Config.DATA_DIR / 'PDF Folder'
        logger.info(f" Data directory: {self.data_directory}")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.PDF_CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()
        logger.info(f"Initialized embedding generator with model: {self.embedding_generator.model_name}")
    
    def load_documents(self, glob_pattern: str = "*.pdf") -> List[Document]:
        """
        Load documents from directory.
        
        Args:
            glob_pattern: Pattern to match files
            
        Returns:
            List of loaded documents
        """
        if not self.data_directory.exists():
            logger.warning(f"Directory {self.data_directory} does not exist")
            return []
        
        logger.info(f"Loading documents from {self.data_directory}")
        
        try:
            loader = DirectoryLoader(
                str(self.data_directory),
                glob=glob_pattern,
                loader_cls=PyPDFLoader,
                recursive=True,
                show_progress=True
            )
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        if not documents:
            logger.warning("No documents to split")
            return []
        
        logger.info(f"Splitting {len(documents)} documents into chunks")
        
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            return []
    
    def process_and_store_documents(self, glob_pattern: str = "*.pdf") -> int:
        """
        Load, split, and store documents in database with embeddings.
        
        Args:
            glob_pattern: Pattern to match files
            
        Returns:
            Number of documents processed
        """
        # Check if data loading is enabled
        if not Config.DATA_LOAD:
            logger.info("DATA_LOAD is set to False. Skipping PDF document processing.")
            return 0
        
        # Load documents
        logger.info(f"Loading documents from {self.data_directory} with pattern {glob_pattern}")
        documents = self.load_documents(glob_pattern)
        if not documents:
            logger.warning("No documents found")
            return 0

        logger.info(f"Loaded {len(documents)} documents")
        # Split documents
        chunks = self.split_documents(documents)
        logger.info(f"Split {len(chunks)} documents into chunks")
        if not chunks:
            logger.warning("No chunks created")
            return 0
        
        # Store in database with embeddings using bulk insertion
        
        # Generate embeddings for all documents at once using EmbeddingGenerator
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_generator.generate_embeddings(texts)
        logger.info(f"Generated embeddings for {len(documents)} documents")
        
        # Debug: Check embedding format
        if embeddings:
            logger.info(f"Embedding type: {type(embeddings[0])}")
            logger.info(f"First embedding length: {len(embeddings[0]) if isinstance(embeddings[0], list) else 'N/A'}")
        
        # Prepare data for bulk insertion
        documents_data = []
        for i, doc in enumerate(documents):
            # Store document metadata
            metadata = {
                'source': doc.metadata.get('source', ''),
                'page': doc.metadata.get('page', 0),
                'total_pages': doc.metadata.get('total_pages', 0)
            }
            
            # Clean content to remove NUL characters
            clean_content = doc.page_content.replace('\x00', '')
            
            documents_data.append({
                'content': clean_content,
                'metadata': metadata,
                'embedding': embeddings[i]
            })
        
        # Bulk insert all documents
        stored_count = db.store_embeddings_bulk(documents_data)
        
        # Generate and store chunks with embeddings
        chunk_texts = [chunk.page_content for chunk in chunks]
        chunk_embeddings = self.embedding_generator.generate_embeddings(chunk_texts)
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        
        # Prepare chunks data for bulk insertion
        chunks_data = []
        for i, chunk in enumerate(chunks):
            chunks_data.append({
                'document_id': 1,  # Will be updated after we get the actual document IDs
                'chunk_text': chunk.page_content.replace('\x00', ''),
                'chunk_index': i,
                'embedding': chunk_embeddings[i]
            })
        
        # Bulk insert all chunks
        stored_chunks = db.store_chunks_bulk(chunks_data)
        
        logger.info(f"Processed and stored {stored_count} documents and {stored_chunks} chunks with embeddings")
        return stored_count
    
    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics about processed documents.
        
        Returns:
            Dictionary with document statistics
        """
        if not self.data_directory.exists():
            return {'error': 'Directory does not exist'}
        
        # Count PDF files
        pdf_files = list(self.data_directory.glob("*.pdf"))
        
        # Get database stats
        doc_count = len(db.execute_query("SELECT COUNT(*) as count FROM document_embeddings"))
        
        return {
            'directory': str(self.data_directory),
            'pdf_files_count': len(pdf_files),
            'pdf_files': [f.name for f in pdf_files],
            'stored_documents': doc_count[0]['count'] if doc_count else 0
        }
    
    def clear_database_documents(self) -> bool:
        """
        Clear all documents from database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            db.execute_query("DELETE FROM document_chunks")
            db.execute_query("DELETE FROM document_embeddings")
            logger.info("Cleared all documents from database")
            return True
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False
