"""Document processing module for RAG analysis."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from ..config import Config
from ..database import db

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
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.PDF_CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
    
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
        Load, split, and store documents in database.
        
        Args:
            glob_pattern: Pattern to match files
            
        Returns:
            Number of documents processed
        """
        # Load documents
        documents = self.load_documents(glob_pattern)
        if not documents:
            return 0
        
        # Split documents
        chunks = self.split_documents(documents)
        if not chunks:
            return 0
        
        # Store in database
        stored_count = 0
        for doc in documents:
            # Store document metadata
            metadata = {
                'source': doc.metadata.get('source', ''),
                'page': doc.metadata.get('page', 0),
                'total_pages': doc.metadata.get('total_pages', 0)
            }
            
            # Store document (embedding will be added later)
            doc_id = db.store_embedding(
                content=doc.page_content,
                metadata=metadata,
                embedding=[]  # Placeholder, will be updated with actual embedding
            )
            
            if doc_id:
                stored_count += 1
        
        logger.info(f"Processed and stored {stored_count} documents")
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
