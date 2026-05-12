"""Embedding generation module for RAG analysis."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from app.config import Config
from app.database import db

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Handles embedding generation for documents and queries."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.embedding_dimension = Config.EMBEDDING_DIMENSION
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        try:
            # Use HuggingFace embeddings with the specified configuration
            model_name = "sentence-transformers/all-mpnet-base-v2"
            model_kwargs = {"device": "cpu"}
            encode_kwargs = {"normalize_embeddings": False}
            
            self.model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
            
            # Update dimension for mpnet-base-v2
            self.embedding_dimension = 768
            logger.info(f"Initialized HuggingFace embedding model: {model_name}")
            logger.info(f"Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Error initializing HuggingFace embedding model: {e}")
            # Fallback to sentence-transformers directly
            try:
                self.model = SentenceTransformer(self.model_name)
                self.embedding_dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"Initialized fallback sentence-transformers model: {self.model_name}")
            except Exception as fallback_error:
                logger.error(f"Fallback model initialization failed: {fallback_error}")
                raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            if isinstance(self.model, SentenceTransformer):
                embeddings = self.model.encode(texts, convert_to_tensor=False)
                result = embeddings.tolist()
                logger.debug(f"SentenceTransformer embeddings type: {type(result)}, first element type: {type(result[0]) if result else 'N/A'}")
                return result
            else:
                # Langchain implementation
                embeddings = self.model.embed_documents(texts)
                logger.debug(f"Langchain embeddings type: {type(embeddings)}, first element type: {type(embeddings[0]) if embeddings else 'N/A'}")
                # Ensure we return a list of lists, not list of dicts
                if embeddings and isinstance(embeddings[0], dict):
                    logger.error("Received dict embeddings from Langchain, this should not happen")
                    return []
                return embeddings
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        try:
            if isinstance(self.model, SentenceTransformer):
                embedding = self.model.encode(text, convert_to_tensor=False)
                return embedding.tolist()
            else:
                # Langchain implementation
                embedding = self.model.embed_query(text)
                return embedding
                
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            return []
    
    def embed_and_store_documents(self, batch_size: int = 100) -> int:
        """
        Generate embeddings for all documents in database and store them.
        
        Args:
            batch_size: Number of documents to process at once
            
        Returns:
            Number of documents processed
        """
        # Get documents without embeddings
        query = """
        SELECT id, content FROM document_embeddings 
        WHERE embedding IS NULL OR embedding = '{}'::vector
        ORDER BY id
        """
        documents = db.execute_query(query)
        
        if not documents:
            logger.info("No documents need embedding")
            return 0
        
        processed_count = 0
        total_docs = len(documents)
        
        logger.info(f"Processing {total_docs} documents for embedding generation")
        
        # Process in batches
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc['content'] for doc in batch]
            doc_ids = [doc['id'] for doc in batch]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            if len(embeddings) != len(texts):
                logger.error(f"Embedding generation mismatch: {len(embeddings)} vs {len(texts)}")
                continue
            
            # Update database
            for doc_id, embedding in zip(doc_ids, embeddings):
                update_query = """
                UPDATE document_embeddings 
                SET embedding = %s::vector
                WHERE id = %s
                """
                db.execute_query(update_query, (embedding, doc_id))
            
            processed_count += len(batch)
            logger.info(f"Processed {processed_count}/{total_docs} documents")
        
        logger.info(f"Completed embedding generation for {processed_count} documents")
        return processed_count
    
    def embed_and_store_chunks(self, batch_size: int = 100) -> int:
        """
        Generate embeddings for document chunks and store them.
        
        Args:
            batch_size: Number of chunks to process at once
            
        Returns:
            Number of chunks processed
        """
        # Get chunks without embeddings
        query = """
        SELECT id, chunk_text FROM document_chunks 
        WHERE embedding IS NULL OR embedding = '{}'::vector
        ORDER BY id
        """
        chunks = db.execute_query(query)
        
        if not chunks:
            logger.info("No chunks need embedding")
            return 0
        
        processed_count = 0
        total_chunks = len(chunks)
        
        logger.info(f"Processing {total_chunks} chunks for embedding generation")
        
        # Process in batches
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk['chunk_text'] for chunk in batch]
            chunk_ids = [chunk['id'] for chunk in batch]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            if len(embeddings) != len(texts):
                logger.error(f"Embedding generation mismatch: {len(embeddings)} vs {len(texts)}")
                continue
            
            # Update database
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                update_query = """
                UPDATE document_chunks 
                SET embedding = %s::vector
                WHERE id = %s
                """
                db.execute_query(update_query, (embedding, chunk_id))
            
            processed_count += len(batch)
            logger.info(f"Processed {processed_count}/{total_chunks} chunks")
        
        logger.info(f"Completed embedding generation for {processed_count} chunks")
        return processed_count
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about embeddings in database.
        
        Returns:
            Dictionary with embedding statistics
        """
        # Count documents with embeddings
        doc_query = """
        SELECT 
            COUNT(*) as total_docs,
            COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as embedded_docs
        FROM document_embeddings
        """
        doc_stats = db.execute_query(doc_query)
        
        # Count chunks with embeddings
        chunk_query = """
        SELECT 
            COUNT(*) as total_chunks,
            COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as embedded_chunks
        FROM document_chunks
        """
        chunk_stats = db.execute_query(chunk_query)
        
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'documents': {
                'total': doc_stats[0]['total_docs'] if doc_stats else 0,
                'embedded': doc_stats[0]['embedded_docs'] if doc_stats else 0
            },
            'chunks': {
                'total': chunk_stats[0]['total_chunks'] if chunk_stats else 0,
                'embedded': chunk_stats[0]['embedded_chunks'] if chunk_stats else 0
            }
        }
