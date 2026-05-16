"""Embedding generation module for RAG analysis."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pickle
import os

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

import config
from app.config import Config
from app.database import db

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Handles embedding generation for documents and queries."""
    
    # Class-level cache for model instance
    _model_cache = {}
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding generator with lazy loading and caching.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.embedding_dimension = Config.EMBEDDING_DIMENSION
        self.model = None
        self._model_initialized = False
        # Don't initialize model immediately - will be loaded on first use
    
    def _ensure_model_initialized(self):
        """Ensure model is initialized before use."""
        if not self._model_initialized:
            logger.info("Initializing embedding model (lazy loading)...")
            self._initialize_model()
            self._model_initialized = True
            logger.info("Embedding model initialized successfully")
    
    def _get_pickle_path(self, model_name: str) -> str:
        """Get the pickle file path for a given model name."""
        # Create a safe filename from model name
        safe_name = model_name.replace("/", "_").replace("-", "_")
        return os.path.join(Config.MODELS_DIR, f"{safe_name}.pkl")
    
    def _save_model_to_pickle(self, model, model_name: str):
        """Save model to pickle file for faster loading."""
        pickle_path = self._get_pickle_path(model_name)
        try:
            with open(pickle_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved model to pickle: {pickle_path}")
        except Exception as e:
            logger.warning(f"Failed to save model to pickle: {e}")
    
    def _load_model_from_pickle(self, model_name: str):
        """Load model from pickle file if it exists."""
        pickle_path = self._get_pickle_path(model_name)
        if os.path.exists(pickle_path):
            try:
                with open(pickle_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Loaded model from pickle: {pickle_path}")
                return model
            except Exception as e:
                logger.warning(f"Failed to load model from pickle: {e}")
                return None
        return None
    
    def _initialize_model(self):
        """Initialize the embedding model with caching."""
        # Check if model is already cached
        cache_key = self.model_name
        if cache_key in self._model_cache:
            logger.info(f"Using cached embedding model: {cache_key}")
            self.model = self._model_cache[cache_key]
            self.embedding_dimension = 768
            return
        
        # Try to load from pickle first
        model_name = "sentence-transformers/all-mpnet-base-v2"
        pickled_model = self._load_model_from_pickle(model_name)
        if pickled_model is not None:
            self.model = pickled_model
            self.embedding_dimension = 768
            self._model_cache[cache_key] = self.model
            logger.info(f"Using pickled model: {model_name}")
            return
        
        try:
            # Use HuggingFace embeddings with the specified configuration
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
            
            # Cache the model for future use
            self._model_cache[cache_key] = self.model
            logger.info(f"Cached embedding model: {cache_key}")
            
            # Save to pickle for faster loading next time
            self._save_model_to_pickle(self.model, model_name)
            
        except Exception as e:
            logger.error(f"Error initializing HuggingFace embedding model: {e}")
            # Fallback to sentence-transformers directly
            try:
                self.model = SentenceTransformer(self.model_name)
                self.embedding_dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"Initialized fallback sentence-transformers model: {self.model_name}")
                # Cache the fallback model as well
                self._model_cache[cache_key] = self.model
                # Save fallback model to pickle as well
                self._save_model_to_pickle(self.model, self.model_name)
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
        
        # Ensure model is initialized before use
        self._ensure_model_initialized()
        
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
        # Ensure model is initialized before use
        self._ensure_model_initialized()
        
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
