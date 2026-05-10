"""Vector search module for RAG analysis."""

import logging
from typing import List, Dict, Any, Optional, Tuple

from ..config import Config
from ..database import db
from .embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


class VectorSearch:
    """Handles vector similarity search for document retrieval."""
    
    def __init__(self, embedding_generator: Optional[EmbeddingGenerator] = None):
        """
        Initialize vector search.
        
        Args:
            embedding_generator: Instance for generating query embeddings
        """
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.similarity_threshold = Config.SIMILARITY_THRESHOLD
        self.search_limit = Config.SEARCH_LIMIT
    
    def search_documents(self, query: str, limit: Optional[int] = None, 
                        threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents based on query.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            threshold: Similarity threshold
            
        Returns:
            List of similar documents with similarity scores
        """
        limit = limit or self.search_limit
        threshold = threshold or self.similarity_threshold
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_single_embedding(query)
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            # Search in database
            results = db.search_similar_embeddings(query_embedding, limit, threshold)
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'id': result['id'],
                    'content': result['content'],
                    'metadata': result.get('metadata', {}),
                    'similarity': result['similarity']
                })
            
            logger.info(f"Found {len(formatted_results)} similar documents for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in document search: {e}")
            return []
    
    def search_chunks(self, query: str, limit: Optional[int] = None,
                      threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search for similar document chunks based on query.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            threshold: Similarity threshold
            
        Returns:
            List of similar chunks with similarity scores
        """
        limit = limit or self.search_limit
        threshold = threshold or self.similarity_threshold
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_single_embedding(query)
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            # Search in database
            results = db.search_similar_chunks(query_embedding, limit, threshold)
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'id': result['id'],
                    'chunk_text': result['chunk_text'],
                    'chunk_index': result['chunk_index'],
                    'document_content': result.get('document_content', ''),
                    'similarity': result['similarity']
                })
            
            logger.info(f"Found {len(formatted_results)} similar chunks for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in chunk search: {e}")
            return []
    
    def hybrid_search(self, query: str, document_weight: float = 0.7,
                     chunk_weight: float = 0.3, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining document and chunk results.
        
        Args:
            query: Search query text
            document_weight: Weight for document results
            chunk_weight: Weight for chunk results
            limit: Maximum number of results
            
        Returns:
            List of hybrid search results
        """
        limit = limit or self.search_limit
        
        try:
            # Get document results
            doc_results = self.search_documents(query, limit=int(limit * document_weight))
            
            # Get chunk results
            chunk_results = self.search_chunks(query, limit=int(limit * chunk_weight))
            
            # Combine and rank results
            combined_results = []
            
            # Add document results with adjusted scores
            for result in doc_results:
                result['type'] = 'document'
                result['adjusted_similarity'] = result['similarity'] * document_weight
                combined_results.append(result)
            
            # Add chunk results with adjusted scores
            for result in chunk_results:
                result['type'] = 'chunk'
                result['adjusted_similarity'] = result['similarity'] * chunk_weight
                combined_results.append(result)
            
            # Sort by adjusted similarity
            combined_results.sort(key=lambda x: x['adjusted_similarity'], reverse=True)
            
            # Return top results
            return combined_results[:limit]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def get_context_for_query(self, query: str, max_context_length: int = 2000) -> str:
        """
        Get relevant context for a query.
        
        Args:
            query: Search query text
            max_context_length: Maximum length of context to return
            
        Returns:
            Formatted context string
        """
        try:
            # Perform hybrid search
            results = self.hybrid_search(query, limit=5)
            
            if not results:
                return "No relevant documents found."
            
            # Build context
            context_parts = []
            current_length = 0
            
            for result in results:
                if result['type'] == 'document':
                    content = result['content']
                else:
                    content = result['chunk_text']
                
                # Check if adding this content would exceed limit
                if current_length + len(content) > max_context_length:
                    # Truncate content if needed
                    remaining_space = max_context_length - current_length
                    if remaining_space > 100:  # Only add if meaningful space remains
                        content = content[:remaining_space - 3] + "..."
                    else:
                        break
                
                context_parts.append(f"[{result['type'].title()} - Similarity: {result['similarity']:.3f}]")
                context_parts.append(content)
                context_parts.append("---")
                current_length += len(content) + 50  # Account for formatting
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting context for query: {e}")
            return "Error retrieving context."
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], 
                          query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search documents by metadata criteria.
        
        Args:
            metadata_filter: Dictionary of metadata criteria
            query: Optional query for relevance ranking
            
        Returns:
            List of matching documents
        """
        try:
            # Build SQL query for metadata search
            where_conditions = []
            params = []
            
            for key, value in metadata_filter.items():
                if isinstance(value, str):
                    where_conditions.append(f"metadata->>'{key}' = %s")
                elif isinstance(value, (int, float)):
                    where_conditions.append(f"metadata->>'{key}'::numeric = %s")
                params.append(value)
            
            if not where_conditions:
                return []
            
            query_sql = f"""
            SELECT id, content, metadata, 
                   CASE WHEN embedding IS NOT NULL AND embedding != ''::vector 
                        THEN 1.0 ELSE 0.5 END as relevance_score
            FROM document_embeddings
            WHERE {' AND '.join(where_conditions)}
            ORDER BY relevance_score DESC
            LIMIT 20
            """
            
            results = db.execute_query(query_sql, params)
            
            # If query provided, re-rank by similarity
            if query and results:
                query_embedding = self.embedding_generator.generate_single_embedding(query)
                if query_embedding:
                    # Calculate similarity for each result
                    for result in results:
                        if result['relevance_score'] == 1.0:  # Has embedding
                            # This would require a custom similarity calculation
                            # For now, keep the relevance score
                            pass
            
            return results
            
        except Exception as e:
            logger.error(f"Error in metadata search: {e}")
            return []
    
    def get_search_stats(self) -> Dict[str, Any]:
        """
        Get statistics about search capabilities.
        
        Returns:
            Dictionary with search statistics
        """
        try:
            # Count searchable documents
            doc_query = """
            SELECT COUNT(*) as count
            FROM document_embeddings
            WHERE embedding IS NOT NULL AND embedding != ''::vector
            """
            doc_count = db.execute_query(doc_query)
            
            # Count searchable chunks
            chunk_query = """
            SELECT COUNT(*) as count
            FROM document_chunks
            WHERE embedding IS NOT NULL AND embedding != ''::vector
            """
            chunk_count = db.execute_query(chunk_query)
            
            return {
                'searchable_documents': doc_count[0]['count'] if doc_count else 0,
                'searchable_chunks': chunk_count[0]['count'] if chunk_count else 0,
                'similarity_threshold': self.similarity_threshold,
                'search_limit': self.search_limit,
                'embedding_model': self.embedding_generator.model_name
            }
            
        except Exception as e:
            logger.error(f"Error getting search stats: {e}")
            return {}
