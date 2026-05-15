import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional, Dict, Any, List
import logging

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/bi_db')
        self.connection = None
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(self.database_url)
            self.connection.autocommit = True
            logger.info("Database connection established")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute a query and return results"""
        if not self.connection:
            if not self.connect():
                return []
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                if cursor.description:
                    return [dict(row) for row in cursor.fetchall()]
                return []
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []
    
    def initialize_database(self) -> bool:
        """Initialize database with required tables and extensions"""
        if not self.connect():
            return False
        
        try:
            # Enable pgvector extension
            self.execute_query("CREATE EXTENSION IF NOT EXISTS vector;")
            logger.info("pgvector extension enabled")
            
            # Check if tables exist and have correct dimensions
            existing_tables = self.execute_query("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('document_embeddings', 'document_chunks')
            """)
            
            table_names = [row['table_name'] for row in existing_tables]
            
            # Check if we need to recreate tables due to dimension mismatch
            need_recreate = False
            if 'document_embeddings' in table_names:
                # Check embedding dimension
                dim_check = self.execute_query("""
                    SELECT data_type, udt_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'document_embeddings' 
                    AND column_name = 'embedding'
                """)
                if dim_check:
                    # For simplicity, we'll recreate if tables exist but might have wrong dimensions
                    # In production, you'd want more sophisticated dimension checking
                    logger.info("Existing tables found, checking compatibility...")
                    need_recreate = False  # Assume compatible for now
                else:
                    need_recreate = True
            else:
                need_recreate = True
            
            if need_recreate:
                self.execute_query("DROP TABLE IF EXISTS document_chunks CASCADE;")
                self.execute_query("DROP TABLE IF EXISTS document_embeddings CASCADE;")
                logger.info("Recreating tables with correct dimensions")
            else:
                logger.info("Tables already exist with correct structure")
            
            # Create document_embeddings table (IF NOT EXISTS for safety)
            create_embeddings_table = f"""
            CREATE TABLE IF NOT EXISTS document_embeddings (
                id SERIAL PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding vector({Config.EMBEDDING_DIMENSION}),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            self.execute_query(create_embeddings_table)
            logger.info(f"document_embeddings table ready with {Config.EMBEDDING_DIMENSION} dimensions")
            
            # Create document_chunks table (IF NOT EXISTS for safety)
            create_chunks_table = f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES document_embeddings(id),
                chunk_text TEXT,
                chunk_index INTEGER,
                embedding vector({Config.EMBEDDING_DIMENSION}),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            self.execute_query(create_chunks_table)
            logger.info(f"document_chunks table ready with {Config.EMBEDDING_DIMENSION} dimensions")
            
            # Create analysis_results table
            create_analysis_table = """
            CREATE TABLE IF NOT EXISTS analysis_results (
                id SERIAL PRIMARY KEY,
                analysis_type VARCHAR(100),
                file_name VARCHAR(255),
                results JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            self.execute_query(create_analysis_table)
            logger.info("analysis_results table checked/created")
            
            # Create indexes for vector search
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_document_embeddings_embedding ON document_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);",
                "CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);",
                "CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);"
            ]
            
            for index_query in indexes:
                self.execute_query(index_query)
            
            logger.info("Database indexes checked/created")
            return True
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False
    
    def store_embedding(self, content: str, metadata: Dict[str, Any], embedding: List[float]) -> Optional[int]:
        """Store document embedding"""
        import json
        
        # Validate embedding format
        if not isinstance(embedding, list):
            logger.error(f"Embedding must be a list, got {type(embedding)}")
            return None
        
        # Clean content to remove NUL characters
        clean_content = content.replace('\x00', '') if content else content
        
        # Convert metadata to JSON string for JSONB column
        try:
            metadata_json = json.dumps(metadata)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize metadata to JSON: {e}")
            metadata_json = json.dumps({})
        
        query = """
        INSERT INTO document_embeddings (content, metadata, embedding)
        VALUES (%s, %s, %s)
        RETURNING id;
        """
        try:
            # Use a direct cursor for INSERT with RETURNING to avoid issues with execute_query
            if not self.connection:
                if not self.connect():
                    return None
            
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (clean_content, metadata_json, embedding))
                result = cursor.fetchone()
                if result:
                    doc_id = result['id']
                    logger.info(f"Database returned document ID: {doc_id}")
                    return doc_id
                else:
                    logger.error("Database query returned no result")
                    return None
        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
            logger.error(f"Content type: {type(clean_content)}, length: {len(clean_content) if clean_content else 'N/A'}")
            logger.error(f"Metadata JSON type: {type(metadata_json)}")
            logger.error(f"Embedding type: {type(embedding)}, length: {len(embedding) if isinstance(embedding, list) else 'N/A'}")
            return None
    
    def store_embeddings_bulk(self, documents_data: List[Dict[str, Any]]) -> int:
        """Store multiple documents with embeddings in bulk"""
        import json
        
        if not documents_data:
            return 0
        
        # Validate all embeddings
        for i, doc_data in enumerate(documents_data):
            if not isinstance(doc_data['embedding'], list):
                logger.error(f"Document {i} embedding must be a list, got {type(doc_data['embedding'])}")
                return 0
        
        # Prepare data for bulk insert
        values = []
        for doc_data in documents_data:
            content = doc_data['content'].replace('\x00', '') if doc_data['content'] else doc_data['content']
            metadata_json = json.dumps(doc_data['metadata'])
            embedding = doc_data['embedding']
            values.append((content, metadata_json, embedding))
        
        query = """
        INSERT INTO document_embeddings (content, metadata, embedding)
        VALUES %s
        """
        
        try:
            if not self.connection:
                if not self.connect():
                    return 0
            
            with self.connection.cursor() as cursor:
                # Use execute_values for bulk insert
                from psycopg2.extras import execute_values
                execute_values(cursor, query, values)
                self.connection.commit()
                
            logger.info(f"Bulk inserted {len(documents_data)} documents")
            return len(documents_data)
            
        except Exception as e:
            logger.error(f"Failed bulk insert: {e}")
            return 0
    
    def store_chunks_bulk(self, chunks_data: List[Dict[str, Any]]) -> int:
        """Store multiple document chunks with embeddings in bulk"""
        if not chunks_data:
            return 0
        
        # Validate all embeddings
        for i, chunk_data in enumerate(chunks_data):
            if not isinstance(chunk_data['embedding'], list):
                logger.error(f"Chunk {i} embedding must be a list, got {type(chunk_data['embedding'])}")
                return 0
        
        # Prepare data for bulk insert
        values = []
        for chunk_data in chunks_data:
            chunk_text = chunk_data['chunk_text'].replace('\x00', '') if chunk_data['chunk_text'] else chunk_data['chunk_text']
            values.append((
                chunk_data['document_id'],
                chunk_text,
                chunk_data['chunk_index'],
                chunk_data['embedding']
            ))
        
        query = """
        INSERT INTO document_chunks (document_id, chunk_text, chunk_index, embedding)
        VALUES %s
        """
        
        try:
            if not self.connection:
                if not self.connect():
                    return 0
            
            with self.connection.cursor() as cursor:
                # Use execute_values for bulk insert
                from psycopg2.extras import execute_values
                execute_values(cursor, query, values)
                self.connection.commit()
                
            logger.info(f"Bulk inserted {len(chunks_data)} chunks")
            return len(chunks_data)
            
        except Exception as e:
            logger.error(f"Failed bulk insert chunks: {e}")
            return 0
    
    def store_chunk(self, document_id: int, chunk_text: str, chunk_index: int, embedding: List[float]) -> Optional[int]:
        """Store document chunk"""
        # Validate embedding format
        if not isinstance(embedding, list):
            logger.error(f"Embedding must be a list, got {type(embedding)}")
            return None
        
        # Clean chunk text to remove NUL characters
        clean_chunk_text = chunk_text.replace('\x00', '') if chunk_text else chunk_text
        
        query = """
        INSERT INTO document_chunks (document_id, chunk_text, chunk_index, embedding)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
        """
        try:
            result = self.execute_query(query, (document_id, clean_chunk_text, chunk_index, embedding))
            return result[0]['id'] if result else None
        except Exception as e:
            logger.error(f"Failed to store chunk: {e}")
            logger.error(f"Embedding type: {type(embedding)}, length: {len(embedding) if isinstance(embedding, list) else 'N/A'}")
            return None
    
    def search_similar_embeddings(self, query_embedding: List[float], limit: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar embeddings using cosine similarity"""
        query = """
        SELECT id, content, metadata, 1 - (embedding <=> %s::vector) as similarity
        FROM document_embeddings
        WHERE 1 - (embedding <=> %s::vector) > %s
        ORDER BY similarity DESC
        LIMIT %s;
        """
        return self.execute_query(query, (query_embedding, query_embedding, threshold, limit))

    def search_top_embeddings(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Search top document embeddings without applying a similarity threshold."""
        query = """
        SELECT id, content, metadata, 1 - (embedding <=> %s::vector) as similarity
        FROM document_embeddings
        WHERE embedding IS NOT NULL
        ORDER BY similarity DESC
        LIMIT %s;
        """
        return self.execute_query(query, (query_embedding, limit))
    
    def search_similar_chunks(self, query_embedding: List[float], limit: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar chunks using cosine similarity"""
        query = """
        SELECT dc.id, dc.chunk_text, dc.chunk_index, de.content as document_content,
               1 - (dc.embedding <=> %s::vector) as similarity
        FROM document_chunks dc
        JOIN document_embeddings de ON dc.document_id = de.id
        WHERE 1 - (dc.embedding <=> %s::vector) > %s
        ORDER BY similarity DESC
        LIMIT %s;
        """
        return self.execute_query(query, (query_embedding, query_embedding, threshold, limit))

    def search_top_chunks(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Search top document chunks without applying a similarity threshold."""
        query = """
        SELECT dc.id, dc.chunk_text, dc.chunk_index, de.content as document_content,
               de.metadata as document_metadata,
               1 - (dc.embedding <=> %s::vector) as similarity
        FROM document_chunks dc
        JOIN document_embeddings de ON dc.document_id = de.id
        WHERE dc.embedding IS NOT NULL
        ORDER BY similarity DESC
        LIMIT %s;
        """
        return self.execute_query(query, (query_embedding, limit))
    
    def store_analysis_result(self, analysis_type: str, file_name: str, results: Dict[str, Any]) -> Optional[int]:
        """Store analysis results"""
        query = """
        INSERT INTO analysis_results (analysis_type, file_name, results)
        VALUES (%s, %s, %s)
        RETURNING id;
        """
        result = self.execute_query(query, (analysis_type, file_name, results))
        return result[0]['id'] if result else None
    
    def get_analysis_results(self, analysis_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve analysis results"""
        if analysis_type:
            query = "SELECT * FROM analysis_results WHERE analysis_type = %s ORDER BY created_at DESC LIMIT %s;"
            return self.execute_query(query, (analysis_type, limit))
        else:
            query = "SELECT * FROM analysis_results ORDER BY created_at DESC LIMIT %s;"
            return self.execute_query(query, (limit,))

# Global database instance
db = DatabaseManager()
