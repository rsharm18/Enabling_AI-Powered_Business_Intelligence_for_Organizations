import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional, Dict, Any, List
import logging

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
            
            # Create document_embeddings table
            create_embeddings_table = """
            CREATE TABLE IF NOT EXISTS document_embeddings (
                id SERIAL PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding vector(1536),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            self.execute_query(create_embeddings_table)
            logger.info("document_embeddings table created")
            
            # Create document_chunks table
            create_chunks_table = """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES document_embeddings(id),
                chunk_text TEXT,
                chunk_index INTEGER,
                embedding vector(1536),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            self.execute_query(create_chunks_table)
            logger.info("document_chunks table created")
            
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
            logger.info("analysis_results table created")
            
            # Create indexes for vector search
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_document_embeddings_embedding ON document_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);",
                "CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);",
                "CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);"
            ]
            
            for index_query in indexes:
                self.execute_query(index_query)
            
            logger.info("Database indexes created")
            return True
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False
    
    def store_embedding(self, content: str, metadata: Dict[str, Any], embedding: List[float]) -> Optional[int]:
        """Store document embedding"""
        query = """
        INSERT INTO document_embeddings (content, metadata, embedding)
        VALUES (%s, %s, %s)
        RETURNING id;
        """
        result = self.execute_query(query, (content, metadata, embedding))
        return result[0]['id'] if result else None
    
    def store_chunk(self, document_id: int, chunk_text: str, chunk_index: int, embedding: List[float]) -> Optional[int]:
        """Store document chunk"""
        query = """
        INSERT INTO document_chunks (document_id, chunk_text, chunk_index, embedding)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
        """
        result = self.execute_query(query, (document_id, chunk_text, chunk_index, embedding))
        return result[0]['id'] if result else None
    
    def search_similar_embeddings(self, query_embedding: List[float], limit: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar embeddings using cosine similarity"""
        query = """
        SELECT id, content, metadata, 1 - (embedding <=> %s) as similarity
        FROM document_embeddings
        WHERE 1 - (embedding <=> %s) > %s
        ORDER BY similarity DESC
        LIMIT %s;
        """
        return self.execute_query(query, (query_embedding, query_embedding, threshold, limit))
    
    def search_similar_chunks(self, query_embedding: List[float], limit: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar chunks using cosine similarity"""
        query = """
        SELECT dc.id, dc.chunk_text, dc.chunk_index, de.content as document_content,
               1 - (dc.embedding <=> %s) as similarity
        FROM document_chunks dc
        JOIN document_embeddings de ON dc.document_id = de.id
        WHERE 1 - (dc.embedding <=> %s) > %s
        ORDER BY similarity DESC
        LIMIT %s;
        """
        return self.execute_query(query, (query_embedding, query_embedding, threshold, limit))
    
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
