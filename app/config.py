"""Application configuration module."""

import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Application configuration settings."""
    
    # Database settings
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/bi_db')
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    OUTPUT_DIR = BASE_DIR / 'output'
    
    # AI/ML settings
    EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
    EMBEDDING_DIMENSION = 768
    
    # LLM settings
    GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
    GROQ_MODEL = 'llama3-8b-8192'  # Default Groq model
    
    # Analysis settings
    CSV_CHUNK_SIZE = 10000
    PDF_CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Vector search settings
    SIMILARITY_THRESHOLD = 0.5
    SEARCH_LIMIT = 5
    
    # Web interface settings
    WEB_HOST = '0.0.0.0'
    WEB_PORT = 7860
    
    # Data loading settings
    DATA_LOAD = os.getenv('DATA_LOAD', 'false').lower() == 'true'
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def get_csv_schema(cls) -> Dict[str, Any]:
        """Default schema for CSV analysis."""
        return {
            'date_column': 'Date',
            'measures': ['Sales'],
            'dimensions': ['Product', 'Region', 'Customer_Gender'],
            'customer_attributes': ['Customer_Age', 'Customer_Gender'],
            'satisfaction_column': 'Customer_Satisfaction'
        }
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
