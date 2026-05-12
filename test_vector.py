#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import db

print("Testing pgvector extension...")
try:
    if db.connect():
        print("Connected to database successfully")
        
        # Test pgvector extension
        result = db.execute_query("SELECT version();")
        print(f"PostgreSQL version: {result}")
        
        # Check if pgvector extension exists
        ext_result = db.execute_query("SELECT * FROM pg_extension WHERE extname = 'vector';")
        print(f"pgvector extension: {ext_result}")
        
        # Check table schemas
        tables = db.execute_query("""
            SELECT table_name, column_name, data_type, udt_name 
            FROM information_schema.columns 
            WHERE table_name IN ('document_embeddings', 'document_chunks') 
            AND column_name = 'embedding'
        """)
        print(f"Table schemas: {tables}")
        
    else:
        print("Failed to connect to database")
except Exception as e:
    print(f"Error: {e}")
finally:
    db.disconnect()
