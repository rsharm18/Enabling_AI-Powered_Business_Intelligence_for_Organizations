#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import db
from app.config import Config

print("=== Database Status Check ===")
print(f"DATA_LOAD setting: {Config.DATA_LOAD}")

try:
    if db.connect():
        print("✓ Connected to database")
        
        # Check document count
        doc_count = db.execute_query("SELECT COUNT(*) as count FROM document_embeddings")
        print(f"Documents in database: {doc_count[0]['count'] if doc_count else 0}")
        
        # Check chunk count
        chunk_count = db.execute_query("SELECT COUNT(*) as count FROM document_chunks")
        print(f"Chunks in database: {chunk_count[0]['count'] if chunk_count else 0}")
        
        # Check for any sample documents
        if doc_count and doc_count[0]['count'] > 0:
            sample_docs = db.execute_query("SELECT id, LEFT(content, 100) as content_preview FROM document_embeddings LIMIT 3")
            print("Sample documents:")
            for doc in sample_docs:
                print(f"  ID {doc['id']}: {doc['content_preview']}...")
        
        # Check similarity threshold
        print(f"Similarity threshold: {Config.SIMILARITY_THRESHOLD}")
        
    else:
        print("✗ Failed to connect to database")
except Exception as e:
    print(f"Error: {e}")
finally:
    db.disconnect()
