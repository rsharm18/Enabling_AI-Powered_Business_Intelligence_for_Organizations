#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import db

def check_database():
    """Check what's actually in the database"""
    print("Checking database contents...")
    
    if not db.connect():
        print("Failed to connect to database")
        return
    
    try:
        # Check document_embeddings
        result = db.execute_query("SELECT COUNT(*) as count FROM document_embeddings")
        doc_count = result[0]["count"] if result else 0
        print(f"Documents in document_embeddings table: {doc_count}")
        
        # Check document_chunks
        result = db.execute_query("SELECT COUNT(*) as count FROM document_chunks")
        chunk_count = result[0]["count"] if result else 0
        print(f"Chunks in document_chunks table: {chunk_count}")
        
        # Show some sample data if exists
        if doc_count > 0:
            result = db.execute_query("SELECT id, LEFT(content, 100) as content_preview, metadata FROM document_embeddings LIMIT 3")
            print("\nSample documents:")
            for row in result:
                print(f"  ID {row['id']}: {row['content_preview']}...")
                print(f"  Metadata: {row['metadata']}")
        
        if chunk_count > 0:
            result = db.execute_query("SELECT id, document_id, LEFT(chunk_text, 100) as chunk_preview FROM document_chunks LIMIT 3")
            print("\nSample chunks:")
            for row in result:
                print(f"  ID {row['id']} (Doc ID {row['document_id']}): {row['chunk_preview']}...")
        
    except Exception as e:
        print(f"Error checking database: {e}")
    finally:
        db.disconnect()

if __name__ == "__main__":
    check_database()
