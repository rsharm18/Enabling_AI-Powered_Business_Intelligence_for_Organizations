#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import db
from app.rag_analysis.embedding_generator import EmbeddingGenerator
from app.rag_analysis.vector_search import VectorSearch
from app.config import Config

print("=== Debug Vector Search ===")
print(f"Similarity threshold: {Config.SIMILARITY_THRESHOLD}")

try:
    if db.connect():
        print("✓ Connected to database")
        
        # Check document count
        doc_count = db.execute_query("SELECT COUNT(*) as count FROM document_embeddings")
        chunk_count = db.execute_query("SELECT COUNT(*) as count FROM document_chunks")
        
        print(f"Documents in database: {doc_count[0]['count'] if doc_count else 0}")
        print(f"Chunks in database: {chunk_count[0]['count'] if chunk_count else 0}")
        
        if doc_count and doc_count[0]['count'] > 0:
            # Get sample documents
            sample_docs = db.execute_query("SELECT id, LEFT(content, 100) as content_preview FROM document_embeddings LIMIT 3")
            print("\nSample documents:")
            for doc in sample_docs:
                print(f"  ID {doc['id']}: {doc['content_preview']}...")
            
            # Test search with different thresholds
            query = "role of AI in BMs"
            print(f"\nTesting search for: '{query}'")
            
            # Initialize search components
            embedding_gen = EmbeddingGenerator()
            vector_search = VectorSearch(embedding_gen)
            
            # Test with very low threshold
            print("\n--- Testing with threshold 0.1 ---")
            results = vector_search.search_documents(query, limit=5, threshold=0.1)
            print(f"Found {len(results)} documents with threshold 0.1")
            for result in results:
                print(f"  Similarity: {result['similarity']:.4f} - {result['content'][:80]}...")
            
            print("\n--- Testing with threshold 0.5 ---")
            results = vector_search.search_documents(query, limit=5, threshold=0.5)
            print(f"Found {len(results)} documents with threshold 0.5")
            for result in results:
                print(f"  Similarity: {result['similarity']:.4f} - {result['content'][:80]}...")
            
            print("\n--- Testing with default threshold ---")
            results = vector_search.search_documents(query, limit=5, threshold=Config.SIMILARITY_THRESHOLD)
            print(f"Found {len(results)} documents with threshold {Config.SIMILARITY_THRESHOLD}")
            for result in results:
                print(f"  Similarity: {result['similarity']:.4f} - {result['content'][:80]}...")
            
            # Test chunks search
            print("\n--- Testing chunks search with threshold 0.1 ---")
            chunk_results = vector_search.search_chunks(query, limit=5, threshold=0.1)
            print(f"Found {len(chunk_results)} chunks with threshold 0.1")
            for result in chunk_results:
                print(f"  Similarity: {result['similarity']:.4f} - {result['chunk_text'][:80]}...")
        else:
            print("No documents found in database")
        
    else:
        print("✗ Failed to connect to database")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    db.disconnect()
