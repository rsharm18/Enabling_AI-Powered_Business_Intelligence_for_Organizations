"""Main entry point for the AI-Powered Business Intelligence Platform."""

import logging
import time
import argparse
import traceback
from pathlib import Path

from config import Config
from database import db
from csv_analysis import CSVAnalyzer, CSVSchema, get_sales_schema
from web_interface import launch_app
from chat_interface import launch_chat_interface

logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)


def initialize_database_with_retry(max_retries=5, retry_delay=10):
    """Initialize database with retry logic."""
    for attempt in range(max_retries):
        logger.info(f"Attempting database initialization (attempt {attempt + 1}/{max_retries})")
        
        if db.initialize_database():
            logger.info("Database initialized successfully")
            return True
        else:
            if attempt < max_retries - 1:
                logger.warning(f"Database initialization failed, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Database initialization failed after all retries")
                return False


def run_csv_analysis(data_file: str = None, schema_file: str = None):
    """Run CSV analysis on specified file."""
    try:
        # Determine data file
        if not data_file:
            data_file = Config.DATA_DIR / 'sales_data.csv'
        
        data_path = Path(data_file)
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            return False
        
        # Load schema
        if schema_file:
            schema_dict = Config.load_schema_from_file(schema_file)
            schema = CSVSchema.from_dict(schema_dict)
        else:
            schema = get_sales_schema()
        
        # Run analysis
        logger.info(f"Analyzing CSV file: {data_path}")
        analyzer = CSVAnalyzer(str(data_path), schema.to_dict())
        results = analyzer.run_full_analysis()
        
        # Store results
        db.store_analysis_result('csv_analysis', data_path.name, results)
        
        # Print summary
        analyzer.print_summary()
        
        # Save outputs
        pickle_path = analyzer.save_analysis_to_pickle()
        dashboard_path = analyzer.create_visualizations()
        
        logger.info(f"Analysis complete. Results: {pickle_path}, Dashboard: {dashboard_path}")
        return True
        
    except Exception as e:
        logger.error(f"CSV analysis failed: {e}")
        return False


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description='AI-Powered Business Intelligence Platform')
    parser.add_argument('--mode', choices=['web', 'chat', 'csv', 'process'], default='chat',
                       help='Application mode: chat (conversational), web (full interface), csv (analysis), process (documents)')
    parser.add_argument('--data-file', help='Path to CSV file for analysis')
    parser.add_argument('--schema-file', help='Path to schema file')
    parser.add_argument('--host', default=Config.WEB_HOST, help='Web interface host')
    parser.add_argument('--port', type=int, default=Config.WEB_PORT, help='Web interface port')
    parser.add_argument('--share', action='store_true', help='Share web interface publicly')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with live code refresh')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AI-POWERED BUSINESS INTELLIGENCE PLATFORM")
    print("=" * 60)
    
    # Initialize database
    logger.info("Initializing database...")
    if not initialize_database_with_retry():
        logger.error("Failed to initialize database. Exiting...")
        return
    
    print("Database initialized successfully!")
    
    # Ensure directories exist
    Config.ensure_directories()
    
    # Run based on mode
    try:
        if args.mode == 'chat':
            logger.info(f"Starting chat interface on {args.host}:{args.port}")
            launch_chat_interface(host=args.host, port=args.port, share=args.share, debug=args.debug)
            
        elif args.mode == 'web':
            logger.info(f"Starting full web interface on {args.host}:{args.port}")
            launch_app(host=args.host, port=args.port, share=args.share, debug=args.debug)
            
        elif args.mode == 'csv':
            logger.info("Running CSV analysis mode")
            success = run_csv_analysis(args.data_file, args.schema_file)
            if not success:
                logger.error("CSV analysis failed")
                
        elif args.mode == 'process':
            logger.info("Running document processing mode")
            from rag_analysis.document_processor import DocumentProcessor
            
            # Process documents with embeddings (using Hugging Face)
            logger.info("Processing documents with embeddings... Start")
            processor = DocumentProcessor()
            doc_count = processor.process_and_store_documents()
            logger.info(f"Processed and stored {doc_count} documents with embeddings")
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        logger.error(f"Application error: {e}\n{traceback.format_exc()}")
    finally:
        db.disconnect()


if __name__ == "__main__":
    main()