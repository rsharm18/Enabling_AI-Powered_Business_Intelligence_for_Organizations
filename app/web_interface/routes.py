"""Route handlers for the web interface."""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from database import db
from csv_analysis import CSVAnalyzer, CSVSchema
from config import Config

logger = logging.getLogger(__name__)


def setup_routes(app):
    """Setup additional routes if needed."""
    # This can be used for additional API endpoints if needed
    pass


class RouteHandlers:
    """Collection of route handlers for the web interface."""
    
    def __init__(self):
        """Initialize route handlers."""
        self.document_processor = DocumentProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_search = VectorSearch(self.embedding_generator)
    
    def get_available_csv_files(self) -> List[str]:
        """Get list of available CSV files in data directory."""
        try:
            csv_files = list(Config.DATA_DIR.glob("*.csv"))
            return [f.name for f in csv_files]
        except Exception as e:
            logger.error(f"Error getting CSV files: {e}")
            return []
    
    def get_available_pdf_files(self) -> List[str]:
        """Get list of available PDF files."""
        try:
            pdf_dir = Config.DATA_DIR / 'PDF Folder'
            if pdf_dir.exists():
                pdf_files = list(pdf_dir.glob("*.pdf"))
                return [f.name for f in pdf_files]
            return []
        except Exception as e:
            logger.error(f"Error getting PDF files: {e}")
            return []
    
    def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent analysis results from database."""
        try:
            results = db.get_analysis_results(limit=limit)
            return results
        except Exception as e:
            logger.error(f"Error getting analysis history: {e}")
            return []
    
    def delete_analysis_result(self, result_id: int) -> bool:
        """Delete an analysis result from database."""
        try:
            db.execute_query("DELETE FROM analysis_results WHERE id = %s", (result_id,))
            return True
        except Exception as e:
            logger.error(f"Error deleting analysis result: {e}")
            return False
    
    def export_analysis_results(self, result_id: int, format: str = 'json') -> Optional[str]:
        """Export analysis results in specified format."""
        try:
            # Get result from database
            results = db.execute_query(
                "SELECT * FROM analysis_results WHERE id = %s", 
                (result_id,)
            )
            
            if not results:
                return None
            
            result = results[0]
            
            if format == 'json':
                import json
                output_path = Config.OUTPUT_DIR / f"analysis_{result_id}.json"
                with open(output_path, 'w') as f:
                    json.dump(result['results'], f, indent=2)
                return str(output_path)
            
            elif format == 'csv':
                # Convert results to CSV if possible
                import pandas as pd
                
                # This would need to be customized based on the result structure
                output_path = Config.OUTPUT_DIR / f"analysis_{result_id}.csv"
                # Placeholder implementation
                return str(output_path)
            
            return None
            
        except Exception as e:
            logger.error(f"Error exporting analysis results: {e}")
            return None
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health information."""
        try:
            # Check database connection
            db_connected = db.connect()
            
            # Check embedding model
            embedding_model_loaded = self.embedding_generator.model is not None
            
            # Check directories
            data_dir_exists = Config.DATA_DIR.exists()
            output_dir_exists = Config.OUTPUT_DIR.exists()
            
            return {
                'database_connected': db_connected,
                'embedding_model_loaded': embedding_model_loaded,
                'data_directory_exists': data_dir_exists,
                'output_directory_exists': output_dir_exists,
                'system_status': 'healthy' if all([
                    db_connected, embedding_model_loaded, data_dir_exists, output_dir_exists
                ]) else 'unhealthy'
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {'system_status': 'error', 'error': str(e)}
    
    def cleanup_old_data(self, days_old: int = 30) -> Dict[str, int]:
        """Clean up old data from database."""
        try:
            # This would implement cleanup logic for old data
            # For now, return placeholder
            return {
                'documents_cleaned': 0,
                'chunks_cleaned': 0,
                'analysis_results_cleaned': 0
            }
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return {}
