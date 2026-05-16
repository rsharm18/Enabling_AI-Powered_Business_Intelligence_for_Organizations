"""Web application interface for the BI platform."""

import gradio as gr
import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from pathlib import Path

from app.config import Config
from app.database import db
from app.csv_analysis import CSVAnalyzer, CSVSchema

logger = logging.getLogger(__name__)


class BIWebInterface:
    """Main web interface for the Business Intelligence platform."""

    def __init__(self):
        """Initialize the web interface."""
        self.csv_analyzer = None
        self.document_processor = None
        self.embedding_generator = None
        self.vector_search = None

        # Ensure directories exist
        Config.ensure_directories()

    def _initialize_rag_components(self):
        """Initialize RAG components if not already initialized."""
        if self.document_processor is None:
            try:
                from app.rag_analysis.document_processor import DocumentProcessor
                from app.rag_analysis.embedding_generator import EmbeddingGenerator
                from app.rag_analysis.vector_search import VectorSearch

                self.document_processor = DocumentProcessor()
                self.embedding_generator = EmbeddingGenerator()
                self.vector_search = VectorSearch(self.embedding_generator)
            except ImportError as e:
                logger.error(f"Failed to initialize RAG components: {e}")
                raise e

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""

        with gr.Blocks(
            title="AI-Powered Business Intelligence Platform",
            css=self._get_custom_css()
        ) as interface:

            gr.Markdown("# AI-Powered Business Intelligence Platform")
            gr.Markdown("Analyze CSV data, process documents, and generate intelligent insights.")

            with gr.Tabs():
                # CSV Analysis Tab
                with gr.Tab("CSV Analysis"):
                    self._create_csv_analysis_tab()

                # Document Processing Tab
                with gr.Tab("Document Processing"):
                    self._create_document_processing_tab()

                # RAG Search Tab
                with gr.Tab("Intelligent Search"):
                    self._create_rag_search_tab()

                # Database Status Tab
                with gr.Tab("Database Status"):
                    self._create_database_status_tab()

        return interface

    def _create_csv_analysis_tab(self):
        """Create CSV analysis interface."""

        with gr.Row():
            with gr.Column(scale=2):
                csv_file = gr.File(
                    label="Upload CSV File",
                    file_types=[".csv"],
                    file_count="single"
                )

                with gr.Accordion("CSV Schema Configuration", open=True):
                    date_column = gr.Textbox(label="Date Column", value="Date")

                    with gr.Row():
                        measures = gr.Textbox(
                            label="Measure Columns (comma-separated)",
                            value="Sales",
                            placeholder="e.g., Sales,Revenue,Profit"
                        )
                        dimensions = gr.Textbox(
                            label="Dimension Columns (comma-separated)",
                            value="Product,Region,Customer_Gender",
                            placeholder="e.g., Product,Region,Category"
                        )

                    with gr.Row():
                        customer_attrs = gr.Textbox(
                            label="Customer Attributes (comma-separated, optional)",
                            value="Customer_Age,Customer_Gender",
                            placeholder="e.g., Age,Gender,Segment"
                        )
                        satisfaction_col = gr.Textbox(
                            label="Satisfaction Column (optional)",
                            value="Customer_Satisfaction",
                            placeholder="e.g., Satisfaction,Rating"
                        )

                analyze_btn = gr.Button("Analyze CSV", variant="primary", size="lg")

            with gr.Column(scale=3):
                with gr.Accordion("Analysis Results", open=True):
                    analysis_output = gr.Textbox(
                        label="Analysis Summary",
                        lines=15,
                        max_lines=20
                    )

                with gr.Row():
                    download_pickle = gr.File(label="Download Analysis (Pickle)")
                    download_dashboard = gr.File(label="Download Dashboard (PNG)")

        # Event handlers
        analyze_btn.click(
            fn=self._analyze_csv,
            inputs=[csv_file, date_column, measures, dimensions, customer_attrs, satisfaction_col],
            outputs=[analysis_output, download_pickle, download_dashboard]
        )

    def _create_document_processing_tab(self):
        """Create document processing interface."""

        with gr.Row():
            with gr.Column(scale=1):
                pdf_dir = gr.Textbox(
                    label="PDF Directory",
                    value=str(Config.DATA_DIR / 'PDF Folder'),
                    placeholder="Path to directory containing PDF files"
                )

                process_btn = gr.Button("Process Documents", variant="primary")
                reset_reload_btn = gr.Button("Reset DB and Reload PDFs/Pickle", variant="stop")
                embed_btn = gr.Button("Generate Embeddings", variant="secondary")

                with gr.Accordion("Processing Options"):
                    batch_size = gr.Slider(
                        label="Batch Size",
                        minimum=10,
                        maximum=500,
                        value=100,
                        step=10
                    )

            with gr.Column(scale=2):
                with gr.Accordion("Document Statistics", open=True):
                    doc_stats = gr.JSON(label="Document Processing Stats")

                with gr.Accordion("Processing Log"):
                    process_log = gr.Textbox(
                        label="Processing Log",
                        lines=10,
                        max_lines=15
                    )

        # Event handlers
        process_btn.click(
            fn=self._process_documents,
            inputs=[pdf_dir],
            outputs=[doc_stats, process_log]
        )

        reset_reload_btn.click(
            fn=self._reset_and_reload_documents,
            inputs=[pdf_dir],
            outputs=[doc_stats, process_log]
        )

        embed_btn.click(
            fn=self._generate_embeddings,
            inputs=[batch_size],
            outputs=[doc_stats, process_log]
        )

    def _create_rag_search_tab(self):
        """Create RAG search interface."""

        with gr.Row():
            with gr.Column(scale=1):
                search_query = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter your question or search terms...",
                    lines=2
                )

                with gr.Accordion("Search Options"):
                    search_type = gr.Radio(
                        label="Search Type",
                        choices=["Hybrid", "Documents Only", "Chunks Only"],
                        value="Hybrid"
                    )

                    similarity_threshold = gr.Slider(
                        label="Similarity Threshold",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.05
                    )

                    max_results = gr.Slider(
                        label="Maximum Results",
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1
                    )

                search_btn = gr.Button("Search", variant="primary", size="lg")

            with gr.Column(scale=2):
                with gr.Accordion("Search Results", open=True):
                    search_results = gr.JSON(label="Results")

                with gr.Accordion("Context"):
                    search_context = gr.Textbox(
                        label="Retrieved Context",
                        lines=10,
                        max_lines=20
                    )

        # Event handlers
        search_btn.click(
            fn=self._perform_search,
            inputs=[search_query, search_type, similarity_threshold, max_results],
            outputs=[search_results, search_context]
        )

    def _create_database_status_tab(self):
        """Create database status interface."""

        with gr.Row():
            with gr.Column():
                refresh_btn = gr.Button("Refresh Status", variant="primary")

                with gr.Accordion("Database Statistics", open=True):
                    db_stats = gr.JSON(label="Database Stats")

                with gr.Accordion("Embedding Statistics"):
                    embedding_stats = gr.JSON(label="Embedding Stats")

                with gr.Accordion("Search Statistics"):
                    search_stats = gr.JSON(label="Search Stats")

        # Event handlers
        refresh_btn.click(
            fn=self._get_database_status,
            outputs=[db_stats, embedding_stats, search_stats]
        )

    def _analyze_csv(self, csv_file, date_column, measures, dimensions,
                    customer_attrs, satisfaction_col) -> Tuple[str, str, str]:
        """Analyze CSV file."""
        try:
            if not csv_file:
                return "Please upload a CSV file.", None, None

            # Parse schema
            measures_list = [m.strip() for m in measures.split(',') if m.strip()]
            dimensions_list = [d.strip() for d in dimensions.split(',') if d.strip()]
            customer_attrs_list = [ca.strip() for ca in customer_attrs.split(',') if ca.strip()] if customer_attrs else []

            schema = CSVSchema(
                date_column=date_column.strip(),
                measures=measures_list,
                dimensions=dimensions_list,
                customer_attributes=customer_attrs_list if customer_attrs_list else None,
                satisfaction_column=satisfaction_col.strip() if satisfaction_col else None
            )

            # Validate schema
            errors = schema.validate()
            if errors:
                return f"Schema validation errors: {', '.join(errors)}", None, None

            # Create analyzer and run analysis
            self.csv_analyzer = CSVAnalyzer(csv_file.name, schema.to_dict())
            results = self.csv_analyzer.run_full_analysis()

            # Store results in database
            db.store_analysis_result('csv_analysis', csv_file.name, results)

            # Generate summary
            summary = self._generate_csv_summary(results)

            # Save files
            pickle_path = self.csv_analyzer.save_analysis_to_pickle()
            dashboard_path = self.csv_analyzer.create_visualizations()

            return summary, pickle_path, dashboard_path

        except Exception as e:
            logger.error(f"CSV analysis error: {e}")
            return f"Error analyzing CSV: {str(e)}", None, None

    def _process_documents(self, pdf_dir: str) -> Tuple[Dict[str, Any], str]:
        """Process documents from directory."""
        try:
            from app.rag_analysis.document_processor import DocumentProcessor

            # Initialize RAG components
            self._initialize_rag_components()

            # Update processor directory
            self.document_processor = DocumentProcessor(pdf_dir)

            # Process documents
            count = self.document_processor.process_and_store_documents(force=True)

            # Get stats
            stats = self.document_processor.get_document_stats()

            log = f"Processed {count} documents from {pdf_dir}"

            return stats, log

        except Exception as e:
            logger.error(f"Document processing error: {e}")
            return {}, f"Error processing documents: {str(e)}"

    def _reset_and_reload_documents(self, pdf_dir: str) -> Tuple[Dict[str, Any], str]:
        """Clear RAG documents, then reload PDFs and the CSV analysis pickle."""
        try:
            from app.rag_analysis.document_processor import DocumentProcessor

            self._initialize_rag_components()
            self.document_processor = DocumentProcessor(pdf_dir)

            count = self.document_processor.reset_and_reload_documents()
            stats = self.document_processor.get_document_stats()

            log = (
                f"Reset the RAG document store and reloaded {count} document sections "
                f"from {pdf_dir} plus any csv_analysis_results.pkl file."
            )
            return stats, log

        except Exception as e:
            logger.error(f"Reset/reload error: {e}")
            return {}, f"Error resetting and reloading documents: {str(e)}"

    def _generate_embeddings(self, batch_size: int) -> Tuple[Dict[str, Any], str]:
        """Generate embeddings for documents."""
        try:
            # Initialize RAG components
            self._initialize_rag_components()

            # Generate document embeddings
            doc_count = self.embedding_generator.embed_and_store_documents(batch_size)

            # Generate chunk embeddings
            chunk_count = self.embedding_generator.embed_and_store_chunks(batch_size)

            # Get stats
            stats = self.embedding_generator.get_embedding_stats()

            log = f"Generated embeddings for {doc_count} documents and {chunk_count} chunks"

            return stats, log

        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return {}, f"Error generating embeddings: {str(e)}"

    def _perform_search(self, query: str, search_type: str, threshold: float,
                       max_results: int) -> Tuple[List[Dict[str, Any]], str]:
        """Perform search based on query."""
        try:
            if not query.strip():
                return [], "Please enter a search query."

            # Initialize RAG components
            self._initialize_rag_components()

            # Perform search based on type
            if search_type == "Documents Only":
                results = self.vector_search.search_documents(query, max_results, threshold)
            elif search_type == "Chunks Only":
                results = self.vector_search.search_chunks(query, max_results, threshold)
            else:  # Hybrid
                results = self.vector_search.hybrid_search(query, limit=max_results)

            # Get context
            context = self.vector_search.get_context_for_query(query)

            return results, context

        except Exception as e:
            logger.error(f"Search error: {e}")
            return [], f"Search error: {str(e)}"

    def _get_database_status(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Get database status information."""
        try:
            # Database stats
            doc_count = db.execute_query("SELECT COUNT(*) as count FROM document_embeddings")
            chunk_count = db.execute_query("SELECT COUNT(*) as count FROM document_chunks")
            analysis_count = db.execute_query("SELECT COUNT(*) as count FROM analysis_results")

            db_stats = {
                'documents': doc_count[0]['count'] if doc_count else 0,
                'chunks': chunk_count[0]['count'] if chunk_count else 0,
                'analysis_results': analysis_count[0]['count'] if analysis_count else 0
            }

            # Only initialize RAG components if needed and not already initialized
            embedding_stats = {}
            search_stats = {}

            # Only get RAG stats if we have documents (otherwise no need to load models)
            if db_stats.get('documents', 0) > 0 or db_stats.get('chunks', 0) > 0:
                try:
                    # Only initialize if not already done
                    if self.embedding_generator is None:
                        self._initialize_rag_components()

                    # Embedding stats
                    if self.embedding_generator:
                        embedding_stats = self.embedding_generator.get_embedding_stats()

                    # Search stats
                    if self.vector_search:
                        search_stats = self.vector_search.get_search_stats()

                except Exception as rag_error:
                    logger.error(f"RAG components error: {rag_error}")
                    embedding_stats = {'error': str(rag_error)}
                    search_stats = {'error': str(rag_error)}
            else:
                # No documents, no need to load models
                embedding_stats = {'message': 'No documents processed yet'}
                search_stats = {'message': 'No documents processed yet'}

            return db_stats, embedding_stats, search_stats

        except Exception as e:
            logger.error(f"Database status error: {e}")
            return {'error': str(e)}, {'error': str(e)}, {'error': str(e)}

    def _generate_csv_summary(self, results: Dict[str, Any]) -> str:
        """Generate a summary of CSV analysis results."""
        summary_parts = []

        # Overview
        if 'overview' in results:
            summary_parts.append("## Overview")
            for measure, stats in results['overview'].items():
                summary_parts.append(f"**{measure}**: ${stats['total']:,.2f} total, ${stats['mean']:,.2f} avg")

        # Key insights
        if 'key_insights' in results:
            summary_parts.append("\n## Key Insights")
            for insight in results['key_insights']:
                summary_parts.append(f"- {insight}")

        # Top dimensions
        if 'dimension_analysis' in results:
            summary_parts.append("\n## Top Performing Dimensions")
            for dimension, measures in results['dimension_analysis'].items():
                if measures:
                    first_measure = list(measures.keys())[0]
                    top_category = max(measures[first_measure].items(), key=lambda x: x[1]['total'])
                    summary_parts.append(f"**{dimension}**: {top_category[0]} (${top_category[1]['total']:,.2f})")

        # Metadata
        if 'metadata' in results:
            summary_parts.append("\n## Analysis Info")
            summary_parts.append(f"**Date**: {results['metadata']['analysis_date']}")
            summary_parts.append(f"**Source**: {results['metadata']['data_source']}")

        return "\n".join(summary_parts)

    def _get_custom_css(self) -> str:
        """Get custom CSS for the interface."""
        return """
        .gradio-container {
            max-width: 1400px !important;
        }
        .gradio-button {
            border-radius: 8px !important;
        }
        """


def get_custom_css() -> str:
    """Get custom CSS for the interface."""
    return """
    .gradio-container {
        max-width: 1400px !important;
    }
    .gradio-button {
        border-radius: 8px !important;
    }
    """


def create_app() -> gr.Blocks:
    """Create and return the Gradio application."""
    interface = BIWebInterface()
    return interface.create_interface()


def launch_app(host: str = None, port: int = None, share: bool = False, debug: bool = False):
    """Launch the web application."""
    host = host or Config.WEB_HOST
    port = port or Config.WEB_PORT

    app = create_app()

    logger.info(f"Launching web interface on {host}:{port}")
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True,
        theme=gr.themes.Soft(),
        debug=debug
    )
