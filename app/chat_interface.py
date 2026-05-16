"""Chat interface for conversational AI with vector search."""

import gradio as gr
import logging
from typing import List, Dict, Any, Optional, Tuple
from app.config import Config
from app.database import db

logger = logging.getLogger(__name__)


class ChatInterface:
    """Conversational chat interface with vector search capabilities."""

    def __init__(self):
        """Initialize chat interface."""
        self.agent = None
        self.chat_history = []

    def _initialize_components(self):
        """Initialize LangGraph agent if not already initialized."""
        if self.agent is None:
            try:
                logger.info("Initializing LangGraph agent (lazy loading)...")
                from app.rag_analysis.langgraph_agent import ConversationalAgent

                self.agent = ConversationalAgent()
                logger.info("LangGraph agent created successfully (components will load on first use)")
            except Exception as e:
                logger.error(f"Failed to initialize LangGraph agent: {e}")
                raise e

    def _get_database_status(self) -> Dict[str, Any]:
        """Get database status for chat context."""
        try:
            doc_count = db.execute_query("SELECT COUNT(*) as count FROM document_embeddings")
            chunk_count = db.execute_query("SELECT COUNT(*) as count FROM document_chunks")

            return {
                'documents': doc_count[0]['count'] if doc_count else 0,
                'chunks': chunk_count[0]['count'] if chunk_count else 0
            }
        except Exception as e:
            logger.error(f"Error getting database status: {e}")
            return {'documents': 0, 'chunks': 0}

    def _format_search_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format search results into a readable response."""
        if not results:
            return f"I couldn't find any relevant information about '{query}' in the documents. The database might be empty or the query might not match any content."

        context_parts = []
        context_parts.append(f"Found {len(results)} relevant results for '{query}':\n")

        for i, result in enumerate(results, 1):
            result_type = result.get('type', 'document')
            similarity = result.get('similarity', 0)

            if result_type == 'document':
                content = result.get('content', '')
            else:  # chunk
                content = result.get('chunk_text', '')

            # Truncate long content
            if len(content) > 300:
                content = content[:300] + "..."

            context_parts.append(f"{i}. [{result_type.title()}] (Similarity: {similarity:.3f})")
            context_parts.append(f"   {content}")
            context_parts.append("")

        return "\n".join(context_parts)

    def chat_response(self, message: str, history: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
        """Generate chat response using LangGraph agent."""
        try:
            history = history or []

            if not message.strip():
                return history, ""

            self._initialize_components()

            db_status = self._get_database_status()

            if db_status['documents'] == 0 and db_status['chunks'] == 0:
                response = (
                    "I don't have any documents to search through. "
                    "Please process some PDF documents first or enable pre-loaded data."
                )
            else:
                response = self.agent.chat(message, history)

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})

            return history, ""

        except Exception as e:
            logger.error(f"Chat response error: {e}")
            error_response = f"Sorry, I encountered an error while processing your question: {str(e)}"

            history = history or []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_response})

            return history, ""

    def _chat_response(self, message: str, history: List[List[str]]) -> List[Dict[str, str]]:
        """Generate chat response using LangGraph agent."""
        try:
            if not message.strip():
                response = "Please ask me a question about the documents."
                return [{"role": "user", "content": message}, {"role": "assistant", "content": response}]

            # Initialize agent
            self._initialize_components()

            # Get database status
            db_status = self._get_database_status()

            if db_status['documents'] == 0 and db_status['chunks'] == 0:
                response = ("I don't have any documents to search through. Please process some PDF documents first "
                           "by using the document processing feature, or set PRE_LOAD_DATA=true in your environment "
                           "and restart the application.")
                return [{"role": "user", "content": message}, {"role": "assistant", "content": response}]

            # Use LangGraph agent for response
            response = self.agent.chat(message, history)

            return [{"role": "user", "content": message}, {"role": "assistant", "content": response}]

        except Exception as e:
            logger.error(f"Chat response error: {e}")
            error_response = f"Sorry, I encountered an error while processing your question: {str(e)}"
            return [{"role": "user", "content": message}, {"role": "assistant", "content": error_response}]

    def clear_chat(self) -> List[List[str]]:
        """Clear chat history."""
        self.chat_history = []
        return []

    def _extract_file_paths(self, files) -> List[str]:
        """Return filesystem paths from Gradio file payloads."""
        if not files:
            return []

        if not isinstance(files, list):
            files = [files]

        paths = []
        for file in files:
            if isinstance(file, str):
                paths.append(file)
            elif isinstance(file, dict):
                path = file.get("name") or file.get("path")
                if not path and isinstance(file.get("file"), dict):
                    path = file["file"].get("path") or file["file"].get("name")
                if path:
                    paths.append(path)
            elif hasattr(file, "name"):
                paths.append(file.name)
            elif hasattr(file, "path"):
                paths.append(file.path)

        return paths

    def upload_documents(self, files) -> str:
        """Index user-uploaded PDFs or pickle files from the chat interface."""
        try:
            file_paths = self._extract_file_paths(files)
            if not file_paths:
                return "Upload at least one PDF or pickle file."

            from app.rag_analysis.document_processor import DocumentProcessor

            processor = DocumentProcessor()
            count = processor.process_uploaded_files(file_paths)
            self.agent = None

            return f"Indexed {count} uploaded document sections from {len(file_paths)} file(s)."
        except Exception as e:
            logger.error(f"Upload processing error: {e}")
            return f"Error processing upload: {str(e)}"

    def reset_and_reload_documents(self) -> str:
        """Reset RAG tables and reload the default PDFs plus CSV analysis pickle."""
        try:
            from app.rag_analysis.document_processor import DocumentProcessor

            processor = DocumentProcessor()
            count = processor.reset_and_reload_documents()
            self.agent = None

            status = (
                f"Reset the RAG document store and reloaded {count} document sections "
                "from the default PDF folder and CSV analysis pickle."
            )
            return status
        except Exception as e:
            logger.error(f"Reset/reload error: {e}")
            return f"Error resetting and reloading documents: {str(e)}"

    def run_evaluation(self, eval_limit: Optional[int]) -> Tuple[str, str, str]:
        """Run QA evaluation from the chat interface and return report details."""
        try:
            from app.evaluation import run_evaluation

            max_cases = int(eval_limit) if eval_limit else None
            report = run_evaluation(max_cases=max_cases)
            summary = report.get("summary", {})
            report_path = summary.get("report_path", str(Config.OUTPUT_DIR / "evaluation_report.json"))
            grader = summary.get("grader", "unknown")

            summary_text = (
                f"Evaluation complete\n"
                f"Total cases: {summary.get('total_cases', 0)}\n"
                f"Passed: {summary.get('passed', 0)}\n"
                f"Failed: {summary.get('failed', 0)}\n"
                f"Pass rate: {summary.get('pass_rate', 0)}\n"
                f"Average score: {summary.get('average_score', 0)}\n"
                f"Evaluator: {grader}"
            )

            return summary_text, grader, report_path
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return f"Error running evaluation: {str(e)}", "error", None

    def create_interface(self) -> gr.Blocks:
        """Create the chat interface."""

        # Custom CSS for better chat appearance
        css = """
        .chat-container {
            height: 600px;
            overflow-y: auto;
        }
        .message.user {
            background-color: #e3f2fd;
            border-radius: 12px;
            padding: 10px;
            margin: 5px;
        }
        .message.assistant {
            background-color: #f5f5f5;
            border-radius: 12px;
            padding: 10px;
            margin: 5px;
        }
        """

        with gr.Blocks(
            title="AI-Powered Business Intelligence - Chat"
        ) as interface:

            gr.Markdown("# AI-Powered Business Intelligence Chat")
            gr.Markdown("Ask questions about your documents and get answers powered by vector search.")

            # Database status display
            with gr.Row():
                db_status_display = gr.JSON(label="Database Status", value=self._get_database_status())

            with gr.Accordion("Document Management", open=False):
                with gr.Row():
                    upload_files = gr.File(
                        label="Upload PDFs or Pickle Files",
                        file_types=[".pdf", ".pkl", ".pickle"],
                        file_count="multiple"
                    )
                    upload_btn = gr.Button("Upload and Index", variant="secondary")
                    reset_reload_btn = gr.Button("Reset DB and Reload Defaults", variant="stop")
                    refresh_status_btn = gr.Button("Refresh Status", variant="secondary")

                document_status = gr.Textbox(
                    label="Document Status",
                    interactive=False,
                    lines=2
                )

            with gr.Accordion("Evaluation", open=False):
                with gr.Row():
                    eval_limit = gr.Number(
                        label="Case Limit",
                        value=4,
                        precision=0,
                        minimum=1
                    )
                    run_eval_btn = gr.Button("Run Evaluation", variant="primary")

                evaluator_name = gr.Textbox(
                    label="Evaluator",
                    interactive=False
                )
                evaluation_summary = gr.Textbox(
                    label="Evaluation Summary",
                    interactive=False,
                    lines=7
                )
                evaluation_report = gr.File(label="Download Evaluation Report")

            # Chat interface
            with gr.Column(elem_classes=["chat-container"]):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    elem_id="chatbot"
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask about AI in business models, data analysis, etc...",
                        lines=2,
                        scale=4
                    )
                    submit = gr.Button("Send", variant="primary", scale=1)
                    clear = gr.Button("Clear", scale=1)

            # Example questions
            gr.Markdown("### Example Questions:")
            examples = [
                "Best Product for Sales",
                "Which product had the highest Sales?",
                "How is AI used in business intelligence systems?",
                "What are the benefits of data-driven decision making?",
                "How can machine learning improve sales forecasting?",
                "Which product had the highest sales and why?"
            ]

            example_buttons = gr.Examples(
                examples=examples,
                inputs=msg,
                label="Click to try these examples:"
            )

            # Event handlers
            # msg.submit(self.chat_response, [msg, chatbot], chatbot)
            # submit.click(self.chat_response, [msg, chatbot], chatbot)

            msg.submit(
                self.chat_response,
                [msg, chatbot],
                [chatbot, msg]
            )

            submit.click(
                self.chat_response,
                [msg, chatbot],
                [chatbot, msg]
            )

            # clear.click(self.clear_chat, outputs=chatbot)

            clear.click(
                lambda: ([], ""),
                outputs=[chatbot, msg]
            )

            upload_btn.click(
                self.upload_documents,
                inputs=[upload_files],
                outputs=document_status
            )

            reset_reload_btn.click(
                self.reset_and_reload_documents,
                outputs=document_status
            )

            refresh_status_btn.click(
                self._get_database_status,
                outputs=db_status_display
            )

            run_eval_btn.click(
                self.run_evaluation,
                inputs=[eval_limit],
                outputs=[evaluation_summary, evaluator_name, evaluation_report]
            )

            # Auto-refresh database status
            interface.load(
                fn=self._get_database_status,
                outputs=db_status_display
            )

        return interface


def launch_chat_interface(host: str = "0.0.0.0", port: int = 7860, share: bool = False, debug: bool = False):
    """Launch the chat interface."""
    try:
        chat_interface = ChatInterface()
        app = chat_interface.create_interface()

        # Custom CSS for better chat appearance
        css = """
        .gradio-container {
            background: #020817 !important;
        }

        /* Chatbot */

        #chatbot {
            min-height: 650px !important;
        }

        /* User bubble */

        .message.user {
            background: #4f46e5 !important;
            color: white !important;
            border-radius: 14px !important;
        }

        .message.user * {
            color: white !important;
        }

        /* Assistant bubble */

        .message.bot {
            background: #111827 !important;
            color: #f3f4f6 !important;
            border: 1px solid #374151 !important;
            border-radius: 14px !important;
        }

        .message.bot * {
            color: #f3f4f6 !important;
        }

        /* Textbox */

        textarea {
            background: #1f2937 !important;
            color: white !important;
            border: 1px solid #374151 !important;
        }

        textarea::placeholder {
            color: #9ca3af !important;
        }

        /* Buttons */

        button.primary {
            background: #4f46e5 !important;
            color: white !important;
        }

        /* Markdown */

        .prose,
        .prose p,
        .prose li {
            color: #f3f4f6 !important;
        }
        """

        logger.info(f"Starting chat interface on {host}:{port}")
        app.launch(
            server_name=host,
            server_port=port,
            share=share,
            show_error=True,
            inbrowser=True,
            theme=gr.themes.Soft(),
            css=css,
            debug=debug
        )

    except Exception as e:
        logger.error(f"Failed to launch chat interface: {e}")
        raise
