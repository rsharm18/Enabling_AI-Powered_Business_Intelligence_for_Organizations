"""LangGraph integration for conversational AI agent."""

import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from app.config import Config
from app.rag_analysis.vector_search import VectorSearch
from app.rag_analysis.embedding_generator import EmbeddingGenerator

# Try to import Groq
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("Groq not available. Install with: pip install langchain-groq")

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the LangGraph agent."""
    messages: Annotated[list, add_messages]
    context: Optional[str]
    query: Optional[str]
    search_results: Optional[List[Dict[str, Any]]]


class ConversationalAgent:
    """Conversational AI agent using LangGraph with vector search."""
    
    def __init__(self):
        """Initialize the conversational agent with lazy loading."""
        self.embedding_generator = None
        self.vector_search = None
        self.llm = None
        self.graph = None
        self._components_initialized = False
        # Don't initialize immediately - will be loaded on first use
    
    def _ensure_components_initialized(self):
        """Ensure components are initialized before use."""
        if not self._components_initialized:
            logger.info("Initializing RAG components (lazy loading)...")
            self._initialize_components()
            self._build_graph()
            self._components_initialized = True
            logger.info("RAG components initialized successfully")
    
    def _initialize_components(self):
        """Initialize RAG components."""
        try:
            self.embedding_generator = EmbeddingGenerator()
            self.vector_search = VectorSearch(self.embedding_generator)

            logger.info(f"RAG components initialization started with Groq: {GROQ_AVAILABLE} and key: {Config.GROQ_API_KEY}")
            # Initialize LLM if Groq is available
            if GROQ_AVAILABLE and Config.GROQ_API_KEY:
                self.llm = ChatGroq(
                    api_key=Config.GROQ_API_KEY,
                    model=Config.GROQ_MODEL,
                    temperature=0.1
                )
                logger.info(f"Groq LLM initialized with model: {Config.GROQ_MODEL}")
            else:
                logger.warning("Groq LLM not available. Using simple response generation.")
                self.llm = None
            
            logger.info("Agent components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent components: {e}")
            raise e
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        
        # Define the nodes
        def retrieve_documents(state: AgentState) -> AgentState:
            """Retrieve relevant documents based on the query."""
            try:
                query = state["messages"][-1].content if state["messages"] else ""
                
                # Perform vector search
                results = self.vector_search.hybrid_search(query, limit=5)
                
                # Format context
                if results:
                    context_parts = []
                    for i, result in enumerate(results, 1):
                        result_type = result.get('type', 'document')
                        similarity = result.get('similarity', 0)
                        
                        if result_type == 'document':
                            content = result.get('content', '')
                        else:
                            content = result.get('chunk_text', '')
                        
                        # Truncate long content
                        if len(content) > 400:
                            content = content[:400] + "..."
                        
                        context_parts.append(f"[Source {i}] {content}")
                    
                    context = "\n\n".join(context_parts)
                else:
                    context = "No relevant documents found for this query."
                
                return {
                    **state,
                    "context": context,
                    "search_results": results,
                    "query": query
                }
                
            except Exception as e:
                logger.error(f"Error in retrieve_documents: {e}")
                return {
                    **state,
                    "context": f"Error retrieving documents: {str(e)}",
                    "search_results": [],
                    "query": state["messages"][-1].content if state["messages"] else ""
                }

        def generate_response(state: AgentState) -> AgentState:
            """Generate a response based on the context and query."""
            try:
                query = state["query"]
                context = state["context"]
                search_results = state["search_results"]
                
                # Use LLM if available, otherwise fall back to simple response
                if self.llm and search_results:
                    # Create prompt for LLM
                    prompt = f"""You are a helpful AI assistant for business intelligence and document analysis. 
Using the provided context from documents, answer the user's question accurately and comprehensively.

User Question: {query}

Context from documents:
{context}

Instructions:
1. Answer the question based on the provided context
2. If the context doesn't contain enough information, acknowledge this
3. Provide a clear, well-structured response
4. Include relevant details from the sources
5. Be conversational and helpful

Answer:"""
                    
                    # Generate response using LLM
                    llm_response = self.llm.invoke(prompt)
                    response = llm_response.content
                    
                else:
                    # Fallback to simple response generation
                    if not search_results:
                        response = (f"I couldn't find any relevant information about '{query}' in the documents. "
                                   "The database might be empty or the query might not match any content. "
                                   "Try rephrasing your question or ensure documents have been processed.")
                    else:
                        # Create a helpful response
                        response_parts = []
                        response_parts.append(f"Based on the documents, here's what I found about '{query}':")
                        response_parts.append("")
                        
                        for i, result in enumerate(search_results, 1):
                            result_type = result.get('type', 'document')
                            similarity = result.get('similarity', 0)
                            
                            if result_type == 'document':
                                content = result.get('content', '')
                            else:
                                content = result.get('chunk_text', '')
                            
                            # Truncate for display
                            if len(content) > 200:
                                content = content[:200] + "..."
                            
                            response_parts.append(f"{i}. [{result_type.title()}] (Relevance: {similarity:.3f})")
                            response_parts.append(f"   {content}")
                            response_parts.append("")
                        
                        response_parts.append(f"Found {len(search_results)} relevant results. Would you like me to elaborate on any specific aspect?")
                        
                        response = "\n".join(response_parts)
                
                # Add AI message to the state
                ai_message = AIMessage(content=response)
                
                return {
                    **state,
                    "messages": state["messages"] + [ai_message]
                }
                
            except Exception as e:
                logger.error(f"Error in generate_response: {e}")
                error_message = AIMessage(content=f"Sorry, I encountered an error: {str(e)}")
                return {
                    **state,
                    "messages": state["messages"] + [error_message]
                }
        
        # Build the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retrieve", retrieve_documents)
        workflow.add_node("generate", generate_response)
        
        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        # Compile the graph
        self.graph = workflow.compile()
        logger.info("LangGraph workflow built successfully")
    
    def chat(self, message: str, history: List[List[str]] = None) -> str:
        """Process a chat message and return the response."""
        try:
            # Ensure components are initialized before use
            self._ensure_components_initialized()
            
            # Convert history to LangChain messages
            messages = []
            
            # Add system message
            system_msg = SystemMessage(content=(
                "You are a helpful AI assistant for business intelligence and document analysis. "
                "Use the provided context from documents to answer questions accurately. "
                "If no relevant information is found, acknowledge this and suggest alternatives."
            ))
            messages.append(system_msg)
            
            # Add conversation history
            if history:
                for item in history:
                    # Handle both list format [human_msg, ai_msg] and dict format
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        human_msg, ai_msg = item[0], item[1]
                        messages.append(HumanMessage(content=human_msg))
                        messages.append(AIMessage(content=ai_msg))
                    elif isinstance(item, dict) and 'role' in item and 'content' in item:
                        if item['role'] == 'user':
                            messages.append(HumanMessage(content=item['content']))
                        elif item['role'] == 'assistant':
                            messages.append(AIMessage(content=item['content']))
            
            # Add current message
            messages.append(HumanMessage(content=message))
            
            # Run the graph
            result = self.graph.invoke({"messages": messages})
            
            # Extract the last AI message
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                return ai_messages[-1].content
            else:
                return "I'm sorry, I couldn't generate a response."
                
        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about the search capabilities."""
        try:
            if self.vector_search:
                return self.vector_search.get_search_stats()
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting search stats: {e}")
            return {}
