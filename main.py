"""
Main entry point for the Aequitas-Fin reasoning agent.
"""
from config.settings import settings
from src.core.database import QdrantDatabase
from src.core.models import LLMModels
from src.reasoning.tools import TavilySearchTool, LocalRetrievalTool
from src.reasoning.graph import create_reasoning_graph


def main():
    """
    Main function to demonstrate the Aequitas-Fin agent.
    """
    print("=" * 60)
    print("Aequitas-Fin - Financial Reasoning Agent")
    print("=" * 60)
    
    # Validate settings
    settings.validate()
    
    # Initialize database
    print("\n[1/4] Initializing Qdrant database...")
    db = QdrantDatabase(path=settings.QDRANT_PATH)
    print(f"✓ Database initialized at {settings.QDRANT_PATH}")
    
    # Initialize LLM
    print("\n[2/4] Initializing LLM...")
    models = LLMModels(
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.MODEL_TEMPERATURE,
        max_tokens=settings.MODEL_MAX_TOKENS
    )
    llm = models.get_default_model()
    print(f"✓ LLM initialized: {settings.DEFAULT_MODEL}")
    
    # Initialize tools
    print("\n[3/4] Initializing tools...")
    
    # Web search tool (optional)
    search_tool = None
    if settings.TAVILY_API_KEY:
        try:
            search_tool = TavilySearchTool(
                api_key=settings.TAVILY_API_KEY,
                max_results=settings.TAVILY_MAX_RESULTS
            )
            print("✓ Tavily search tool initialized")
        except Exception as e:
            print(f"⚠ Tavily search tool not available: {e}")
    else:
        print("⚠ Tavily API key not set - web search disabled")
    
    # Local retrieval tool (requires embeddings - placeholder for now)
    retrieval_tool = None
    print("⚠ Local retrieval requires embeddings - currently disabled")
    print("  (Use langchain_community.embeddings to enable)")
    
    # Build reasoning graph
    print("\n[4/4] Building reasoning graph...")
    graph = create_reasoning_graph(
        llm=llm,
        retrieval_tool=retrieval_tool,
        search_tool=search_tool
    )
    print("✓ Reasoning graph built successfully")
    
    # Interactive loop
    print("\n" + "=" * 60)
    print("Agent ready! Type 'exit' or 'quit' to stop.")
    print("=" * 60)
    
    while True:
        try:
            # Get user input
            query = input("\nYour query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!")
                break
            
            # Run the graph
            print("\nProcessing...")
            result = graph.run(query, max_iterations=settings.MAX_ITERATIONS)
            
            # Display result
            answer = result.get("answer", "No answer generated")
            print(f"\n{'-' * 60}")
            print(f"Answer:\n{answer}")
            print(f"{'-' * 60}")
            
            # Display debug info
            if result.get("retrieved_docs"):
                print(f"\n📚 Retrieved {len(result['retrieved_docs'])} local documents")
            if result.get("web_results"):
                print(f"🌐 Retrieved {len(result['web_results'])} web results")
            print(f"🔄 Iterations: {result.get('iteration', 0)}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
