# Aequitas-Fin 🏦

An intelligent financial reasoning agent powered by LangGraph, combining RAG (Retrieval-Augmented Generation) and web search capabilities.

## 🌟 Features

- **Qdrant Vector Database**: Local vector storage for document retrieval
- **Multi-Source RAG**: Combines local knowledge base with web search
- **LangGraph State Machine**: Intelligent routing between retrieval and search
- **Flexible LLM Support**: Compatible with Llama-3 and Mistral via Ollama
- **Modular Architecture**: Clean, documented, and extensible codebase

## 📁 Project Structure

```
Aequitas-Fin/
├── config/
│   ├── settings.py         # Application configuration
│   └── prompts.py          # Prompt templates
├── src/
│   ├── core/
│   │   ├── database.py     # Qdrant client connection
│   │   └── models.py       # LLM model wrappers
│   ├── reasoning/
│   │   ├── state.py        # LangGraph state definitions
│   │   ├── tools.py        # RAG and web search tools
│   │   ├── nodes.py        # Graph node implementations
│   │   └── graph.py        # LangGraph state machine
│   ├── ingestion/
│   │   └── ingest_docs.py  # Document ingestion (to be implemented)
│   └── evaluation/
│       ├── metrics.py      # Evaluation metrics (to be implemented)
│       └── run_evals.py    # Evaluation runner (to be implemented)
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
└── .env.example           # Environment variable template
```

## 🚀 Installation

### Prerequisites

1. **Python 3.9+**
2. **Ollama** (for running local LLMs)
   ```bash
   # Install Ollama from https://ollama.ai
   # Pull models:
   ollama pull llama3
   ollama pull mistral
   ```

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Daivik2605/Aequitas-Fin.git
   cd Aequitas-Fin
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

4. Get a Tavily API key (optional, for web search):
   - Sign up at https://tavily.com
   - Add your API key to `.env`:
     ```
     TAVILY_API_KEY=your_key_here
     ```

## 💻 Usage

### Running the Agent

```bash
python main.py
```

The interactive agent will start and you can ask questions:

```
Your query: What are the latest trends in fintech?
Processing...
------------------------------------------------------------
Answer:
[Agent response with information from web search and/or local docs]
------------------------------------------------------------
```

### Programmatic Usage

```python
from config.settings import settings
from src.core.database import QdrantDatabase
from src.core.models import LLMModels
from src.reasoning.tools import TavilySearchTool
from src.reasoning.graph import create_reasoning_graph

# Initialize components
db = QdrantDatabase(path=settings.QDRANT_PATH)
models = LLMModels()
llm = models.get_llama3()

# Initialize tools
search_tool = TavilySearchTool(api_key=settings.TAVILY_API_KEY)

# Create reasoning graph
graph = create_reasoning_graph(
    llm=llm,
    search_tool=search_tool
)

# Run a query
result = graph.run("What is the current state of the stock market?")
print(result["answer"])
```

## 🏗️ Architecture

### Core Components

#### 1. Database (`src/core/database.py`)
- **QdrantDatabase**: Manages vector database connections
- Supports local persistent storage and cloud instances
- Handles collection creation, vector upserts, and similarity search

#### 2. Models (`src/core/models.py`)
- **LLMModels**: Wrapper for Llama-3 and Mistral models
- Uses Ollama backend for local LLM execution
- Configurable temperature and token limits

#### 3. State (`src/reasoning/state.py`)
- **AgentState**: TypedDict defining the data flow through the graph
- Tracks query, messages, retrieved documents, web results, and answers

#### 4. Tools (`src/reasoning/tools.py`)
- **TavilySearchTool**: Web search via Tavily API
- **LocalRetrievalTool**: Vector similarity search in Qdrant
- Both convertible to LangChain Tool format

#### 5. Nodes (`src/reasoning/nodes.py`)
- **router_node**: Decides between RAG, web search, or answer generation
- **rag_retrieval_node**: Fetches documents from local database
- **web_search_node**: Searches the web for current information
- **generate_answer_node**: Synthesizes information into final answer

#### 6. Graph (`src/reasoning/graph.py`)
- **ReasoningGraph**: LangGraph state machine
- Orchestrates routing, retrieval, search, and generation
- Supports streaming and async execution

### Data Flow

```
User Query
    ↓
Router Node (decides action)
    ↓
    ├─→ RAG Retrieval ──┐
    ├─→ Web Search ─────┤
    │                   ↓
    └─→ Generate Answer (synthesizes information)
         ↓
    Final Answer
```

## 🔧 Configuration

Key settings in `.env`:

```bash
# Database
QDRANT_PATH=./qdrant_storage
QDRANT_COLLECTION=aequitas_documents

# LLM
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=llama3
MODEL_TEMPERATURE=0.7

# Search
TAVILY_API_KEY=your_key_here

# Agent
MAX_ITERATIONS=5
```

## 📝 Key Implementation Details

### Qdrant Database Class
- Flexible initialization (local path, host:port, or cloud URL)
- Vector CRUD operations with error handling
- Collection management and similarity search

### LangGraph State Machine
- Conditional routing based on query analysis
- Iterative refinement with max iteration limits
- Clean state management with TypedDict

### Modular Tool System
- Tools can be enabled/disabled independently
- LangChain-compatible tool interface
- Easy to extend with new tools

## 🧪 Testing

```bash
# Test database connectivity
python -c "from src.core.database import QdrantDatabase; db = QdrantDatabase(path='./test_db'); print('✓ Database OK')"

# Test model initialization
python -c "from src.core.models import LLMModels; models = LLMModels(); print('✓ Models OK')"

# Test imports
python -c "from src.reasoning.graph import create_reasoning_graph; print('✓ Graph OK')"
```

## 🤝 Contributing

This is a modular, well-documented codebase designed for easy extension:

1. **Add new tools**: Extend `src/reasoning/tools.py`
2. **Add new nodes**: Extend `src/reasoning/nodes.py`
3. **Customize prompts**: Edit `config/prompts.py`
4. **Add document ingestion**: Implement `src/ingestion/ingest_docs.py`

## 📄 License

[Your License Here]

## 👥 Authors

- Daivik2605

## 🙏 Acknowledgments

- LangChain & LangGraph for the framework
- Qdrant for vector database
- Tavily for web search API
- Ollama for local LLM serving
