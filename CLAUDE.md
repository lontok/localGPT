# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Document Ingestion
```bash
# Ingest documents with CUDA (default)
python ingest.py

# Ingest documents with CPU
python ingest.py --device_type cpu

# Ingest documents with Apple Silicon
python ingest.py --device_type mps

# Ingest documents with Intel HPU
python ingest.py --device_type hpu
```

### Running LocalGPT
```bash
# Run CLI interface with CUDA (default)
python run_localGPT.py

# Run with specific device
python run_localGPT.py --device_type mps  # Apple Silicon
python run_localGPT.py --device_type cpu  # CPU only
python run_localGPT.py --device_type hpu  # Intel HPU

# Run with additional options
python run_localGPT.py --show_sources     # Show source chunks
python run_localGPT.py --use_history      # Enable chat history
python run_localGPT.py --save_qa          # Save Q&A to CSV
```

### Running the API
```bash
# Start the API server
python run_localGPT_API.py
```

### Running the GUI
```bash
# Option 1: Streamlit UI
python localGPT_UI.py

# Option 2: Flask UI (requires API running)
# Terminal 1:
python run_localGPT_API.py
# Terminal 2:
cd localGPTUI && python localGPTUI.py
# Open http://localhost:5111/
```

### Code Quality
```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Run specific linters
black . --line-length 119
isort . --profile black --line-length 119
flake8 . --max-line-length=119 --extend-ignore=E203
```

## Architecture

LocalGPT is a Retrieval-Augmented Generation (RAG) application designed for complete privacy. Key components:

### Core Components
- **Document Ingestion** (`ingest.py`): Processes documents from `SOURCE_DOCUMENTS/` and creates embeddings stored in ChromaDB
- **Query Engine** (`run_localGPT.py`): Uses local LLMs to answer questions based on ingested documents
- **API Layer** (`run_localGPT_API.py`): Flask-based REST API for programmatic access
- **User Interfaces**: Two options - Streamlit (`localGPT_UI.py`) and Flask web UI (`localGPTUI/`)

### Key Technologies
- **LangChain**: Core framework for LLM orchestration and document processing
- **ChromaDB**: Vector database for storing document embeddings locally
- **LlamaCpp-Python**: For running quantized models efficiently
- **HuggingFace Transformers**: For loading and running various LLM formats

### Model Configuration
All model settings are in `constants.py`:
- **LLM Models**: Configure via `MODEL_ID` and `MODEL_BASENAME`
- **Embedding Models**: Configure via `EMBEDDING_MODEL_NAME`
- **Hardware Settings**: `N_GPU_LAYERS`, `N_BATCH`, `CONTEXT_WINDOW_SIZE`
- **Default LLM**: Meta-Llama-3-8B-Instruct
- **Default Embeddings**: hkunlp/instructor-large

### Document Processing Flow
1. Documents placed in `SOURCE_DOCUMENTS/` folder
2. `ingest.py` loads documents using format-specific loaders (defined in `DOCUMENT_MAP`)
3. Documents are chunked and embedded using the configured embedding model
4. Embeddings stored in ChromaDB at `DB/` directory
5. Query time: Similar chunks retrieved and used as context for LLM response

### Supported Formats
PDF, TXT, MD, PY, CSV, XLS/XLSX, DOC/DOCX, HTML

### Hardware Support
- NVIDIA GPUs (CUDA)
- Apple Silicon (Metal/MPS)
- CPU-only mode
- Intel Gaudi HPU

### Important Directories
- `SOURCE_DOCUMENTS/`: Input documents location
- `DB/`: ChromaDB vector store
- `models/`: Downloaded model files
- `localGPTUI/`: Flask-based web interface
- `gaudi_utils/`: Intel HPU support utilities