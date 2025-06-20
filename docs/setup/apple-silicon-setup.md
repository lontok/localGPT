# LocalGPT Setup Guide for Apple Silicon Macs

This guide walks through setting up LocalGPT on Apple Silicon Macs (M1/M2/M3) with Metal acceleration for educational purposes. You'll learn how Retrieval-Augmented Generation (RAG) works by building a private chatbot that can answer questions about your documents.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10 or later
- Git (for cloning the repository)
- At least 16GB RAM recommended
- ~10GB free disk space for models

## Educational Overview

LocalGPT demonstrates key concepts in modern AI:
- **RAG (Retrieval-Augmented Generation)**: Combining document search with LLM responses
- **Vector Embeddings**: Converting text into searchable mathematical representations
- **Local LLM Inference**: Running large language models on personal hardware
- **Privacy-First AI**: All processing happens locally, no data leaves your machine

## Setup Steps

### 1. Create Virtual Environment - Isolate Project Dependencies

**Why**: Prevents conflicts between project libraries and system Python packages, ensuring reproducible environments.

```bash
python3 -m venv localGPT_env
source localGPT_env/bin/activate
```

### 2. Install Dependencies - Get Required Libraries

**Why**: Installs LangChain for orchestration, ChromaDB for vector storage, Streamlit for UI, and other essential components of the RAG pipeline.

```bash
pip install -r requirements.txt
```

### 3. Install llama-cpp-python with Metal Support - Enable GPU Acceleration

**Why**: Metal support allows the LLM to use your Mac's GPU, making inference 5-10x faster than CPU-only processing.

```bash
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
```

### 4. Verify Model Configuration - Ensure Optimal Model Selection

**Why**: The pre-configured Llama-3 model balances quality and performance for Apple Silicon, providing good results without overwhelming your hardware.

The `constants.py` file is already configured with:
```python
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_BASENAME = None
```

### 5. Add Your Documents - Create Knowledge Base

**Why**: Documents in this folder become the chatbot's knowledge source. The system can only answer questions based on information from these documents.

Place your documents in the `SOURCE_DOCUMENTS/` folder:
- Supported formats: PDF, TXT, MD, PY, CSV, XLS/XLSX, DOC/DOCX, HTML
- A sample PDF (`Orca_paper.pdf`) is included for testing
- Try adding your own study materials, papers, or notes!

### 6. Create Vector Embeddings - Make Documents Searchable

**Why**: This converts your documents into high-dimensional vectors that enable semantic search - finding relevant passages based on meaning, not just keywords.

```bash
python ingest.py --device_type mps
```

**What's happening**: 
- Documents are split into chunks
- Each chunk is converted to a 768-dimensional vector
- Vectors are stored in ChromaDB for fast similarity search
- First run downloads the embedding model (~1.5GB)

### 7. Launch the Streamlit UI - Interactive Chat Interface

**Why**: Provides a user-friendly way to interact with your documents, similar to ChatGPT but running entirely on your machine.

```bash
python localGPT_UI.py
```

The UI will open at `http://localhost:8501`

**Note**: First run downloads the Llama-3 model (~5GB). After that, everything runs locally.

### 8. Test Your Setup - Validate the RAG Pipeline

**Why**: Confirms that document retrieval, context injection, and LLM generation are working together correctly.

1. Open the Streamlit UI in your browser
2. Ask questions about your documents
3. Enable "Show Sources" to see which document chunks were retrieved
4. Observe how the LLM uses retrieved context to answer questions

## Alternative UIs

### Flask Web UI
```bash
# Terminal 1: Start API
python run_localGPT_API.py

# Terminal 2: Start Flask UI
cd localGPTUI && python localGPTUI.py
```
Access at `http://localhost:5111`

### Command Line Interface
```bash
python run_localGPT.py --device_type mps
```

## Educational Insights

### Understanding the RAG Pipeline

1. **Query Processing**: Your question is converted to a vector embedding
2. **Retrieval**: ChromaDB finds the most similar document chunks
3. **Context Formation**: Retrieved chunks become context for the LLM
4. **Generation**: The LLM generates an answer using the retrieved context
5. **Response**: You get an answer grounded in your documents

### Learning Opportunities

- **Experiment with chunk sizes** in the ingestion process
- **Compare different embedding models** to see retrieval quality changes
- **Try various LLMs** to understand model capabilities
- **Analyze retrieved sources** to understand semantic search

## Troubleshooting

### Low Memory Issues
If you encounter memory issues, try:
1. Using a smaller model (edit `constants.py` to use a GGUF quantized model)
2. Reducing `N_GPU_LAYERS` in `constants.py`
3. Closing other applications

### Performance Tips
- Metal acceleration should be automatically detected
- Monitor Activity Monitor to ensure GPU is being used
- First queries are slower due to model loading
- Use quantized models (GGUF format) for better memory efficiency

## Next Steps for Learning

- **Document Variety**: Add different types of documents to see how retrieval performs
- **Model Comparison**: Try different models in `constants.py` to understand trade-offs
- **Embedding Analysis**: Use `--show_sources` to study which chunks get retrieved
- **Build Applications**: Use the API to create custom RAG applications
- **Fine-tuning**: Experiment with different chunking strategies and retrieval parameters