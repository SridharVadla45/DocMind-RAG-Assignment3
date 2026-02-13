# DocMind-RAG

A simple Retrieval-Augmented Generation (RAG) system built using LangChain, ChromaDB, and Jina AI embeddings.

## Setup

1.  **Install dependencies**:
    ```bash
    uv sync
    ```

2.  **Environment Variables**:
    - Copy `.env.example` to `.env`.
    - Add your `GOOGLE_API_KEY`.

3.  **Add Documents**:
    - Place your PDF files in the `data/` directory.

4.  **Run the RAG System**:
    ```bash
    uv run rag_system.py
    ```

## Features
- **Document Loading**: Automatically loads all PDFs from the `data/` folder.
- **Text Splitting**: Uses `RecursiveCharacterTextSplitter` (Size: 1000, Overlap: 200).
- **Vector Store**: Stores embeddings in a local ChromaDB.
- **Embeddings**: Uses Jina AI's local model as a free alternative.
- **Q&A**: Interactive command-line interface for asking questions.
- **Test Results**: Automatically runs predefined queries and saves them to `output/results.txt`.
