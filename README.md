# RAG Project

A Retrieval-Augmented Generation (RAG) system built with LangChain and ChromaDB for intelligent document querying and question answering.

## Overview

This project implements a complete RAG pipeline that can:
- Load and process documents from web sources and PDFs
- Create vector embeddings using OpenAI's text-embedding-3-large model
- Store embeddings in ChromaDB for efficient similarity search
- Answer questions using retrieved context with GPT-4o-mini

## Project Structure

```
rag_project/
├── main.py           # Main RAG pipeline implementation
├── pdf_utils.py      # PDF processing utilities
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/craig-stevenson/rag-simple.git
cd rag_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file with your OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Basic RAG Pipeline

Run the main script to create a vector database and ask questions:

```bash
python main.py
```

### PDF Processing

Use the PDF utilities to extract and process PDF documents:

```python
from pdf_utils import pdf_to_chunks, pdf_to_markdown

# Convert PDF to text chunks
chunks = pdf_to_chunks("path/to/your/document.pdf")

# Convert PDF to markdown
markdown_path = pdf_to_markdown("path/to/your/document.pdf")
```

### Models Used

- **Embeddings**: OpenAI text-embedding-3-large
- **Language Model**: GPT-4o-mini
- **Vector Database**: ChromaDB

## API Requirements

This project requires an OpenAI API key for:
- Text embeddings generation
- Language model inference

## Database

The project uses ChromaDB for vector storage with persistent storage in the `./chroma` directory. The database is created once and can be reused for subsequent queries.

## Contact
craig.stevenson@gmail.com

---

*This README will be updated with additional details and examples.*
