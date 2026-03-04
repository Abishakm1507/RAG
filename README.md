# AI Resume Analyzer (RAG Project)

This repository implements a **Retrieval-Augmented Generation (RAG)** service that
allows users to ask questions about the contents of a PDF knowledge base and
generate answers using a large language model. It’s been structured for an AI
championship project, but the core components are general-purpose and can be
re‑used for other document‑grounded question‑answering tasks.

---

## 🔍 Overview

The application provides a **FastAPI** server with a single endpoint (`/query`) that
accepts a question and returns an answer grounded in the text of a PDF stored in
`data/knowledge.pdf`. Behind the scenes the service:

1. Loads and extracts text from the PDF.
2. Splits text into overlapping chunks using LangChain’s splitter.
3. Embeds chunks using a SentenceTransformer model.
4. Stores embeddings in a FAISS vector index for fast similarity search.
5. Queries a Groq-powered LLM (`llama-3.3-70b-versatile`) with the retrieved
   context to generate an answer.

The engine is initialized once when the server starts (singleton pattern) and
serves all incoming requests.

---

## 🗂 Project Structure

```
RAG/
├─ app/
│  ├─ api/main.py           # FastAPI entrypoint
│  ├─ rag/                 # RAG pipeline modules
│  │  ├─ loader.py         # PDF loader
│  │  ├─ chunker.py        # Text chunker
│  │  ├─ embeddings.py     # Embedding wrapper
│  │  ├─ vectorstore.py    # FAISS-based retrieval
│  │  └─ engine.py         # Orchestrates RAG workflow
│  └─ core/config.py       # Configuration constants
├─ data/
│  └─ knowledge.pdf        # PDF knowledge base 
├─ docker/Dockerfile       # Container build definition
├─ requirements.txt        # Python dependencies
├─ .env                    # Environment variables 
├─ README.md               # Project documentation 
└─ myvenv/                 # Local virtual environment 
```

---

## 🧠 How the RAG Engine Works

- **PDFLoader** reads the PDF and concatenates text from each page.
- **LangchainTextChunker** splits the text into manageable chunks (default 500
  characters with 50 overlap) to ensure context for retrieval.
- **EmbeddingModel** uses HuggingFace embeddings to convert text and queries to
  vectors.
- **VectorStore** builds a FAISS index from the chunk embeddings and performs
  k‑nearest-neighbor searches.
- **RAGEngine.generate_answer** retrieves the top‑K chunks, constructs a prompt
  template, and invokes a Groq LLM to answer using only the retrieved context.


---

## 📦 Dependencies

List available in `requirements.txt`. Key libraries:

* fastapi, uvicorn – web server
* pypdf – PDF text extraction
* langchain, langchain-community – tooling for embeddings and chunking
* sentence-transformers – for default embeddings
* faiss-cpu – vector store
* langchain_huggingface, langchain-groq – connectors to external LLMs


---

## 🛠 Development Tips

* Add tests under `tests/` (not yet present) and run with `pytest`.
* To experiment with another LLM provider, modify `RAGEngine._initialize` and
  update dependencies accordingly.
* Tweak chunk size/overlap in config for different document types.
* Consider caching the FAISS index to disk for faster restarts.

---

## ✅ Next Steps / To Do

1. Add more robust error handling and logging.
2. Support multiple document formats (e.g. Word, text files).
3. Add authentication to the API if exposing publicly.
4. Write comprehensive tests for each component.

---

Thanks for using the RAG project! Contributions and improvements are welcome.

Feel free to adapt this README further as your project evolves.
