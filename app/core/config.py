from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data directory
DATA_DIR = BASE_DIR / "data"

# PDF file path
PDF_PATH = DATA_DIR / "knowledge.pdf"

# Chunking configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval configuration
TOP_K = 3