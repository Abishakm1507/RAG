from langchain_huggingface import HuggingFaceEmbeddings
from typing import List


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer embeddings.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = HuggingFaceEmbeddings(
            model_name=model_name
        )

    def embed_documents(self, texts: List[str]):
        return self.model.embed_documents(texts)

    def embed_query(self, query: str):
        return self.model.embed_query(query)