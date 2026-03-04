from langchain_community.vectorstores import FAISS
from typing import List


class VectorStore:
    """
    FAISS-based vector store for document retrieval.
    """

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.store = None

    def build(self, texts: List[str]):
        """
        Build FAISS index from text chunks.
        """
        self.store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings.model
        )

    def search(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve top-k relevant chunks.
        """
        if self.store is None:
            raise ValueError("Vector store not initialized. Call build() first.")

        docs = self.store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]