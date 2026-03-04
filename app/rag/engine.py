from app.core.config import (
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    TOP_K
)

from app.rag.loader import PDFLoader
from app.rag.chunker import LangchainTextChunker
from app.rag.embeddings import EmbeddingModel
from app.rag.vectorstore import VectorStore

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv


class RAGEngine:
    """
    Singleton-style RAG Engine.
    Initialized once and serves all queries.
    """

    def __init__(self):
        self.vector_store = None
        self.llm = None
        self._initialize()

    def _initialize(self):
        load_dotenv()

        # 1️⃣ Load PDF
        text = PDFLoader(PDF_PATH).load()

        # 2️⃣ Chunk
        chunks = LangchainTextChunker(
            CHUNK_SIZE,
            CHUNK_OVERLAP
        ).chunk(text)

        # 3️⃣ Embeddings
        embeddings = EmbeddingModel(EMBEDDING_MODEL_NAME)

        # 4️⃣ Vector Store
        self.vector_store = VectorStore(embeddings)
        self.vector_store.build(chunks)

        # 5️⃣ LLM
        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile"
        )

    def generate_answer(self, question: str) -> str:
        """
        Generate an answer using the vector store with a grounded prompt.
        """

        # Retrieve top-k relevant chunks
        contexts = self.vector_store.search(
            query=question,
            k=TOP_K
        )

        combined_text = "\n\n".join(contexts)

        prompt_template = """
You are a helpful assistant.

Use ONLY the information provided in the context below to answer the question.
If the answer is not present in the context, respond with: "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )

        final_prompt = prompt.format(
            context=combined_text,
            question=question
        )

        response = self.llm.invoke(final_prompt)

        return response.content