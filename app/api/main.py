from fastapi import FastAPI, Query, HTTPException
from app.rag.engine import RAGEngine

app = FastAPI(
    title="AI Resume Analyzer API",
    description="RAG-powered question answering using PDF knowledge base",
    version="1.0.0"
)

# Initialize RAG Engine once (singleton-style)
rag_engine = RAGEngine()


@app.get("/query")
def query(question: str = Query(..., description="User question")):
    """
    Return an LLM-generated answer grounded using the PDF content.
    """
    try:
        answer = rag_engine.generate_answer(question)

        return {
            "question": question,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )