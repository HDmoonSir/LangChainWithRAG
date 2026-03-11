import typing as tp
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from loguru import logger

from src.rag.pipeline import RAGPipeline

app = FastAPI(title="High-Performance Multi-LLM RAG Engine")

# RAG 파이프라인 싱글톤 초기화
pipeline = RAGPipeline()

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/chat/stream")
async def chat_stream(request: QueryRequest):
    """
    RAG 파이프라인을 통한 스트리밍 채팅 엔드포인트
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    logger.info(f"API Request received: {request.query}")
    
    def generate():
        try:
            for chunk in pipeline.run(request.query):
                # ChatOpenAI의 chunk에서 내용 추출
                content = getattr(chunk, 'content', str(chunk))
                if content:
                    yield content
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            yield f"\n[Error]: {str(e)}"

    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
