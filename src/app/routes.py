import typing as tp
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from src.rag.pipeline import RAGPipeline
from src.app.schemas import ChatRequest
from src.app.streaming import ChatStreamGenerator

# 라우터 생성 (가이드라인 네이밍 규칙 준수)
obj_chatRouter: APIRouter = APIRouter(tags=["chat"])

def get_pipeline(obj_request: Request) -> RAGPipeline:
    """
    FastAPI app.state에서 RAG 파이프라인 인스턴스를 추출하여 반환한다.
    """
    return tp.cast(RAGPipeline, obj_request.app.state.pipeline)

@obj_chatRouter.post("/chat/stream")
async def chat_stream(
    obj_request: ChatRequest,
    obj_pipeline: RAGPipeline = Depends(get_pipeline)
) -> StreamingResponse:
    """
    사용자의 질의를 받아 RAG 파이프라인을 실행하고 표준 SSE 스트림을 응답한다.
    """
    if not obj_request.str_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info(f"Received chat stream request: {obj_request.str_query}")
    
    # 1. 파이프라인 실행 (AsyncGenerator 반환)
    obj_pipelineRun: tp.AsyncGenerator[tp.Any, None] = obj_pipeline.run(str_query=obj_request.str_query)
    
    # 2. SSE 변환기 초기화
    obj_streamGenerator: ChatStreamGenerator = ChatStreamGenerator(obj_pipelineRun=obj_pipelineRun)
    
    # 3. 스트리밍 응답 반환
    return StreamingResponse(
        obj_streamGenerator.generate(),
        media_type="text/event-stream"
    )

@obj_chatRouter.get("/health")
async def health_check() -> tp.Dict[str, str]:
    """
    서비스 상태를 확인하기 위한 헬스체크 엔드포인트이다.
    """
    return dict(status="healthy")
