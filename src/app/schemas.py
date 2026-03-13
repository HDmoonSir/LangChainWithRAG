import typing as tp
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """
    사용자 채팅 질의에 대한 요청 데이터 모델이다.
    """
    str_query: str = Field(..., description="사용자 질의 텍스트", alias="query")

    class Config:
        populate_by_name = True

class SSEFrame(BaseModel):
    """
    표준화된 SSE(Server-Sent Events) 응답 프레임 정의이다.
    모든 데이터는 JSON 포맷으로 전달되어 클라이언트 파싱을 용이하게 한다.
    """
    str_event: str = Field(..., description="이벤트 유형 (token, error, metadata, finish)")
    dict_data: tp.Dict[str, tp.Any] = Field(default_factory=dict, description="실제 전달 데이터")
