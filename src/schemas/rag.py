import typing as tp
from pydantic import BaseModel, Field

class RetrievedDocument(BaseModel):
    """
    검색 및 리랭킹을 거쳐 최종적으로 반환되는 문서의 데이터 계약이다.
    """
    str_content: str = Field(..., description="문서의 텍스트 내용")
    dict_metadata: tp.Dict[str, tp.Any] = Field(default_factory=dict, description="문서 관련 메타데이터")
    float_score: float = Field(..., description="리랭커에 의해 계산된 정밀 유사도 점수")
