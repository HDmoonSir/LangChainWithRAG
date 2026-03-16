import typing as tp
from pydantic import BaseModel, AnyUrl, ConfigDict

class LLMServerConfig(BaseModel):
    """vLLM 등 개별 LLM 서버 설정"""
    model_name: str
    temperature: float
    url: AnyUrl
    
    model_config = ConfigDict(extra="forbid")

class VectorDBConfig(BaseModel):
    """벡터 DB 및 로컬 추론 모델 설정"""
    collection_name: str
    embedding_model: str
    reranker_model: str
    qdrant_url: AnyUrl
    local_model_device: str # 'cpu' 또는 'cuda:0' 등 필수 입력
    
    model_config = ConfigDict(extra="forbid")
