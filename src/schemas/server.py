import typing as tp
from pydantic import BaseModel, ConfigDict
from .common import LLMServerConfig, VectorDBConfig

class ServerConfig(BaseModel):
    """FastAPI 서버 전용 설정 모델"""
    llm_servers: tp.Dict[str, LLMServerConfig]
    vector_db: VectorDBConfig
    prompts: tp.Dict[str, str]
    
    model_config = ConfigDict(extra="forbid")
