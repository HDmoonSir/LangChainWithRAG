import typing as tp
from pydantic import BaseModel, ConfigDict
from .common import VectorDBConfig

class IngestParams(BaseModel):
    """인제스트 프로세스 파라미터"""
    chunk_size: int
    chunk_overlap: int
    input_glob: str
    
    model_config = ConfigDict(extra="forbid")

class IngestConfig(BaseModel):
    """인제스트 배치용 설정 모델"""
    vector_db: VectorDBConfig
    ingest: IngestParams
    
    model_config = ConfigDict(extra="forbid")
