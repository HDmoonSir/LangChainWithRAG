import typing as tp
from pydantic import BaseModel, AnyUrl, Field, ConfigDict

class LLMServerConfig(BaseModel):
    """
    개별 LLM 서버의 설정을 정의하는 데이터 계약 모델이다.
    """
    str_modelName: str = Field(alias="model_name")
    float_temperature: float = Field(alias="temperature")
    obj_url: tp.Optional[AnyUrl] = Field(default=None, alias="url")

    model_config = ConfigDict(populate_by_name=True)

class VectorDBConfig(BaseModel):
    """
    벡터 데이터베이스 및 로컬 추론 모델(Embedding, Reranker) 설정을 정의한다.
    vLLM이 점유한 GPU(0, 1번 등)를 고려하여 런타임 장치를 명시적으로 제어한다.
    """
    str_collectionName: str = Field(alias="collection_name")
    str_embeddingModel: str = Field(alias="embedding_model")
    str_rerankerModel: str = Field(alias="reranker_model")
    obj_qdrantUrl: tp.Optional[AnyUrl] = Field(default=None, alias="qdrant_url")
    
    # "cpu", "cuda:0", "cuda:1" 등 구체적인 실행 장치를 나타낸다.
    str_localModelRuntimeDevice: str = Field(default="cpu", alias="local_model_device")

    model_config = ConfigDict(populate_by_name=True)

class AppConfig(BaseModel):
    """
    애플리케이션의 모든 설정을 포함하는 루트 데이터 계약 모델이다.
    """
    dict_llmServers: tp.Dict[str, LLMServerConfig] = Field(alias="llm_servers")
    obj_vectorDb: VectorDBConfig = Field(alias="vector_db")
    dict_prompts: tp.Dict[str, str] = Field(alias="prompts")

    model_config = ConfigDict(populate_by_name=True)
