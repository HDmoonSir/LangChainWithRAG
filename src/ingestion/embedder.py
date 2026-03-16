import torch
import typing as tp
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from loguru import logger
from src.schemas.ingest import IngestConfig
from src.schemas.common import VectorDBConfig

class RAGEmbedder:
    """
    텍스트를 벡터로 변환하여 Qdrant에 저장한다.
    가변 인수 사용을 금지하며 명시적 타입을 사용한다.
    """
    def __init__(self, config: IngestConfig) -> None:
        """설정을 기반으로 임베딩 모델 및 벡터 DB를 초기화한다."""
        self.config: IngestConfig = config
        obj_vdbCfg: VectorDBConfig = config.vector_db
        
        str_device: str = obj_vdbCfg.local_model_device
        self.embeddings: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
            model_name=obj_vdbCfg.embedding_model,
            model_kwargs=dict(device=str_device),
            encode_kwargs=dict(normalize_embeddings=True)
        )
        
        self.client: QdrantClient = QdrantClient(url=str(obj_vdbCfg.qdrant_url))
        self.collection_name: str = obj_vdbCfg.collection_name
        
        self._ensure_collection()
        
        self.vector_store: QdrantVectorStore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )

    def _ensure_collection(self) -> None:
        """컬렉션 존재 여부를 확인하고 없으면 생성한다."""
        list_probe: tp.List[float] = self.embeddings.embed_query(text="test")
        int_dim: int = len(list_probe)
        
        list_cols: tp.List[tp.Any] = self.client.get_collections().collections
        if not any(obj_c.name == self.collection_name for obj_c in list_cols):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=int_dim, 
                    distance=models.Distance.COSINE
                )
            )

    def upsert_documents(
        self, 
        list_texts: tp.List[str], 
        list_metadatas: tp.List[tp.Dict[str, tp.Any]], 
        list_ids: tp.List[str]
    ) -> None:
        """텍스트와 메타데이터를 벡터 DB에 저장한다."""
        self.vector_store.add_texts(
            texts=list_texts, 
            metadatas=list_metadatas, 
            ids=list_ids
        )
