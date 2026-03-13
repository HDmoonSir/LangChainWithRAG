import torch
import typing as tp
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from loguru import logger
from src.config.schemas import AppConfig, VectorDBConfig

class RAGEmbedder:
    """
    텍스트를 벡터로 변환하고 벡터 데이터베이스(Qdrant)에 저장하는 책임을 가진다.
    """
    def __init__(self, obj_config: AppConfig) -> None:
        """
        주입받은 설정을 통해 임베딩 모델과 벡터 DB 클라이언트를 초기화한다.
        실제 환경의 GPU 가용성을 체크하여 런타임 오류를 방지한다.
        """
        self.obj_config: AppConfig = obj_config
        obj_vdbCfg: VectorDBConfig = obj_config.obj_vectorDb
        
        # 1. 실행 장치 결정 (Dynamic Device Selection)
        str_requestedDevice: str = obj_vdbCfg.str_localModelRuntimeDevice
        str_finalDevice: str = "cpu"
        
        if "cuda" in str_requestedDevice.lower():
            if torch.cuda.is_available():
                str_finalDevice = str_requestedDevice
                logger.info(f"Using requested GPU device: {str_finalDevice}")
            else:
                logger.warning(f"CUDA requested ({str_requestedDevice}) but not available. Falling back to 'cpu'.")
                str_finalDevice = "cpu"
        else:
            str_finalDevice = "cpu"
            logger.info("Using 'cpu' device as requested.")

        # 2. 임베딩 모델 초기화
        self.obj_embeddingModel: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
            model_name=obj_vdbCfg.str_embeddingModel,
            model_kwargs=dict(device=str_finalDevice),
            encode_kwargs=dict(normalize_embeddings=True)
        )
        
        # 3. Qdrant 클라이언트 초기화
        self.obj_client: QdrantClient = QdrantClient(url=str(obj_vdbCfg.obj_qdrantUrl))
        self.str_collectionName: str = obj_vdbCfg.str_collectionName
        
        # 4. 컬렉션 정합성 확인 및 생성 (차원 자동 감지)
        self._ensure_collection_integrity()
        
        # 5. Vector Store 래퍼 초기화
        self.obj_vectorStore: QdrantVectorStore = QdrantVectorStore(
            client=self.obj_client,
            collection_name=self.str_collectionName,
            embedding=self.obj_embeddingModel
        )
        logger.info(f"RAGEmbedder initialized. Collection: {self.str_collectionName}, Final Device: {str_finalDevice}")

    def _ensure_collection_integrity(self) -> None:
        """
        임베딩 모델의 실제 차원을 확인하고 Qdrant 컬렉션과의 일치 여부를 검증한다.
        """
        list_probeVec: tp.List[float] = self.obj_embeddingModel.embed_query("dimension_check")
        int_modelDim: int = len(list_probeVec)
        
        list_collections: tp.List[tp.Any] = self.obj_client.get_collections().collections
        bool_exists: bool = any(obj_c.name == self.str_collectionName for obj_c in list_collections)
        
        if not bool_exists:
            logger.info(f"Creating new collection '{self.str_collectionName}' (Dim: {int_modelDim})")
            self.obj_client.create_collection(
                collection_name=self.str_collectionName,
                vectors_config=models.VectorParams(
                    size=int_modelDim,
                    distance=models.Distance.COSINE
                )
            )
            return

        obj_info: tp.Any = self.obj_client.get_collection(self.str_collectionName)
        int_existingDim: int = tp.cast(int, obj_info.config.params.vectors.size)
        
        if int_existingDim != int_modelDim:
            raise ValueError(
                f"Collection dimension mismatch! Existing: {int_existingDim}, Model: {int_modelDim}."
            )

    def upsert_documents(
        self, 
        list_texts: tp.List[str], 
        list_metadatas: tp.List[tp.Dict[str, tp.Any]], 
        list_ids: tp.List[str]
    ) -> None:
        """
        결정적 ID를 사용하여 텍스트와 메타데이터를 저장하거나 업데이트(Upsert)한다.
        """
        if not list_texts:
            logger.warning("No texts provided for upsert.")
            return

        if not (len(list_texts) == len(list_metadatas) == len(list_ids)):
            raise ValueError("Mismatched lengths of texts, metadatas, or ids.")

        logger.info(f"Upserting {len(list_texts)} chunks into collection '{self.str_collectionName}'")
        self.obj_vectorStore.add_texts(
            texts=list_texts,
            metadatas=list_metadatas,
            ids=list_ids
        )
        logger.success(f"Successfully upserted {len(list_texts)} chunks.")
