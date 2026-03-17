import asyncio
import typing as tp
from loguru import logger
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import CrossEncoder

from src.schemas.server import ServerConfig
from src.schemas.rag import RetrievedDocument

class RAGRetriever:
    """
    인프라(벡터스토어, 임베딩, 리랭커)를 캡슐화하여 검색을 수행한다.
    내부에서 필요한 모델들을 설정에 맞게 직접 초기화한다.
    """
    def __init__(self, config: ServerConfig) -> None:
        """설정을 기반으로 검색 인프라를 내부적으로 구축한다."""
        self.config: ServerConfig = config
        obj_vdbCfg = config.vector_db
        str_device: str = obj_vdbCfg.local_model_device

        # 1. 임베딩 모델 초기화
        self.embeddings: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
            model_name=obj_vdbCfg.embedding_model,
            model_kwargs=dict(device=str_device),
            encode_kwargs=dict(normalize_embeddings=True)
        )

        # 2. 리랭커 모델 초기화
        self.reranker: CrossEncoder = CrossEncoder(
            model_name=obj_vdbCfg.reranker_model, 
            device=str_device
        )

        # 3. 벡터 스토어 연결
        self.q_client: QdrantClient = QdrantClient(url=str(obj_vdbCfg.qdrant_url))
        self.vector_store: QdrantVectorStore = QdrantVectorStore(
            client=self.q_client,
            collection_name=obj_vdbCfg.collection_name,
            embedding=self.embeddings
        )

        # 4. 컬렉션 차원 정합성 검증 (실시간 정합성 체크)
        try:
            obj_info = self.q_client.get_collection(collection_name=obj_vdbCfg.collection_name)
            int_existingDim: int = obj_info.config.params.vectors.size
            list_probe = self.embeddings.embed_query("test")
            int_modelDim: int = len(list_probe)

            if int_existingDim != int_modelDim:
                str_err: str = (
                    f"Dimension mismatch! DB Collection: {int_existingDim}, "
                    f"Current Model: {int_modelDim}. Please check your embedding model."
                )
                logger.error(str_err)
                # 서버 기동 중단이 아닌 경고 후 진행하거나 필요 시 raise 가능
            else:
                logger.info(f"Verified collection dimension: {int_existingDim}")
                
        except Exception as obj_err:
            logger.warning(f"Could not verify collection dimension: {obj_err}")

        logger.info("RAGRetriever infrastructure (Embeddings/Reranker/Qdrant) initialized.")

    async def aretrieve(self, str_query: str, int_topK: int = 5) -> tp.List[RetrievedDocument]:
        """비동기 벡터 검색 및 리랭킹을 수행한다."""
        list_initialDocs: tp.List[tp.Any] = await self.vector_store.asimilarity_search(
            query=str_query, 
            k=int_topK * 2
        )
        if not list_initialDocs: return list()

        list_pairs: tp.List[tp.List[str]] = [[str_query, d.page_content] for d in list_initialDocs]
        list_scores: tp.List[float] = await asyncio.to_thread(self.reranker.predict, sentences=list_pairs)
        
        list_scoredDocs: tp.List[RetrievedDocument] = list()
        for int_i, float_score in enumerate(list_scores):
            if float_score < -10.0: continue
            list_scoredDocs.append(RetrievedDocument(
                str_content=list_initialDocs[int_i].page_content,
                dict_metadata=dict(list_initialDocs[int_i].metadata),
                float_score=float(float_score)
            ))
        list_scoredDocs.sort(key=lambda x: x.float_score, reverse=True)
        return list_scoredDocs[:int_topK]
