import asyncio
import typing as tp
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import CrossEncoder
from loguru import logger
from src.schemas.server import ServerConfig
from src.schemas.rag import RetrievedDocument

class RAGRetriever:
    """
    벡터 DB 검색 및 리랭킹을 수행한다.
    가변 인수 사용을 금지하며 명시적 타입을 사용한다.
    """
    def __init__(
        self, 
        config: ServerConfig, 
        vector_store: QdrantVectorStore, 
        reranker: CrossEncoder
    ) -> None:
        """의존성을 주입받아 초기화한다."""
        self.config: ServerConfig = config
        self.vector_store: QdrantVectorStore = vector_store
        self.reranker: CrossEncoder = reranker
        logger.info("RAGRetriever initialized.")

    async def aretrieve(self, str_query: str, int_topK: int = 5) -> tp.List[RetrievedDocument]:
        """비동기 벡터 검색 및 리랭킹을 수행한다."""
        # 1. 벡터 유사도 검색
        list_initialDocs: tp.List[tp.Any] = await self.vector_store.asimilarity_search(
            query=str_query, 
            k=int_topK * 2
        )
        
        if not list_initialDocs:
            return list()

        # 2. 리랭킹 데이터 구성
        list_pairs: tp.List[tp.List[str]] = list()
        for obj_doc in list_initialDocs:
            list_pairs.append(list([str_query, obj_doc.page_content]))
            
        # 3. 리랭커 실행 (별도 스레드)
        list_scores: tp.List[float] = await asyncio.to_thread(
            self.reranker.predict, 
            sentences=list_pairs
        )
        
        # 4. 결과 정렬 및 데이터 모델 변환
        list_scoredDocs: tp.List[RetrievedDocument] = list()
        for int_i, float_score in enumerate(list_scores):
            if float_score < -10.0: 
                continue
            list_scoredDocs.append(RetrievedDocument(
                str_content=list_initialDocs[int_i].page_content,
                dict_metadata=dict(list_initialDocs[int_i].metadata),
                float_score=float(float_score)
            ))
            
        list_scoredDocs.sort(key=lambda x: x.float_score, reverse=True)
        return list_scoredDocs[:int_topK]
