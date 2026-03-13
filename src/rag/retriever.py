import asyncio
import typing as tp
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import CrossEncoder
from loguru import logger
from src.config.schemas import AppConfig
from src.rag.schemas import RetrievedDocument

class RAGRetriever:
    """
    벡터 데이터베이스에서 관련 문서를 검색하고 리랭킹을 통해 순위를 재조정하는 책임을 가진다.
    의존성 주입과 비동기 처리를 명시적으로 수행한다.
    """
    def __init__(
        self, 
        obj_config: AppConfig, 
        obj_vectorStore: QdrantVectorStore, 
        obj_reranker: CrossEncoder
    ) -> None:
        """
        주입받은 설정을 통해 검색 엔진의 구성 요소를 초기화한다.
        """
        self.obj_config: AppConfig = obj_config
        self.obj_vectorStore: QdrantVectorStore = obj_vectorStore
        self.obj_reranker: CrossEncoder = obj_reranker
        
        logger.info(f"RAGRetriever initialized (Dependency Injected). Collection: {obj_config.obj_vectorDb.str_collectionName}")

    async def aretrieve(self, str_query: str, int_topK: int = 5) -> tp.List[RetrievedDocument]:
        """
        질의에 대해 1차 벡터 검색 및 리랭킹을 비동기적으로 수행한다.
        CPU 집약적인 리랭킹 작업은 별도의 스레드로 분리한다.
        """
        logger.info(f"Retrieving documents for query: {str_query}")
        
        # A. 1차 벡터 유사도 검색 (비동기 I/O 바운드)
        list_initialDocs: tp.List[tp.Any] = await self.obj_vectorStore.asimilarity_search(
            query=str_query, 
            k=int_topK * 2
        )
        
        if not list_initialDocs:
            logger.warning("No documents found in initial vector search.")
            return list()

        # B. 리랭킹을 위한 데이터 쌍 구성 사용)
        list_pairs: tp.List[tp.List[str]] = list()
        for obj_doc in list_initialDocs:
            list_pairs.append(list([str_query, obj_doc.page_content]))
            
        # C. 리랭커를 통한 정밀 점수 계산 (CPU 바운드 작업은 별도 스레드 분리)
        list_scores: tp.List[float] = await asyncio.to_thread(self.obj_reranker.predict, list_pairs)
        
        # D. 점수와 문서를 결합 및 필터링하여 명시적 데이터 계약 준수
        list_scoredDocs: tp.List[RetrievedDocument] = list()
        for int_i, float_score in enumerate(list_scores):
            if float_score < -10.0:
                continue
                
            obj_item: RetrievedDocument = RetrievedDocument(
                str_content=list_initialDocs[int_i].page_content,
                dict_metadata=dict(list_initialDocs[int_i].metadata),
                float_score=float_score
            )
            list_scoredDocs.append(obj_item)
            
        # E. 최종 정렬 및 결과 반환
        list_scoredDocs.sort(key=lambda x: x.float_score, reverse=True)
        list_finalResults: tp.List[RetrievedDocument] = list_scoredDocs[:int_topK]
        
        logger.info(f"Final retrieved documents: {len(list_finalResults)}")
        return list_finalResults
