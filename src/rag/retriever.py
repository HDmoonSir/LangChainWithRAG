import typing as tp
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from loguru import logger
from src.utils.config_loader import config

class RAGRetriever:
    def __init__(self):
        vdb_cfg = config.get_vector_db_config()
        self.qdrant_url = vdb_cfg["qdrant_url"]
        self.embedding_model_name = vdb_cfg["embedding_model"]
        self.reranker_model_name = vdb_cfg["reranker_model"]
        self.collection_name = vdb_cfg["collection_name"]

        # 1. 임베딩 모델 로드 (BGE-M3)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 2. Qdrant 클라이언트 및 벡터 스토어 연결
        self.client = QdrantClient(url=self.qdrant_url, timeout=60)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )

        # 3. 리랭커 모델 로드
        try:
            self.reranker = CrossEncoder(self.reranker_model_name, device='cpu')
            logger.info(f"Reranker loaded: {self.reranker_model_name}")
        except Exception as e:
            logger.warning(f"Reranker loading failed: {e}")
            self.reranker = None

    def retrieve(self, query: str, top_k: int = 5, rerank_top_k: int = 15) -> tp.List[tp.Dict[str, tp.Any]]:
        candidates = self.vector_store.similarity_search_with_score(query, k=rerank_top_k)
        
        if not candidates or not self.reranker:
            return [{"content": d.page_content, "metadata": d.metadata} for d, _ in candidates[:top_k]]

        pairs = [[query, doc.page_content] for doc, _ in candidates]
        rerank_scores = self.reranker.predict(pairs)

        results = []
        for i, (doc, _) in enumerate(candidates):
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "rerank_score": float(rerank_scores[i])
            })
        
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results[:top_k]
