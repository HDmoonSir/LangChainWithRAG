import os
import typing as tp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

class RAGEmbedder:
    def __init__(self, collection_name: str = "kb_documents"):
        self.model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = collection_name
        
        # BGE-M3 임베딩 모델 초기화
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},  # GPU 사용 시 'cuda'로 변경 권장
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Qdrant 클라이언트 초기화
        self.client = QdrantClient(url=self.qdrant_url)
        self._ensure_collection()
        
        # LangChain 연동 Vector Store (최신 사양에 맞춰 'embedding' 인자 사용)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model
        )
        logger.info(f"Embedder initialized with model: {self.model_name}")

    def _ensure_collection(self):
        """
        Qdrant 컬렉션이 없으면 생성합니다.
        BGE-M3의 차원은 1024입니다.
        """
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            logger.info(f"Creating Qdrant collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1024,  # BGE-M3 dense vector size
                    distance=models.Distance.COSINE
                )
            )

    def add_documents(self, texts: tp.List[str], metadatas: tp.List[tp.Dict[str, tp.Any]]):
        """
        텍스트와 메타데이터를 벡터화하여 저장합니다.
        """
        logger.info(f"Indexing {len(texts)} chunks into Qdrant...")
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)
        logger.success("Indexing complete.")

if __name__ == "__main__":
    # 간단한 연결 테스트
    embedder = RAGEmbedder()
    print("Qdrant & Embedding Setup Ready.")
