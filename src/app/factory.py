import torch
import typing as tp
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder

from src.config.loader import ConfigLoaderService
from src.config.schemas import AppConfig, LLMServerConfig, VectorDBConfig
from src.rag.router import SemanticRouter
from src.rag.rewriter import QueryRewriter
from src.rag.retriever import RAGRetriever
from src.rag.pipeline import RAGPipeline
from src.app.routes import obj_chatRouter

class AppFactory:
    """
    FastAPI 애플리케이션의 구성, 조립 및 생명주기를 관리하는 팩토리 클래스이다.
    모든 의존성을 여기서 생성하고 하위 계층으로 주입한다.
    """
    @staticmethod
    def create_app() -> FastAPI:
        """
        FastAPI 인스턴스를 생성하고 라우터 및 Lifespan을 구성한다.
        """
        obj_app: FastAPI = FastAPI(
            title="Production-Ready RAG Engine",
            description="SSE 기반 고성능 비동기 RAG 서비스",
            version="1.1.0",
            lifespan=AppFactory.lifespan
        )

        # 라우터 등록
        obj_app.include_router(obj_chatRouter)

        return obj_app

    @staticmethod
    @asynccontextmanager
    async def lifespan(obj_app: FastAPI) -> tp.AsyncGenerator[None, None]:
        """
        서버 시작 시 설정을 로드하고 모든 의존성을 조립한다 (Composition Root).
        실행 장치 가용성을 체크하여 안전한 폴백을 수행한다.
        """
        logger.info("--- Application Startup: Orchestrating Dependencies ---")
        
        try:
            # 1. 설정 로드
            obj_configLoader: ConfigLoaderService = ConfigLoaderService()
            obj_appConfig: AppConfig = obj_configLoader.build_app_config(
                str_yamlPath="config/settings.yaml",
                str_envPath=".env"
            )
            obj_vdbCfg: VectorDBConfig = obj_appConfig.obj_vectorDb
            
            # 2. 실행 장치 결정 (Dynamic Device Selection)
            str_requestedDevice: str = obj_vdbCfg.str_localModelRuntimeDevice
            str_finalDevice: str = "cpu"
            
            if "cuda" in str_requestedDevice.lower():
                if torch.cuda.is_available():
                    str_finalDevice = str_requestedDevice
                    logger.info(f"Using requested GPU for local models: {str_finalDevice}")
                else:
                    logger.warning(f"CUDA requested ({str_requestedDevice}) but not available. Falling back to 'cpu'.")
                    str_finalDevice = "cpu"
            else:
                str_finalDevice = "cpu"
                logger.info("Using 'cpu' device for local models as requested.")

            # 3. 로컬 임베딩 모델 및 벡터 스토어 클라이언트 생성
            obj_embeddingModel: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
                model_name=obj_vdbCfg.str_embeddingModel,
                model_kwargs=dict(device=str_finalDevice),
                encode_kwargs=dict(normalize_embeddings=True)
            )
            
            # LangChain QdrantVectorStore는 내부 비동기 처리를 위해 동기 클라이언트를 안정적으로 사용함
            obj_qdrantClient: QdrantClient = QdrantClient(url=str(obj_vdbCfg.obj_qdrantUrl))
            obj_vectorStore: QdrantVectorStore = QdrantVectorStore(
                client=obj_qdrantClient,
                collection_name=obj_vdbCfg.str_collectionName,
                embedding=obj_embeddingModel
            )
            
            # 4. 리랭커 모델 생성
            obj_reranker: CrossEncoder = CrossEncoder(
                obj_vdbCfg.str_rerankerModel, 
                device=str_finalDevice
            )
            
            # 5. 메인 답변 생성용 LLM 클라이언트 생성
            obj_mainLlmCfg: LLMServerConfig = obj_appConfig.dict_llmServers["main"]
            obj_generatorLlm: ChatOpenAI = ChatOpenAI(
                model=obj_mainLlmCfg.str_modelName,
                openai_api_key="none",
                openai_api_base=str(obj_mainLlmCfg.obj_url),
                temperature=obj_mainLlmCfg.float_temperature,
                streaming=True
            )
            
            # 6. 하위 서비스 객체 생성 및 의존성 주입
            obj_router: SemanticRouter = SemanticRouter(obj_config=obj_appConfig)
            obj_rewriter: QueryRewriter = QueryRewriter(obj_config=obj_appConfig)
            obj_retriever: RAGRetriever = RAGRetriever(
                obj_config=obj_appConfig,
                obj_vectorStore=obj_vectorStore,
                obj_reranker=obj_reranker
            )
            
            # 7. 최종 RAG 파이프라인 조립
            obj_app.state.pipeline = RAGPipeline(
                obj_config=obj_appConfig,
                obj_router=obj_router,
                obj_rewriter=obj_rewriter,
                obj_retriever=obj_retriever,
                obj_generatorLlm=obj_generatorLlm
            )
            
            logger.info("RAG Engine dependencies successfully orchestrated.")
            
        except Exception as obj_err:
            logger.error(f"Failed to orchestrate dependencies: {obj_err}")
            raise obj_err

        yield

        logger.info("--- Application Shutdown: Cleaning Up Resources ---")
