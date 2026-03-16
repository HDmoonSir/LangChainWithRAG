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
from src.schemas.server import ServerConfig
from src.langchain.router import SemanticRouter
from src.langchain.rewriter import QueryRewriter
from src.langchain.retriever import RAGRetriever
from src.langchain.pipeline import RAGPipeline
from src.app.routes import obj_chatRouter

class AppFactory:
    """
    FastAPI 애플리케이션의 구성 및 생명주기를 관리한다.
    lifespan 내에서 모든 무거운 객체를 1회 생성하여 주입한다.
    """
    @staticmethod
    def create_app(config_path: str, env_path: tp.Optional[str] = None) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            logger.info(f"--- Application Startup: Loading config from {config_path} ---")
            
            # 1. 설정 로드
            loader = ConfigLoaderService()
            config: ServerConfig = loader.build_server_config(yaml_path=config_path, env_path=env_path)
            
            # 2. 로컬 모델 초기화 (임베딩, 리랭커)
            device = config.vector_db.local_model_device
            embeddings = HuggingFaceEmbeddings(
                model_name=config.vector_db.embedding_model,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True}
            )
            reranker = CrossEncoder(config.vector_db.reranker_model, device=device)
            
            # 3. 벡터 스토어 클라이언트 초기화
            q_client = QdrantClient(url=str(config.vector_db.qdrant_url))
            v_store = QdrantVectorStore(
                client=q_client,
                collection_name=config.vector_db.collection_name,
                embedding=embeddings
            )
            
            # 4. LLM 클라이언트 초기화 (vLLM)
            main_llm_cfg = config.llm_servers["main"]
            main_llm = ChatOpenAI(
                model=main_llm_cfg.model_name,
                openai_api_base=str(main_llm_cfg.url),
                openai_api_key="none",
                temperature=main_llm_cfg.temperature,
                streaming=True
            )
            
            # 5. 하위 컴포넌트 조립 및 의존성 주입
            router = SemanticRouter(config=config)
            rewriter = QueryRewriter(config=config)
            retriever = RAGRetriever(
                config=config,
                vector_store=v_store,
                reranker=reranker
            )
            
            # 6. 최종 RAG 파이프라인 생성 및 app.state 저장
            app.state.pipeline = RAGPipeline(
                config=config,
                router=router,
                rewriter=rewriter,
                retriever=retriever,
                generator_llm=main_llm
            )
            
            logger.info("RAG Pipeline and dependencies successfully initialized in lifespan.")
            yield
            logger.info("--- Application Shutdown: Cleaning up resources ---")

        app = FastAPI(
            title="Production-Ready RAG Engine",
            version="2.0.0",
            lifespan=lifespan
        )
        app.include_router(obj_chatRouter)
        return app
