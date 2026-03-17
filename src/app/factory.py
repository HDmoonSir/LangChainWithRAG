import typing as tp
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from src.config.loader import ConfigLoaderService
from src.schemas.server import ServerConfig
from src.langchain.llm_router import RouterGenerator
from src.langchain.llm_rewriter import RewriterGenerator
from src.langchain.llm_main import MainGenerator
from src.langchain.llm_sub import SubGenerator
from src.langchain.retriever import RAGRetriever
from src.langchain.pipeline import RAGPipeline
from src.app.routes import obj_chatRouter

class AppFactory:
    """
    FastAPI 애플리케이션의 구성 및 생명주기를 관리한다.
    lifespan 내에서 모든 컴포넌트를 조립하여 주입한다.
    """
    @staticmethod
    def create_app(config_path: str, env_path: tp.Optional[str] = None) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            logger.info(f"--- Application Startup: Loading config from {config_path} ---")
            
            # 1. 설정 로드
            loader = ConfigLoaderService()
            config: ServerConfig = loader.build_server_config(yaml_path=config_path, env_path=env_path)
            
            # 2. 각 책임을 가진 도메인 컴포넌트 생성
            # 각 클래스는 내부적으로 config를 참조하여 자신의 LLM/인프라를 초기화함
            router = RouterGenerator(config=config)
            rewriter = RewriterGenerator(config=config)
            retriever = RAGRetriever(config=config)
            main_gen = MainGenerator(config=config)
            sub_gen = SubGenerator(config=config)
            
            # 3. 최종 RAG 파이프라인 조립
            app.state.pipeline = RAGPipeline(
                config=config,
                router=router,
                rewriter=rewriter,
                retriever=retriever,
                main_gen=main_gen,
                sub_gen=sub_gen
            )
            
            logger.info("RAG Engine successfully initialized with Clean Architecture.")
            yield
            logger.info("--- Application Shutdown: Cleaning up resources ---")

        app = FastAPI(
            title="Production-Ready RAG Engine",
            version="2.0.0",
            lifespan=lifespan
        )
        app.include_router(obj_chatRouter)
        return app
