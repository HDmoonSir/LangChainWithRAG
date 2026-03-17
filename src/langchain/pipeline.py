import typing as tp
from loguru import logger

from src.schemas.server import ServerConfig
from src.langchain.llm_router import RouterGenerator
from src.langchain.llm_rewriter import RewriterGenerator
from src.langchain.llm_main import MainGenerator
from src.langchain.llm_sub import SubGenerator
from src.langchain.retriever import RAGRetriever
from src.schemas.rag import RetrievedDocument

class RAGPipeline:
    """
    전체 RAG 프로세스를 오케스트레이션한다.
    가변 인수 사용을 금지하며 명시적 타입을 사용한다.
    """
    def __init__(
        self, 
        config: ServerConfig,
        router: RouterGenerator,
        rewriter: RewriterGenerator,
        retriever: RAGRetriever,
        main_gen: MainGenerator,
        sub_gen: SubGenerator
    ) -> None:
        """하위 컴포넌트를 주입받아 파이프라인을 구축한다."""
        self.config: ServerConfig = config
        self.router: RouterGenerator = router
        self.rewriter: RewriterGenerator = rewriter
        self.retriever: RAGRetriever = retriever
        self.main_gen: MainGenerator = main_gen
        self.sub_gen: SubGenerator = sub_gen
        logger.info("RAGPipeline successfully assembled with specialized components.")

    def _format_context(self, list_docs: tp.List[RetrievedDocument]) -> str:
        """검색된 문서를 텍스트 컨텍스트로 변환한다."""
        list_chunks: tp.List[str] = list()
        for int_i, obj_doc in enumerate(list_docs):
            list_chunks.append(f"[Document {int_i + 1}]\n{obj_doc.str_content}")
        return "\n\n".join(list_chunks)

    async def run(self, str_query: str) -> tp.AsyncGenerator[tp.Any, None]:
        """RAG 워크플로우를 비동기적으로 실행한다."""
        # 1. 의도 분류 (Router)
        str_intent: str = await self.router.aroute_query(str_query=str_query)
        
        # 2. 일상 대화 처리 (Sub LLM)
        if str_intent == "GENERAL_CONVERSATION":
            async for obj_chunk in self.sub_gen.astream_chat(str_query=str_query):
                yield obj_chunk
            return

        if str_intent == "AMBIGUOUS":
            yield "질문이 너무 짧거나 모호합니다. 내용을 구체적으로 말씀해 주세요."
            return
        
        # 3. RAG 프로세스 (Rewriter -> Retriever -> Main Generator)
        str_cleanQuery: str = str_query.strip().lstrip("#")
        str_refinedQuery: str = await self.rewriter.arewrite(str_query=str_cleanQuery)
        
        list_docs: tp.List[RetrievedDocument] = await self.retriever.aretrieve(str_query=str_refinedQuery)
        if not list_docs:
            yield "죄송합니다. 관련 정보를 찾지 못했습니다."
            return
        
        str_context: str = self._format_context(list_docs=list_docs)
        async for obj_chunk in self.main_gen.agenerate(str_context=str_context, str_question=str_refinedQuery):
            yield obj_chunk
