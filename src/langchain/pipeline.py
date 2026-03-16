import typing as tp
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from src.schemas.server import ServerConfig
from src.langchain.router import SemanticRouter
from src.langchain.rewriter import QueryRewriter
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
        router: SemanticRouter,
        rewriter: QueryRewriter,
        retriever: RAGRetriever,
        generator_llm: ChatOpenAI
    ) -> None:
        """하위 컴포넌트를 주입받아 파이프라인을 구축한다."""
        self.config: ServerConfig = config
        self.router: SemanticRouter = router
        self.rewriter: QueryRewriter = rewriter
        self.retriever: RAGRetriever = retriever
        self.generator_llm: ChatOpenAI = generator_llm
        
        str_systemPrompt: str = self.config.prompts.get("rag_answer_system", "")
        self.rag_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(list([
            ("system", str_systemPrompt),
            ("user", "### Context:\n{context}\n\n### Question:\n{question}\n\n### Answer:")
        ]))
        logger.info("RAGPipeline initialized.")

    def _format_context(self, list_docs: tp.List[RetrievedDocument]) -> str:
        """검색된 문서를 텍스트 컨텍스트로 변환한다."""
        list_chunks: tp.List[str] = list()
        for int_i, obj_doc in enumerate(list_docs):
            list_chunks.append(f"[Document {int_i + 1}]\n{obj_doc.str_content}")
        return "\n\n".join(list_chunks)

    async def run(self, str_query: str) -> tp.AsyncGenerator[tp.Any, None]:
        """RAG 워크플로우를 비동기적으로 실행한다."""
        str_intent: str = await self.router.aroute_query(str_query=str_query)
        
        if str_intent == "GENERAL_CONVERSATION":
            async for obj_chunk in self.generator_llm.astream(
                input=f"사용자의 질문에 한국어로 친절하게 답해주세요: {str_query}"
            ):
                yield obj_chunk
            return

        if str_intent == "AMBIGUOUS":
            yield "질문이 너무 짧거나 모호합니다. 내용을 구체적으로 말씀해 주세요."
            return
        
        # RETRIEVAL_REQUIRED (RAG 경로)
        str_cleanQuery: str = str_query.strip().lstrip("#")
        str_refinedQuery: str = await self.rewriter.arewrite(str_query=str_cleanQuery)
        
        list_docs: tp.List[RetrievedDocument] = await self.retriever.aretrieve(str_query=str_refinedQuery)
        if not list_docs:
            yield "죄송합니다. 관련 정보를 찾지 못했습니다."
            return
        
        str_context: str = self._format_context(list_docs=list_docs)
        async for obj_chunk in self.generator_llm.astream(
            input=self.rag_prompt.format_messages(
                context=str_context, 
                question=str_refinedQuery
            )
        ):
            yield obj_chunk
