import typing as tp
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from src.config.schemas import AppConfig
from src.rag.router import SemanticRouter
from src.rag.rewriter import QueryRewriter
from src.rag.retriever import RAGRetriever
from src.rag.schemas import RetrievedDocument

class RAGPipeline:
    """
    비동기 RAG 프로세스의 전체 흐름을 오케스트레이션하는 책임을 가진다.
    '#' 트리거를 통한 검색 여부 결정 및 질문 정돈(Refine)을 수행한다.
    """
    def __init__(
        self, 
        obj_config: AppConfig,
        obj_router: SemanticRouter,
        obj_rewriter: QueryRewriter,
        obj_retriever: RAGRetriever,
        obj_generatorLlm: ChatOpenAI
    ) -> None:
        """
        주입받은 하위 컴포넌트와 설정을 보관한다.
        """
        self.obj_config: AppConfig = obj_config
        self.obj_router: SemanticRouter = obj_router
        self.obj_rewriter: QueryRewriter = obj_rewriter
        self.obj_retriever: RAGRetriever = obj_retriever
        self.obj_generatorLlm: ChatOpenAI = obj_generatorLlm
        
        # 답변 생성을 위한 프롬프트 템플릿 구성
        str_systemPrompt: str = self.obj_config.dict_prompts.get("rag_answer_system", "")
        self.obj_ragPrompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(list([
            ("system", str_systemPrompt),
            ("user", "### Context:\n{context}\n\n### Question:\n{question}\n\n### Answer:")
        ]))
        
        logger.info("RAGPipeline initialized via explicit dependency injection.")

    def _format_context(self, list_docs: tp.List[RetrievedDocument]) -> str:
        """
        검색된 문서 목록을 LLM 주입을 위한 텍스트 컨텍스트로 변환한다.
        """
        list_formattedChunks: tp.List[str] = list()
        for int_i, obj_doc in enumerate(list_docs):
            list_formattedChunks.append(f"[Document {int_i + 1}]\n{obj_doc.str_content}")
            
        return "\n\n".join(list_formattedChunks)

    async def run(self, str_query: str) -> tp.AsyncGenerator[tp.Any, None]:
        """
        사용자 질의를 처리하는 전체 RAG 파이프라인 워크플로우를 비동기적으로 실행한다.
        1. '#' 트리거 확인
        2. RAG 필요 시 질문 정돈 및 검색
        3. 정돈된 질문과 컨텍스트로 답변 생성
        """
        logger.info(f"--- RAG Pipeline Execution Start: '{str_query}' ---")
        
        # 1. 트리거 기반 의도 분류 (비동기 호출)
        str_intent: str = await self.obj_router.aroute_query(str_query=str_query)
        
        # 2. 일반 대화 경로 (트리거 '#'가 없는 경우)
        if str_intent == "GENERAL_CONVERSATION":
            logger.info("Routing Path: General Conversation")
            async for obj_chunk in self.obj_generatorLlm.astream(f"사용자의 질문에 한국어로 친절하게 답해주세요: {str_query}"):
                yield obj_chunk
            return

        if str_intent == "AMBIGUOUS":
            logger.info("Routing Path: Ambiguous Query")
            yield "질문이 너무 짧거나 모호합니다. 내용을 구체적으로 말씀해 주세요."
            return
        
        # 3. 지식 기반 검색 경로 (RETRIEVAL_REQUIRED - 트리거 '#'가 있는 경우)
        logger.info("Routing Path: Knowledge Base Retrieval with Refinement")
        
        # A. 트리거 제거 및 질문 정돈 (라우터 모델 활용)
        str_cleanQuery: str = str_query.strip().lstrip("#")
        str_refinedQuery: str = await self.obj_rewriter.arewrite(str_query=str_cleanQuery)
        
        # B. 정돈된 질문으로 관련 문서 검색 및 리랭킹 (비동기 호출)
        list_retrievedDocs: tp.List[RetrievedDocument] = await self.obj_retriever.aretrieve(
            str_query=str_refinedQuery, 
            int_topK=5
        )
        
        # C. 검색 결과 부재 시 처리
        if not list_retrievedDocs:
            logger.warning("No relevant documents found.")
            yield "죄송합니다. 관련 규정을 찾지 못했습니다. 질문 내용을 다듬어 주시거나 '#' 기호 없이 일상 대화를 시도해 보세요."
            return
        
        # D. 컨텍스트 구성 및 최종 답변 생성 (정돈된 질문 주입)
        str_context: str = self._format_context(list_docs=list_retrievedDocs)
        
        logger.info(f"Generating response using refined query: {str_refinedQuery}")
        async for obj_chunk in self.obj_generatorLlm.astream(
            self.obj_ragPrompt.format_messages(
                context=str_context, 
                question=str_refinedQuery
            )
        ):
            yield obj_chunk
