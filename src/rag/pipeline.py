import typing as tp
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from src.utils.config_loader import config
from src.rag.router import SemanticRouter
from src.rag.rewriter import QueryRewriter
from src.rag.retriever import RAGRetriever

class RAGPipeline:
    def __init__(self):
        self.router = SemanticRouter()
        self.rewriter = QueryRewriter()
        self.retriever = RAGRetriever()
        
        main_cfg = config.get_llm_config("main")
        self.generator_llm = ChatOpenAI(
            model=main_cfg["model_name"],
            openai_api_key="none",
            openai_api_base=main_cfg["url"],
            temperature=main_cfg["temperature"],
            streaming=True
        )
        
        self.rag_prompt = ChatPromptTemplate.from_template(
            config.get_prompt("rag_answer_system") + "\n\n### Context:\n{context}\n\n### Question:\n{question}\n\n### Answer:"
        )
        logger.info("Full RAG Pipeline initialized.")

    def format_docs(self, docs: tp.List[tp.Dict[str, tp.Any]]) -> str:
        return "\n\n".join([f"[Doc {i+1}] {d['content']}" for i, d in enumerate(docs)])

    def run(self, query: str) -> tp.Generator[str, None, None]:
        logger.info(f"--- Pipeline start for query: {query} ---")
        
        # 전처리: '#' 기호 제거 (라우팅 판단 후 실제 검색/답변용)
        clean_query = query.strip().lstrip("#")
        
        # Step 1: Router
        intent_res = self.router.route_query(query)
        intent = getattr(intent_res, 'intent', str(intent_res))
        logger.info(f"Step 1: Intent classified as -> {intent}")
        
        # Step 2: Routing Logic
        if "GENERAL_CONVERSATION" in intent:
            logger.info("Step 2: Routing to General Conversation")
            return self.generator_llm.stream(f"사용자의 질문에 한국어로 친절하게 답해주세요: {clean_query}")
        
        elif "AMBIGUOUS" in intent:
            logger.info("Step 2: Routing to Ambiguous Handler")
            return (s for s in ["질문이 조금 모호합니다. 어떤 규정이 궁금하신지 구체적으로 말씀해 주시면 찾아드리겠습니다."])
        
        # Step 3: Retrieval Path
        logger.info("Step 2: Routing to RAG Path")
        
        optimized_query = self.rewriter.rewrite(clean_query)
        logger.info(f"Step 3: Query rewritten to -> {optimized_query}")
        
        retrieved_docs = self.retriever.retrieve(optimized_query)
        logger.info(f"Step 4: Retrieved {len(retrieved_docs)} documents")
        
        context = self.format_docs(retrieved_docs)
        
        logger.info("Step 5: Generating final answer with context...")
        return self.generator_llm.stream(
            self.rag_prompt.format(context=context, question=clean_query)
        )
