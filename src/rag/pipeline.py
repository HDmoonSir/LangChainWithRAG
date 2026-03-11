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
        logger.info("Full RAG Pipeline initialized with YAML.")

    def format_docs(self, docs: tp.List[tp.Dict[str, tp.Any]]) -> str:
        return "\n\n".join([f"[Doc {i+1}] {d['content']}" for i, d in enumerate(docs)])

    def run(self, query: str) -> tp.Generator[str, None, None]:
        intent_res = self.router.route_query(query)
        
        # intent_res가 객체면 .intent, 문자열이면 그대로 사용
        intent = getattr(intent_res, 'intent', str(intent_res))
        
        if "GENERAL_CONVERSATION" in intent:
            return self.generator_llm.stream(f"Respond kindly in Korean: {query}")
        
        elif "AMBIGUOUS" in intent:
            return (s for s in ["질문이 모호합니다. 좀 더 자세히 설명해 주세요."])
        
        optimized_query = self.rewriter.rewrite(query)
        retrieved_docs = self.retriever.retrieve(optimized_query)
        context = self.format_docs(retrieved_docs)
        
        return self.generator_llm.stream(
            self.rag_prompt.format(context=context, question=query)
        )
