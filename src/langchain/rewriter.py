import typing as tp
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from src.schemas.server import ServerConfig
from src.schemas.common import LLMServerConfig

class QueryRewriter:
    """
    사용자의 질문을 분석하여 검색에 최적화된 형태로 재작성한다.
    가변 인수 사용을 금지하며 명시적 타입을 사용한다.
    """
    def __init__(self, config: ServerConfig) -> None:
        """설정을 기반으로 LLM 및 프롬프트를 초기화한다."""
        self.config: ServerConfig = config
        obj_routerCfg: LLMServerConfig = config.llm_servers["router"]
        
        self.llm: ChatOpenAI = ChatOpenAI(
            model=obj_routerCfg.model_name,
            openai_api_base=str(obj_routerCfg.url),
            openai_api_key="none",
            temperature=obj_routerCfg.temperature
        )
        
        str_systemPrompt: str = self.config.prompts.get("rewriter_system", "")
        self.prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(list([
            ("system", str_systemPrompt),
            ("user", "{query}")
        ]))
        self.parser: StrOutputParser = StrOutputParser()
        
        logger.info(f"QueryRewriter initialized with model: {obj_routerCfg.model_name}")

    async def arewrite(self, str_query: str) -> str:
        """질의를 검색에 적합한 키워드 중심 문장으로 재작성한다."""
        try:
            obj_chain = self.prompt | self.llm | self.parser
            str_rewritten: str = await obj_chain.ainvoke(input=dict(query=str_query))
            return str_rewritten.strip()
        except Exception as obj_err:
            logger.error(f"Query rewriting failed: {obj_err}. Using original query.")
            return str_query
