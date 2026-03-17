import typing as tp
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from src.schemas.server import ServerConfig

class RewriterGenerator:
    """
    검색 쿼리를 최적화한다.
    'rewriter' 모델 설정을 사용한다.
    """
    def __init__(self, config: ServerConfig) -> None:
        """설정을 기반으로 리와이터용 LLM을 초기화한다."""
        self.config: ServerConfig = config
        obj_cfg = config.llm_servers["rewriter"]
        
        self.llm: ChatOpenAI = ChatOpenAI(
            model=obj_cfg.model_name,
            openai_api_base=str(obj_cfg.url),
            openai_api_key="none",
            temperature=obj_cfg.temperature
        )
        
        str_systemPrompt: str = self.config.prompts.get("rewriter_system", "")
        self.prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(list([
            ("system", str_systemPrompt),
            ("user", "{query}")
        ]))
        self.parser: StrOutputParser = StrOutputParser()
        logger.info(f"RewriterGenerator initialized with model: {obj_cfg.model_name}")

    async def arewrite(self, str_query: str) -> str:
        """질의를 검색에 적합한 키워드로 재작성한다."""
        try:
            obj_chain = self.prompt | self.llm | self.parser
            str_rewritten: str = await obj_chain.ainvoke(input=dict(query=str_query))
            return str_rewritten.strip()
        except Exception as obj_err:
            logger.error(f"Rewriting failed: {obj_err}. Using original.")
            return str_query
