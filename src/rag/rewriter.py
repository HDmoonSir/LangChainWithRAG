import typing as tp
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from src.config.schemas import AppConfig, LLMServerConfig

class QueryRewriter:
    """
    사용자의 질문을 분석하여 검색에 최적화된 형태로 재작성하고 문장을 정돈하는 책임을 가진다.
    라우터 모델(qwen-router)을 활용하여 질문의 핵심 의도를 추출한다.
    """
    def __init__(self, obj_config: AppConfig) -> None:
        """
        주입받은 설정을 통해 재작성용 LLM과 프롬프트 체인을 초기화한다.
        """
        self.obj_config: AppConfig = obj_config
        obj_routerCfg: LLMServerConfig = obj_config.dict_llmServers["router"]
        
        # 1. 재작성용 LLM 초기화 (라우터 모델 활용)
        self.obj_llm: ChatOpenAI = ChatOpenAI(
            model=obj_routerCfg.str_modelName,
            openai_api_key="none",
            openai_api_base=str(obj_routerCfg.obj_url),
            temperature=obj_routerCfg.float_temperature
        )
        
        # 2. 프롬프트 및 출력 파서 설정
        str_systemPrompt: str = self.obj_config.dict_prompts.get("rewriter_system", "")
        self.obj_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(list([
            ("system", str_systemPrompt),
            ("user", "{query}")
        ]))
        self.obj_parser: StrOutputParser = StrOutputParser()
        
        logger.info(f"QueryRewriter initialized with model: {obj_routerCfg.str_modelName}")

    async def arewrite(self, str_query: str) -> str:
        """
        사용자의 질의를 검색에 적합한 키워드 중심의 정돈된 문장으로 재작성한다.
        """
        try:
            obj_chain = self.obj_prompt | self.obj_llm | self.obj_parser
            
            # 비동기 추론 실행
            str_rewritten: str = await obj_chain.ainvoke(dict(query=str_query))
            str_final: str = str_rewritten.strip()
            
            logger.info(f"Query refined: '{str_query}' -> '{str_final}'")
            return str_final
            
        except Exception as obj_err:
            logger.error(f"Query rewriting failed: {obj_err}. Using original query.")
            return str_query
