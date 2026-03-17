import typing as tp
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from src.schemas.server import ServerConfig

class RouterGenerator:
    """
    사용자의 질의 의도를 분류한다.
    'router' 모델 설정을 사용한다.
    """
    def __init__(self, config: ServerConfig) -> None:
        """설정을 기반으로 라우터용 설정을 보관한다."""
        self.config: ServerConfig = config
        
        # [FUTURE] LLM 기반 의도 분류 활성화 대비 구조
        """
        obj_cfg = config.llm_servers["router"]
        self.llm: ChatOpenAI = ChatOpenAI(
            model=obj_cfg.model_name,
            openai_api_base=str(obj_cfg.url),
            openai_api_key="none",
            temperature=obj_cfg.temperature
        )
        str_routerSystemPrompt: str = self.config.prompts.get("router_system", "")
        self.router_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(list([
            ("system", str_routerSystemPrompt),
            ("user", "{query}")
        ]))
        """
        logger.info("RouterGenerator initialized (Trigger-based mode).")

    async def aroute_query(self, str_query: str) -> str:
        """질의의 특성을 확인하여 의도를 분류한다."""
        str_stripped: str = str_query.strip()
        if not str_stripped: return "AMBIGUOUS"
        if str_stripped.startswith("#"): return "RETRIEVAL_REQUIRED"
        return "GENERAL_CONVERSATION"
