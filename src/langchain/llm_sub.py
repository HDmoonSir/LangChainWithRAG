import typing as tp
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from src.schemas.server import ServerConfig

class SubGenerator:
    """
    일상적인 대화 및 인사를 처리한다.
    'sub' 모델 설정을 사용한다.
    """
    def __init__(self, config: ServerConfig) -> None:
        """설정을 기반으로 채팅용 LLM 및 프롬프트를 초기화한다."""
        self.config: ServerConfig = config
        obj_cfg = config.llm_servers["sub"]
        
        self.llm: ChatOpenAI = ChatOpenAI(
            model=obj_cfg.model_name,
            openai_api_base=str(obj_cfg.url),
            openai_api_key="none",
            temperature=obj_cfg.temperature,
            streaming=True
        )
        
        str_generalSystemPrompt: str = self.config.prompts.get("general_chat_system", "")
        self.prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(list([
            ("system", str_generalSystemPrompt),
            ("user", "{query}")
        ]))
        logger.info(f"SubGenerator initialized with model: {obj_cfg.model_name}")

    async def astream_chat(self, str_query: str) -> tp.AsyncGenerator[tp.Any, None]:
        """일상 질의에 대해 스트리밍 응답을 생성한다."""
        async for obj_chunk in self.llm.astream(
            input=self.prompt.format(query=str_query)
        ):
            yield obj_chunk
