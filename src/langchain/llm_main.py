import typing as tp
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from src.schemas.server import ServerConfig

class MainGenerator:
    """
    검색된 컨텍스트를 바탕으로 최종 RAG 답변을 생성한다.
    'main' 모델 설정을 사용한다.
    """
    def __init__(self, config: ServerConfig) -> None:
        """설정을 기반으로 생성용 LLM 및 프롬프트를 초기화한다."""
        self.config: ServerConfig = config
        obj_cfg = config.llm_servers["main"]
        
        self.llm: ChatOpenAI = ChatOpenAI(
            model=obj_cfg.model_name,
            openai_api_base=str(obj_cfg.url),
            openai_api_key="none",
            temperature=obj_cfg.temperature,
            streaming=True
        )
        
        str_ragSystemPrompt: str = self.config.prompts.get("rag_answer_system", "")
        self.prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(list([
            ("system", str_ragSystemPrompt),
            ("user", "### Context:\n{context}\n\n### Question:\n{question}\n\n### Answer:")
        ]))
        logger.info(f"MainGenerator initialized with model: {obj_cfg.model_name}")

    async def agenerate(self, str_context: str, str_question: str) -> tp.AsyncGenerator[tp.Any, None]:
        """컨텍스트와 질문을 결합하여 스트리밍 방식으로 답변을 생성한다."""
        async for obj_chunk in self.llm.astream(
            input=self.prompt.format_messages(context=str_context, question=str_question)
        ):
            yield obj_chunk
