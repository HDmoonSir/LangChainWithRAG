import typing as tp
from loguru import logger
from src.config.schemas import AppConfig

class SemanticRouter:
    """
    사용자의 자연어 질의에서 '#' 트리거 여부를 확인하여 처리 경로를 결정하는 책임을 가진다.
    LLM 기반 분류 대신 규칙 기반(Rule-based)의 결정적 라우팅을 수행한다.
    """
    def __init__(self, obj_config: AppConfig) -> None:
        """
        주입받은 설정을 보관한다. 트리거 방식에서는 별도의 모델 로딩이 필요하지 않다.
        """
        self.obj_config: AppConfig = obj_config
        logger.info("SemanticRouter initialized (Trigger-based mode).")

    async def aroute_query(self, str_query: str) -> str:
        """
        질의의 시작 문자를 확인하여 의도를 즉시 분류한다.
        - '#'으로 시작: RETRIEVAL_REQUIRED (RAG 검색 경로)
        - 그 외: GENERAL_CONVERSATION (일반 대화 경로)
        - 빈 값: AMBIGUOUS
        """
        str_stripped: str = str_query.strip()
        
        if not str_stripped:
            logger.warning("Empty query received.")
            return "AMBIGUOUS"

        # 트리거 기호 '#' 확인
        if str_stripped.startswith("#"):
            logger.info(f"Trigger '#' detected. Routing to RETRIEVAL_REQUIRED.")
            return "RETRIEVAL_REQUIRED"
        
        logger.info(f"No trigger detected. Routing to GENERAL_CONVERSATION.")
        return "GENERAL_CONVERSATION"
