import typing as tp
from loguru import logger
from src.schemas.server import ServerConfig

class SemanticRouter:
    """
    사용자의 자연어 질의에서 '#' 트리거 여부를 확인하여 처리 경로를 결정한다.
    가변 인수 사용을 금지하며 명시적 타입을 사용한다.
    """
    def __init__(self, config: ServerConfig) -> None:
        """주입받은 설정을 보관한다."""
        self.config: ServerConfig = config
        logger.info("SemanticRouter initialized (Trigger-based mode).")

    async def aroute_query(self, str_query: str) -> str:
        """
        질의의 시작 문자를 확인하여 의도를 즉시 분류한다.
        - '#'으로 시작: RETRIEVAL_REQUIRED
        - 그 외: GENERAL_CONVERSATION
        """
        str_stripped: str = str_query.strip()
        
        if not str_stripped:
            return "AMBIGUOUS"

        if str_stripped.startswith("#"):
            return "RETRIEVAL_REQUIRED"
        
        return "GENERAL_CONVERSATION"
