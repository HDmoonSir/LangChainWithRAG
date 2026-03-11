import typing as tp
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from loguru import logger
from src.utils.config_loader import config

class IntentClassification(BaseModel):
    intent: str = Field(description="The intent classification")
    reason: str = Field(description="The reason for classification")

class SemanticRouter:
    def __init__(self):
        llm_cfg = config.get_llm_config("router")
        self.llm = ChatOpenAI(
            model=llm_cfg["model_name"],
            openai_api_key="none",
            openai_api_base=llm_cfg["url"],
            temperature=0.0
        )
        
        self.parser = JsonOutputParser(pydantic_object=IntentClassification)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", config.get_prompt("router_system")),
            ("user", "{query}")
        ])
        
        logger.info("Semantic Router initialized. Manual RAG trigger: '#'")

    def route_query(self, query: str) -> IntentClassification:
        """
        사용자 쿼리의 의도를 분류합니다. 
        '#' 기호로 시작하면 즉시 RAG 경로를 타도록 강제합니다.
        """
        clean_query = query.strip()
        
        # 1. '#' 기호 강제 RAG 로직 (Manual Trigger)
        if clean_query.startswith("#"):
            logger.info("Manual RAG trigger (#) detected.")
            return IntentClassification(
                intent="RETRIEVAL_REQUIRED", 
                reason="Manual trigger via hashtag"
            )

        # 2. 모델 기반 분류 (그 외의 경우)
        try:
            res_msg = self.llm.invoke(self.prompt.format(query=clean_query))
            content = res_msg.content.strip()
            
            try:
                parsed = self.parser.parse(content)
                return IntentClassification(**parsed)
            except:
                content_upper = content.upper()
                if "RETRIEVAL" in content_upper:
                    return IntentClassification(intent="RETRIEVAL_REQUIRED", reason="Keyword match in model response")
                return IntentClassification(intent="GENERAL_CONVERSATION", reason="Fallback to general conversation")
                    
        except Exception as e:
            logger.error(f"Routing error: {e}")
            return IntentClassification(intent="RETRIEVAL_REQUIRED", reason="Critical error fallback to RAG")
