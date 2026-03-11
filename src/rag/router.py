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
        
        self.chain = self.prompt | self.llm
        logger.info("Semantic Router initialized with robust error handling.")

    def route_query(self, query: str) -> IntentClassification:
        """
        사용자 쿼리의 의도를 분류합니다. 
        0.5B 모델의 불안정한 출력을 대비해 강력한 후처리를 수행합니다.
        """
        try:
            # invoke 결과는 AIMessage
            res_msg = self.llm.invoke(self.prompt.format(query=query))
            content = res_msg.content.strip()
            
            # 1. JSON 파싱 시도
            try:
                parsed = self.parser.parse(content)
                return IntentClassification(**parsed)
            except:
                # 2. 텍스트 직접 매칭 (Fallback)
                content_upper = content.upper()
                if "RETRIEVAL" in content_upper:
                    return IntentClassification(intent="RETRIEVAL_REQUIRED", reason="Keyword match: retrieval")
                elif "GENERAL" in content_upper:
                    return IntentClassification(intent="GENERAL_CONVERSATION", reason="Keyword match: general")
                else:
                    return IntentClassification(intent="AMBIGUOUS", reason="Keyword match: ambiguous")
                    
        except Exception as e:
            logger.error(f"Routing critical error: {e}")
            return IntentClassification(intent="RETRIEVAL_REQUIRED", reason="Critical error fallback")
