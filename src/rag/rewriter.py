import typing as tp
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from src.utils.config_loader import config

class QueryRewriter:
    def __init__(self):
        llm_cfg = config.get_llm_config("router")
        self.llm = ChatOpenAI(
            model=llm_cfg["model_name"],
            openai_api_key="none",
            openai_api_base=llm_cfg["url"],
            temperature=0.1
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", config.get_prompt("rewriter_system")),
            ("user", "{query}")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
        logger.info("Query Rewriter initialized with YAML config.")

    def rewrite(self, query: str) -> str:
        try:
            rewritten = self.chain.invoke({"query": query})
            return rewritten.strip().replace('"', '').replace("'", "")
        except Exception as e:
            logger.error(f"Error in rewriting: {e}")
            return query
