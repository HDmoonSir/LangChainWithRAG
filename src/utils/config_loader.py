import os
import yaml
import typing as tp
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class ConfigLoader:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.settings = yaml.safe_load(f)

    def get_llm_config(self, key: str) -> tp.Dict[str, tp.Any]:
        cfg = self.settings.get("llm_servers", {}).get(key, {}).copy()
        # URL 정보는 환경 변수에서 가져옴
        if key == "router":
            cfg["url"] = os.getenv("ROUTER_LLM_URL")
        elif key == "main":
            cfg["url"] = os.getenv("MAIN_LLM_URL")
        return cfg

    def get_vector_db_config(self) -> tp.Dict[str, tp.Any]:
        cfg = self.settings.get("vector_db", {}).copy()
        # URL 정보는 환경 변수에서 가져옴
        cfg["qdrant_url"] = os.getenv("QDRANT_URL")
        return cfg

    def get_prompt(self, key: str) -> str:
        return self.settings.get("prompts", {}).get(key, "")

config = ConfigLoader()
