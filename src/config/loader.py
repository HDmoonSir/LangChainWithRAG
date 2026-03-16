import os
import typing as tp
from pathlib import Path
from omegaconf import OmegaConf
from dotenv import load_dotenv
from loguru import logger

from src.schemas.server import ServerConfig
from src.schemas.ingest import IngestConfig

class ConfigLoaderService:
    """
    OmegaConf Resolve를 사용하여 YAML(${oc.env:VAR, DEFAULT})과 ENV를 결합하고
    Pydantic으로 최종 검증을 수행하는 서비스이다.
    """
    @staticmethod
    def _load_base_container(yaml_path: str, env_path: tp.Optional[str] = None) -> tp.Dict[str, tp.Any]:
        # 1. 환경 변수 파일 로드
        if env_path and Path(env_path).exists():
            load_dotenv(dotenv_path=env_path)
            logger.info(f"Loaded environment variables from {env_path}")
        elif env_path:
            logger.warning(f"Env file not found at {env_path}, skipping load_dotenv.")

        # 2. YAML 파일 존재 확인
        if not Path(yaml_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
            
        # 3. OmegaConf 로드 및 Resolve (환경 변수 치환)
        obj_conf = OmegaConf.load(yaml_path)
        dict_conf = OmegaConf.to_container(obj_conf, resolve=True)
        return tp.cast(tp.Dict[str, tp.Any], dict_conf)

    def build_server_config(self, yaml_path: str, env_path: tp.Optional[str] = None) -> ServerConfig:
        """서버용 설정을 빌드하고 검증한다."""
        dict_conf = self._load_base_container(yaml_path, env_path)
        return ServerConfig.model_validate(dict_conf)

    def build_ingest_config(self, yaml_path: str, env_path: tp.Optional[str] = None) -> IngestConfig:
        """인제스트용 설정을 빌드하고 검증한다."""
        dict_conf = self._load_base_container(yaml_path, env_path)
        return IngestConfig.model_validate(dict_conf)
