import os
import typing as tp
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from dotenv import load_dotenv
from src.config.schemas import AppConfig

class ConfigLoaderService:
    """
    외부 리소스로부터 설정을 로드하고 AppConfig 객체를 조립하는 책임을 가진다.
    """
    def build_app_config(self, str_yamlPath: str, str_envPath: str) -> AppConfig:
        """
        주입받은 경로를 바탕으로 YAML과 환경변수를 병합하여 검증된 AppConfig를 반환한다.
        vLLM 등 외부 환경 요인을 고려한 명시적 장치 주입을 지원한다.
        """
        # 1. 환경 변수 파일 로드 (I/O Boundary)
        load_dotenv(dotenv_path=str_envPath)
        
        # 2. YAML 설정 파일 로드 (I/O Boundary)
        obj_yamlPath: Path = Path(str_yamlPath)
        if not obj_yamlPath.exists():
            raise FileNotFoundError(f"Configuration file not found: {str_yamlPath}")
            
        # OmegaConf를 통한 정적 설정 로드 및 명시적 타입 캐스팅
        obj_yamlConfig: DictConfig = tp.cast(DictConfig, OmegaConf.load(obj_yamlPath))
        
        # 3. 환경 변수를 주입하기 위한 DotList 생성 사용)
        list_envDotList: tp.List[str] = list()
        list_envDotList.append(f"llm_servers.router.url={os.getenv('ROUTER_LLM_URL', '')}")
        list_envDotList.append(f"llm_servers.main.url={os.getenv('MAIN_LLM_URL', '')}")
        list_envDotList.append(f"vector_db.qdrant_url={os.getenv('QDRANT_URL', '')}")
        
        # vLLM 환경을 고려한 로컬 모델 전용 실행 장치 주입 (기본값 cpu)
        list_envDotList.append(
            f"vector_db.local_model_device={os.getenv('LOCAL_MODEL_DEVICE', 'cpu')}"
        )
        
        # 4. 정적 설정(YAML)과 동적 설정(ENV)의 계층적 병합
        obj_envConfig: DictConfig = tp.cast(DictConfig, OmegaConf.from_dotlist(list_envDotList))
        obj_mergedConfig: DictConfig = tp.cast(DictConfig, OmegaConf.merge(obj_yamlConfig, obj_envConfig))
        
        # 5. Pydantic 모델 검증을 위해 순수 Python dict로 변환
        dict_finalConfig: tp.Dict[str, tp.Any] = tp.cast(
            tp.Dict[str, tp.Any],
            OmegaConf.to_container(obj_mergedConfig, resolve=True)
        )
        
        # 6. 최종 데이터 계약 모델(AppConfig) 생성 및 반환
        return AppConfig.model_validate(dict_finalConfig)
