import os
import uvicorn
from fastapi import FastAPI
from src.app.factory import AppFactory

def create_app() -> FastAPI:
    """Uvicorn --factory 엔트리포인트"""
    config_path = os.getenv("CONFIG_PATH", "config/settings.server.yaml")
    env_path = os.getenv("ENV_PATH", ".env")
    return AppFactory.create_app(config_path=config_path, env_path=env_path)

if __name__ == "__main__":
    # 로컬 직접 실행 시 uvicorn 가동
    uvicorn.run(
        "run_server:create_app",
        host="0.0.0.0",
        port=8000,
        factory=True,
        reload=True
    )
