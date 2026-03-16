import os
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.app.factory import AppFactory

def create_app() -> FastAPI:
    """Uvicorn --factory 엔트리포인트"""
    config_path = os.getenv("CONFIG_PATH", "config/settings.server.yaml")
    env_path = os.getenv("ENV_PATH", ".env")
    
    # 1. 원본 앱 생성
    app: FastAPI = AppFactory.create_app(config_path=config_path, env_path=env_path)
    
    # 2. 정적 파일 마운트 (static/ 디렉토리)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    # 3. 루트(/) 경로에서 index.html 서빙
    @app.get("/", tags=["UI"])
    async def get_ui():
        return FileResponse("static/index.html")
    
    return app

if __name__ == "__main__":
    # 로컬 직접 실행 시 uvicorn 가동
    uvicorn.run(
        "run_server:create_app",
        host="0.0.0.0",
        port=8000,
        factory=True,
        reload=True
    )
