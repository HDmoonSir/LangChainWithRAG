import uvicorn
from fastapi import FastAPI
from src.app.factory import AppFactory

# 1. 앱 팩토리를 통한 애플리케이션 인스턴스 생성
app: FastAPI = AppFactory.create_app()

if __name__ == "__main__":
    # 2. Uvicorn 서버 실행 상세 설정
    uvicorn.run(
        "run_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # 운영 환경 안전성을 위해 False
        workers=1      # 비동기 LLM 서비스의 특성상 단일 워커의 동시성 활용 권장
    )
