# Production-Ready Multi-LLM RAG Engine

본 프로젝트는 LangChain(LCEL)을 기반으로 구축된 고성능 로컬 RAG 서비스입니다. 0.5B 모델을 통한 지능형 라우팅과 7B 모델의 강력한 답변 생성을 결합하였으며, 엔터프라이즈 급의 안정성을 위한 구조적 고도화가 적용되었습니다.

## 🌟 Key Features

- **Multi-LLM Pipeline**: 
  - **Router**: Qwen2.5-0.5B (지연 시간 최소화 및 트리거 기반 의도 분류)
  - **Generator**: Qwen2.5-7B (정밀한 답변 생성)
- **Two-stage Retrieval & Reranking**: 
  - **Stage 1 (Vector Search)**: BGE-M3 임베딩을 통한 다국어 하이브리드 검색 지원
  - **Stage 2 (Reranking)**: BGE-Reranker-v2-m3를 이용해 검색 결과의 관련성을 재평가하여 최상위 컨텍스트 정밀 재정렬
- **Standardized SSE Streaming**: 
  - `token`, `error`, `metadata`, `finish` 프레임으로 표준화된 JSON 기반 실시간 응답
- **Intelligent Routing**: 
  - `#` 트리거를 통한 명시적 지식 검색(RAG) 강제 활성화 지원

## 🛡️ Technical Excellence

- **Deterministic Indexing**: 문서 해시(Fingerprint) 기반의 고유 ID 생성 로직을 통해 동일 문서 재업로드 시 `upsert`를 수행하여 중복 데이터 자동 제거
- **Auto-Dimension Detection**: 사용 중인 임베딩 모델의 차원(Dimension)을 실시간으로 감지하여 Qdrant 컬렉션 정합성을 자동으로 검증 및 생성
- **Robust Configuration**: Pydantic 및 OmegaConf를 통한 환경 변수(`.env`) 검증 및 타입 안전성 확보
- **Lifespan Management**: FastAPI `lifespan`을 통해 모델 클라이언트를 서버 시작 시 1회 초기화 및 주입
- **Modular Architecture**: `schemas`, `config`, `langchain`, `ingestion` 계층의 명확한 분리로 유지보수성 극대화

## 🛠 Tech Stack

- **Framework**: LangChain (LCEL)
- **Serving**: vllm (OpenAI Compatible API)
- **Vector DB**: Qdrant
- **Parsing**: Docling (High-performance Document Analysis)
- **API**: FastAPI (Asynchronous Stream Support)

## 📁 Project Structure

```text
llm_langchain/
├── src/
│   ├── app/               # API Transport Layer (Routes, Factory, Streaming)
│   ├── langchain/         # Core Domain Logic (Pipeline, Router, Retriever, Rewriter)
│   ├── ingestion/         # Data Pipeline (Docling, Embedder, Fingerprint)
│   ├── schemas/           # Centralized Pydantic Models (Data Contracts)
│   └── config/            # Structured Configuration Loader
├── config/                # YAML Settings & Centralized Prompts
├── static/                # Web Chat Interface (HTML, JS, CSS)
├── data/docs/             # Source Documents (PDF/Docx)
├── docker/                # Infrastructure Orchestration (Qdrant, vLLM)
├── run_ingest.py          # Ingestion Entrypoint
└── run_server.py          # API Server Entrypoint
```

## 🚀 Getting Started

### 1. 인프라 가동 (Docker)
로컬 인프라(Qdrant 및 vLLM 모델 서버)를 기동합니다.
```bash
# Qdrant 및 모델 서버 기동 (llm-langchain-net 사용)
docker-compose -f docker/docker-compose.infra.yaml up -d
```

### 2. 의존성 설치 및 환경 설정
시스템 환경(CUDA 버전 등)에 맞춰 필요한 패키지를 설치합니다.
```bash
pip install -r requirements.txt
cp .env.example .env  # 환경에 맞춰 ROUTER_LLM_URL, MAIN_LLM_URL 등을 수정하세요.
```

### 3. 데이터 인제스천 (중복 방지 지원)
`data/docs/` 폴더에 문서를 배치한 후 실행합니다.
```bash
python run_ingest.py
```

### 4. 서버 실행
FastAPI 서버를 실행합니다.
```bash
python run_server.py
```

> **Note on Dockerization**: 본 프로젝트의 API 서버용 `Dockerfile`은 사용자별 실행 환경(CUDA 가용 여부, 기본 이미지 선호도 등)이 상이하므로 포함되어 있지 않습니다. 컨테이너 배포가 필요한 경우, 프로젝트 루트에 `Dockerfile`을 생성하여 `docker/docker-compose.server.yaml`을 참조해 빌드하시기 바랍니다.

## 💻 Web Interface

서버 실행 후 브라우저에서 `http://localhost:7070/`에 접속하면 실시간 채팅 UI를 바로 사용할 수 있습니다.

## 📡 API Usage (SSE JSON Frame)

```bash
curl -X POST http://localhost:8000/chat/stream \
     -H "Content-Type: application/json" \
     -d '{"query": "#우리 회사 연차 규정을 요약해줘."}'
```

---
본 프로젝트는 **Titan V (12GB VRAM) 3장** 환경에서 검증되었으며, GPU 가용성에 따라 동적으로 실행 장치(CUDA/CPU)를 선택하도록 설계되었습니다.
