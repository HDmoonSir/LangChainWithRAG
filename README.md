# Production-Ready Multi-LLM RAG Engine

본 프로젝트는 LangChain(LCEL)을 기반으로 구축된 고성능 로컬 RAG 서비스입니다. 0.5B 모델을 통한 지능형 라우팅과 7B 모델의 강력한 답변 생성을 결합하였으며, 엔터프라이즈 급의 안정성을 위한 구조적 고도화가 적용되었습니다.

## 🌟 Key Features

- **Multi-LLM Pipeline**: 
  - **Router/Rewriter**: Qwen2.5-0.5B (지연 시간 최소화 및 의도 분류)
  - **Generator**: Qwen2.5-7B (정밀한 답변 생성)
- **Hybrid Retrieval & Reranking**: 
  - BGE-M3 임베딩을 통한 다국어 하이브리드 검색 지원
  - BGE-Reranker-v2-m3를 이용한 검색 결과 2차 정밀 재정렬
- **Standardized SSE Streaming**: 
  - `token`, `error`, `metadata`, `finish` 프레임으로 표준화된 JSON 기반 실시간 응답
- **Intelligent Routing**: 
  - `#` 트리거를 통한 명시적 지식 검색(RAG) 강제 활성화 지원

## 🛡️ Technical Excellence

- **Robust Configuration**: Pydantic 기반의 Settings 모델을 도입하여 필수 환경 변수(`.env`) 검증 및 타입 안전성 확보
- **Lifespan Management**: FastAPI의 `lifespan` 이벤트를 통해 파이프라인 및 모델 클라이언트를 서버 시작 시 명시적으로 초기화 및 조립
- **Deterministic Indexing**: 문서 해시(Fingerprint) 기반의 ID 생성 로직을 통해 동일 문서 재업로드 시 `upsert`를 통한 중복 데이터 제거
- **Auto-Dimension Detection**: 임베딩 모델의 차원을 실시간 감지하여 Qdrant 컬렉션 정합성을 자동으로 검증 및 생성
- **Modular Architecture**: 설정, 엔진, API 계층의 명확한 분리를 통한 유지보수성 및 확장성 확보

## 🛠 Tech Stack

- **Framework**: LangChain (LCEL)
- **Serving**: vLLM (OpenAI Compatible API)
- **Parsing**: Docling (High-performance Document Analysis)
- **Vector DB**: Qdrant
- **Models**: Qwen2.5-0.5B-Instruct, Qwen2.5-7B-Instruct-GPTQ-Int4
- **API**: FastAPI (Asynchronous Stream Support)

## 📁 Project Structure

```text
llm_langchain/
├── src/
│   ├── app/               # API Transport Layer (Routes, Factory, Schemas, Streaming)
│   ├── rag/               # Core Domain Logic (Pipeline, Router, Retriever, Rewriter)
│   ├── ingestion/         # Data Pipeline (Docling, Embedder, Fingerprint)
│   ├── config/            # Structured Configuration System
│   └── utils/             # Common Utilities
├── config/                # YAML Settings & Centralized Prompts
├── data/docs/             # Source Documents (PDF/Docx)
├── run_ingest.py          # Ingestion Entrypoint (Upsert Logic)
└── run_server.py          # API Server Entrypoint (Lifespan Logic)
```

## 🚀 Getting Started

### 1. 인프라 가동 (Docker)
```bash
docker-compose up -d
```

### 2. 의존성 설치 및 환경 설정
```bash
pip install -r requirements.txt
cp .env.example .env  # 필수 환경 변수(VLLM_URL 등) 설정
```

### 3. 데이터 인제스천 (중복 방지 지원)
`data/docs/` 폴더에 문서를 배치한 후 실행합니다.
```bash
python run_ingest.py
```

### 4. 서버 실행
```bash
python run_server.py
```

## 📡 API Usage (SSE JSON Frame)

```bash
curl -X POST http://localhost:8000/chat/stream \
     -H "Content-Type: application/json" \
     -d '{"query": "#우리 회사 연차 규정을 요약해줘."}'
```

**응답 예시 (Standardized Frame):**
```json
data: {"str_event": "token", "dict_data": {"token": "연"}}
data: {"str_event": "token", "dict_data": {"token": "차"}}
data: {"str_event": "finish", "dict_data": {"status": "complete"}}
```

---
본 프로젝트는 Titan V (12GB VRAM) 3장 환경에서 검증되었으며, GPU 가용성에 따라 동적으로 실행 장치(CUDA/CPU)를 선택합니다.
