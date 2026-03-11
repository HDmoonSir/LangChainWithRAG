# High-Performance Multi-LLM RAG Engine

LangChain을 기반으로 한 로컬 구동형 Multi-LLM RAG 서비스 프로토타입입니다. 0.5B 모델을 이용한 지능형 라우팅 및 7B 모델을 이용한 고성능 답변 생성을 지원합니다.

## 🌟 Key Features

- **Multi-LLM Architecture**: 0.5B 모델(Router/Rewriter)과 7B-Int4 모델(Generator)의 역할 분담으로 효율성 극대화
- **Intelligent Routing**: 질문의 의도(일상 대화 vs 문서 질문)를 파악하여 최적의 경로로 처리
- **High-Performance Ingestion**: Docling을 사용한 정교한 문서 파싱 및 BGE-M3 임베딩 적용
- **Reranking**: BGE-Reranker-v2-m3를 통한 검색 결과 정밀 재정렬
- **FastAPI Streaming**: 지연 시간을 최소화하는 실시간 스트리밍 응답 제공
- **Robustness**: 0.5B 모델의 불안정한 출력을 보완하는 강력한 에러 핸들링 로직

## 🛠 Tech Stack

- **Framework**: LangChain (LCEL)
- **Serving**: vLLM (Docker)
- **Models**: Qwen2.5-0.5B-Instruct, Qwen2.5-7B-Instruct-GPTQ-Int4
- **Parsing**: Docling
- **Embedding**: BGE-M3
- **Vector DB**: Qdrant
- **Reranker**: BGE-Reranker-v2-m3
- **API**: FastAPI

## 📁 Project Structure

```text
llm_langchain/
├── config/
│   └── settings.yaml      # 프롬프트 및 비민감 설정 관리
├── data/
│   └── docs/              # 분석 대상 PDF/Docx 문서 보관
├── src/
│   ├── ingestion/         # 문서 파싱 및 벡터화 로직
│   ├── rag/               # Router, Rewriter, Retriever 등 핵심 로직
│   └── utils/             # Config Loader 등 유틸리티
├── .env.example           # 환경 변수 템플릿
├── docker-compose.yml     # Qdrant 및 vLLM 통합 인프라 설정
├── run_ingest.py          # 데이터 인제스천 실행 스크립트
├── run_server.py          # FastAPI 서버 실행 스크립트
└── requirements.txt       # 의존성 패키지 목록
```

## 🚀 Getting Started

### 1. 인프라 가동 (Docker)
NVIDIA Container Toolkit이 설치된 환경에서 아래 명령어를 실행합니다.
```bash
docker-compose up -d
```

### 2. 환경 설정
`.env.example` 파일을 복사하여 `.env` 파일을 생성하고 실제 IP 주소를 입력합니다.
```bash
cp .env.example .env
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 데이터 인제스천
`data/docs/` 폴더에 PDF 파일을 넣은 후 실행합니다.
```bash
python run_ingest.py
```

### 5. 서버 실행
```bash
python run_server.py
```

## 📡 API Usage

### Streaming Chat
```bash
curl -X POST http://localhost:8000/chat/stream \
     -H "Content-Type: application/json" \
     -d '{"query": "유급 휴가 규정에 대해서 설명해줘"}' \
     --no-buffer
```

---
본 프로젝트는 Titan V (12GB VRAM) 3장 환경에 최적화되어 설계되었습니다.
