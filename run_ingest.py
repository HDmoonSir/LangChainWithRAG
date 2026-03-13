import os
import glob
import typing as tp
from pathlib import Path
from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter

# GPU 관련 라이브러리 초기화 전 환경변수 설정
# vLLM과 로컬 임베딩 모델 간의 CUDA 장치 충돌 방지 및 안전한 GPU 활용 보장
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # CPU-only 모드 해제

from src.config.loader import ConfigLoaderService
from src.config.schemas import AppConfig
from src.ingestion.docling_processor import DoclingProcessor
from src.ingestion.embedder import RAGEmbedder
from src.ingestion.fingerprint import FingerprintService

def main() -> None:
    """
    애플리케이션의 인덱싱(Ingestion) 프로세스를 총괄하는 메인 진입점이다.
    설정 로드, 의존성 조립, 파일 스캔 및 인덱싱 오케스트레이션을 수행한다.
    """
    logger.info("Starting ingestion process (CPU-only mode for safety)...")

    # 1. 설정 로드 (Composition Root / Entrypoint Ownership)
    obj_configLoader: ConfigLoaderService = ConfigLoaderService()
    obj_appConfig: AppConfig = obj_configLoader.build_app_config(
        str_yamlPath="config/settings.yaml",
        str_envPath=".env"
    )

    # 2. 의존성 조립 (Explicit Dependency Injection)
    obj_processor: DoclingProcessor = DoclingProcessor()
    obj_fingerprinter: FingerprintService = FingerprintService()
    obj_embedder: RAGEmbedder = RAGEmbedder(obj_config=obj_appConfig)

    # 3. 텍스트 스플리터 설정 (마크다운 구조 최적화)
    obj_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=list(["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""])
    )

    # 4. 대상 파일 스캔 (I/O Boundary)
    str_docsPattern: str = "data/docs/*"
    list_files: tp.List[Path] = [Path(str_p) for str_p in glob.glob(str_docsPattern)]
    
    if not list_files:
        logger.warning(f"No files found in {str_docsPattern}. Ingestion skipped.")
        return

    logger.info(f"Found {len(list_files)} files for processing.")

    # 5. 파일별 처리 루프
    for obj_filePath in list_files:
        if not obj_filePath.is_file():
            continue
            
        try:
            # A. 문서 파싱 및 내용 추출
            dict_docData: tp.Dict[str, tp.Any] = obj_processor.process_document(str(obj_filePath))
            
            # B. 파일 지문(Fingerprint) 생성
            str_fileHash: str = obj_fingerprinter.get_file_fingerprint(obj_filePath)
            
            # C. 텍스트 청킹 실행
            list_chunks: tp.List[str] = obj_splitter.split_text(dict_docData["content"])
            logger.info(f"Split '{obj_filePath.name}' into {len(list_chunks)} chunks.")
            
            # D. 결정적 ID 및 메타데이터 목록 생성
            list_ids: tp.List[str] = list()
            list_metadatas: tp.List[tp.Dict[str, tp.Any]] = list()
            
            for int_i, str_text in enumerate(list_chunks):
                str_cid: str = obj_fingerprinter.get_chunk_id(str_fileHash, int_i)
                list_ids.append(str_cid)
                
                dict_meta: tp.Dict[str, tp.Any] = dict(dict_docData["metadata"])
                dict_meta["doc_id"] = str_fileHash
                dict_meta["chunk_index"] = int_i
                list_metadatas.append(dict_meta)
            
            # E. 벡터 DB Upsert
            obj_embedder.upsert_documents(
                list_texts=list_chunks,
                list_metadatas=list_metadatas,
                list_ids=list_ids
            )
            logger.success(f"Successfully indexed file: {obj_filePath.name}")
            
        except Exception as obj_err:
            logger.error(f"Error processing file '{obj_filePath.name}': {obj_err}")

    logger.info("Ingestion process completed.")

if __name__ == "__main__":
    main()
