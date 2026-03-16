import os
import glob
import typing as tp
from pathlib import Path
from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.loader import ConfigLoaderService
from src.schemas.ingest import IngestConfig, IngestParams
from src.ingestion.docling_processor import DoclingProcessor
from src.ingestion.embedder import RAGEmbedder
from src.ingestion.fingerprint import FingerprintService

def main() -> None:
    """
    환경 변수로부터 설정 경로를 읽어 인제스트 프로세스를 실행한다.
    가변 인수(*args, **kwargs)를 사용하지 않으며 모든 타입을 명시한다.
    """
    # 1. 실행 환경 변수 로드 (run_server.py와 동일한 방식)
    str_configPath: str = os.getenv("CONFIG_PATH", "config/settings.ingest.yaml")
    str_envPath: str = os.getenv("ENV_PATH", ".env")

    logger.info(f"Starting ingestion process using config: {str_configPath}")

    # 2. 설정 로드 (Dependency Injection Root)
    obj_loader: ConfigLoaderService = ConfigLoaderService()
    obj_config: IngestConfig = obj_loader.build_ingest_config(
        yaml_path=str_configPath, 
        env_path=str_envPath
    )

    # 3. 의존성 조립 (Explicit Initialization)
    obj_processor: DoclingProcessor = DoclingProcessor()
    obj_fingerprinter: FingerprintService = FingerprintService()
    obj_embedder: RAGEmbedder = RAGEmbedder(config=obj_config)

    # 4. 텍스트 스플리터 설정 (Pydantic 스키마 기반 필드 접근)
    obj_params: IngestParams = obj_config.ingest
    obj_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
        chunk_size=obj_params.chunk_size,
        chunk_overlap=obj_params.chunk_overlap,
        separators=list(["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""])
    )

    # 5. 파일 스캔 및 처리 (I/O Boundary)
    list_files: tp.List[Path] = [Path(str_p) for str_p in glob.glob(obj_params.input_glob)]
    if not list_files:
        logger.warning(f"No files found for pattern: {obj_params.input_glob}")
        return

    # 6. 인덱싱 루프 수행
    for obj_filePath in list_files:
        if not obj_filePath.is_file():
            continue
            
        try:
            # A. 문서 파싱 및 내용 추출 (명시적 인수 전달)
            dict_docData: tp.Dict[str, tp.Any] = obj_processor.process_document(str_filePath=str(obj_filePath))
            
            # B. 파일 지문(Fingerprint) 생성
            str_fileHash: str = obj_fingerprinter.get_file_fingerprint(obj_path=obj_filePath)
            
            # C. 텍스트 청킹 실행 (명시적 인수 전달)
            list_chunks: tp.List[str] = obj_splitter.split_text(text=dict_docData["content"])
            
            list_ids: tp.List[str] = list()
            list_metadatas: tp.List[tp.Dict[str, tp.Any]] = list()
            
            for int_i, str_text in enumerate(list_chunks):
                str_cid: str = obj_fingerprinter.get_chunk_id(str_fileHash=str_fileHash, int_index=int_i)
                list_ids.append(str_cid)
                
                dict_meta: tp.Dict[str, tp.Any] = dict(dict_docData["metadata"])
                dict_meta.update({"doc_id": str_fileHash, "chunk_index": int_i})
                list_metadatas.append(dict_meta)
            
            # D. 벡터 DB Upsert (명시적 인수 전달)
            obj_embedder.upsert_documents(
                list_texts=list_chunks, 
                list_metadatas=list_metadatas, 
                list_ids=list_ids
            )
            logger.success(f"Successfully indexed: {obj_filePath.name}")
            
        except Exception as obj_err:
            logger.error(f"Error processing file '{obj_filePath.name}': {obj_err}")

    logger.info("Ingestion process completed.")

if __name__ == "__main__":
    main()
