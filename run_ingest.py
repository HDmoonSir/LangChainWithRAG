import typing as tp
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from src.ingestion.docling_processor import DoclingProcessor
from src.ingestion.embedder import RAGEmbedder

def main():
    # 1. 초기화
    processor = DoclingProcessor()
    embedder = RAGEmbedder()
    
    # 마크다운 구조를 고려한 텍스트 분할 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
    )

    # 2. 문서 찾기
    docs_path = "data/docs/*"
    files = glob.glob(docs_path)
    if not files:
        logger.warning(f"No files found in {docs_path}. Please add PDF or Docx files.")
        return

    # 3. 문서별 처리 루프
    for file_path in files:
        try:
            # A. Docling으로 파싱
            doc_data = processor.process_document(file_path)
            
            # B. 텍스트 청킹
            chunks = text_splitter.split_text(doc_data["content"])
            logger.info(f"Split {file_path} into {len(chunks)} chunks.")
            
            # C. 메타데이터 생성 및 저장
            metadatas = [
                {**doc_data["metadata"], "chunk_index": i} 
                for i in range(len(chunks))
            ]
            
            embedder.add_documents(chunks, metadatas)
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    main()
