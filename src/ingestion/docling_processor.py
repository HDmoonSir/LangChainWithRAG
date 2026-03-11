import typing as tp
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from loguru import logger

class DoclingProcessor:
    def __init__(self):
        self.converter = DocumentConverter()
        logger.info("Docling DocumentConverter initialized.")

    def process_document(self, file_path: str) -> tp.Dict[str, tp.Any]:
        """
        문서를 파싱하여 텍스트 및 구조 정보를 반환합니다.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing document: {path.name}")
        result = self.converter.convert(path)
        
        # Markdown 형식으로 내보내기 (구조 유지에 용이)
        content_md = result.document.export_to_markdown()
        
        return {
            "metadata": {
                "source": path.name,
                "file_type": path.suffix,
                "title": result.document.name or path.stem
            },
            "content": content_md,
            "raw_doc": result.document
        }

if __name__ == "__main__":
    # 테스트 코드
    import sys
    if len(sys.argv) > 1:
        processor = DoclingProcessor()
        data = processor.process_document(sys.argv[1])
        print(f"--- Processed {data['metadata']['source']} ---")
        print(data['content'][:500] + "...")
