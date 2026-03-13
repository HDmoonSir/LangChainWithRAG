import typing as tp
from docling.document_converter import DocumentConverter
from loguru import logger

class DoclingProcessor:
    """
    다양한 포맷의 문서를 구조화된 텍스트로 변환하는 책임을 가진다.
    """
    def __init__(self) -> None:
        """
        Docling 변환기를 초기화한다.
        """
        self.obj_converter: DocumentConverter = DocumentConverter()
        logger.info("DoclingProcessor initialized.")

    def process_document(self, str_filePath: str) -> tp.Dict[str, tp.Any]:
        """
        파일을 파싱하여 마크다운 텍스트와 메타데이터를 표준 딕셔너리로 반환한다.
        """
        logger.info(f"Processing document: {str_filePath}")
        
        # 1. 문서 변환 실행
        obj_result: tp.Any = self.obj_converter.convert(str_filePath)
        
        # 2. 마크다운 형식으로 내용 추출 (네이밍 규칙 str_*)
        str_markdownContent: str = obj_result.document.export_to_markdown()
        
        # 3. 데이터 계약을 위한 결과물 조립 사용)
        dict_output: tp.Dict[str, tp.Any] = dict()
        dict_output["content"] = str_markdownContent
        
        # 메타데이터 추출 및 조립
        dict_meta: tp.Dict[str, tp.Any] = dict()
        dict_meta["source_path"] = str_filePath
        dict_meta["input_format"] = str(obj_result.input.format)
        
        dict_output["metadata"] = dict_meta
        
        return dict_output
