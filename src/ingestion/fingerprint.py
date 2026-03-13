import hashlib
import uuid
import typing as tp
from pathlib import Path

class FingerprintService:
    """
    데이터의 고유 식별자를 생성하는 책임을 가진다.
    결정적 ID 생성을 통해 중복 인덱싱을 방지한다.
    Qdrant 호환성을 위해 UUID v5 (결정적 UUID) 형식을 사용한다.
    """
    def __init__(self) -> None:
        # 네임스페이스 UUID 정의 (결정적 생성을 위한 기준)
        self.obj_namespace: uuid.UUID = uuid.NAMESPACE_DNS

    def get_file_fingerprint(self, obj_path: Path) -> str:
        """
        파일의 속성(이름, 크기, 수정시간)을 기반으로 결정적 해시를 생성한다.
        """
        obj_hasher: tp.Any = hashlib.sha256()
        
        # 파일명 인코딩 및 업데이트
        obj_hasher.update(obj_path.name.encode("utf-8"))
        
        # 파일 메타데이터 정보 포함
        obj_stat = obj_path.stat()
        obj_hasher.update(str(obj_stat.st_size).encode())
        obj_hasher.update(str(int(obj_stat.st_mtime)).encode())
        
        return str(obj_hasher.hexdigest())

    def get_chunk_id(self, str_fileHash: str, int_index: int) -> str:
        """
        파일 수준의 해시와 청크 인덱스를 결합하여 유효한 UUID 식별자를 생성한다.
        Qdrant는 Point ID로 unsigned int 또는 UUID만 허용한다.
        """
        str_combinedKey: str = f"{str_fileHash}:{int_index}"
        
        # UUID v5를 사용하여 동일한 키에 대해 항상 동일한 UUID 생성 (결정적 ID)
        obj_chunkUuid: uuid.UUID = uuid.uuid5(self.obj_namespace, str_combinedKey)
        
        return str(obj_chunkUuid)
