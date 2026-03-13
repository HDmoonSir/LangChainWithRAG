import json
import typing as tp
from loguru import logger
from src.app.schemas import SSEFrame

class SSEFrameBuilder:
    """
    표준화된 JSON SSE(Server-Sent Events) 프레임을 생성하고 직렬화하는 책임을 가진다.
    """
    @staticmethod
    def build_frame(str_event: str, dict_data: tp.Dict[str, tp.Any]) -> str:
        """
        이벤트 유형과 데이터를 기반으로 SSE 문자열 프레임을 생성한다.
        """
        obj_frame: SSEFrame = SSEFrame(str_event=str_event, dict_data=dict_data)
        str_jsonPayload: str = json.dumps(obj_frame.model_dump(), ensure_ascii=False)
        return f"data: {str_jsonPayload}\n\n"

class ChatStreamGenerator:
    """
    RAG 파이프라인의 출력을 SSE 포맷으로 변환하는 비동기 제너레이터를 관리한다.
    """
    def __init__(self, obj_pipelineRun: tp.AsyncGenerator[tp.Any, None]) -> None:
        self.obj_pipelineRun: tp.AsyncGenerator[tp.Any, None] = obj_pipelineRun

    async def generate(self) -> tp.AsyncGenerator[str, None]:
        """
        파이프라인 이벤트를 순회하며 SSE 프레임 문자열을 생성하여 yield한다.
        """
        try:
            # 스트리밍 시작 알림
            yield SSEFrameBuilder.build_frame("start", dict(message="Generating response..."))

            async for obj_chunk in self.obj_pipelineRun:
                # LangChain 메시지 청크에서 텍스트 토큰 추출
                str_token: str = getattr(obj_chunk, 'content', str(obj_chunk))
                if str_token:
                    yield SSEFrameBuilder.build_frame("token", dict(token=str_token))

            # 스트리밍 정상 종료 알림
            yield SSEFrameBuilder.build_frame("finish", dict(status="complete"))

        except Exception as obj_error:
            logger.error(f"Error during streaming: {obj_error}")
            yield SSEFrameBuilder.build_frame("error", dict(message=str(obj_error)))
