# =============================================================================
# reranker_lc.py
# -----------------------------------------------------------------------------
# 역할: LangChain 기반 cross-encoder 재순위 모듈.
#       reranker.py(LlamaIndex)의 LangChain 대응 구현.
#
# LlamaIndex → LangChain 매핑:
#   SentenceTransformerRerank  →  CrossEncoderReranker (langchain)
#                                  + HuggingFaceCrossEncoder (langchain_community)
#   reranker.postprocess_nodes →  ContextualCompressionRetriever (tools_lc.py에서 조합)
#
# 설계:
#   HuggingFaceCrossEncoder 모델은 한 번만 로드한다.
#   레인별로 top_n이 다르므로 CrossEncoderReranker 인스턴스를 레인 수만큼 생성한다.
#   tools_lc.py가 이 인스턴스를 ContextualCompressionRetriever에 전달한다.
#
# 환경변수:
#   USE_RERANKER   : "0"이면 None 반환 (재순위 없이 벡터 검색만 사용)
#   RERANKER_MODEL : HuggingFace 모델 이름 (기본 BAAI/bge-reranker-v2-m3)
# =============================================================================

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

USE_RERANKER   = os.getenv("USE_RERANKER",   "1") != "0"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


# =============================================================================
# 레인별 재순위기 묶음
# =============================================================================

@dataclass
class LaneRerankers:
    """
    세 검색 레인 각각에 맞는 CrossEncoderReranker 인스턴스.

    레인별 final_top_n:
      regulation/law : 3개 (사례보다 짧아 컨텍스트 여유 있음)
      case           : 2개 (사례 본문이 길어 컨텍스트 절약)
    """
    regulation: object   # CrossEncoderReranker, top_n=3
    law:        object   # CrossEncoderReranker, top_n=3
    case:       object   # CrossEncoderReranker, top_n=2


# =============================================================================
# 팩토리 함수
# =============================================================================

def build_reranker() -> Optional[LaneRerankers]:
    """
    LaneRerankers 인스턴스를 생성한다.

    USE_RERANKER=0 이면 None을 반환한다 (재순위 없이 벡터 검색 결과만 사용).

    HuggingFaceCrossEncoder는 한 번만 로드하고, CrossEncoderReranker 인스턴스를
    레인 수만큼(3개) 생성한다 — 각 레인의 top_n이 다르기 때문이다.

    Returns:
        LaneRerankers 인스턴스, 또는 USE_RERANKER=0일 때 None
    """
    if not USE_RERANKER:
        logger.info("[reranker_lc] USE_RERANKER=0 — 재순위기 비활성화")
        return None

    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    from langchain.retrievers.document_compressors import CrossEncoderReranker

    logger.info(
        f"[reranker_lc] 모델 로딩: {RERANKER_MODEL} "
        "(첫 실행 시 HuggingFace에서 다운로드)"
    )
    ce = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)

    return LaneRerankers(
        regulation=CrossEncoderReranker(model=ce, top_n=3),
        law=CrossEncoderReranker(model=ce, top_n=3),
        case=CrossEncoderReranker(model=ce, top_n=2),
    )
