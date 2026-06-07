# =============================================================================
# reranker.py
# -----------------------------------------------------------------------------
# 역할: cross-encoder 기반 재순위 모듈.
#
# 파이프라인:
#   벡터 검색 (top_k=10) → cross-encoder 재순위 → 상위 N개 반환
#
# 모델 선택 (BAAI/bge-reranker-v2-m3):
#   원문 텍스트 쌍을 재채점한다. 따라서 임베딩 모델과 같은 계열일 필요가 없다.
#
#
# 첫 실행:
#   모델은 HuggingFace Hub에서 자동 다운로드된다 (~2.2 GB).
#   ~/.cache/huggingface/hub/ 에 캐시되어 이후 실행부터는 로컬에서 로드한다.
#   더 가벼운 대안: RERANKER_MODEL=BAAI/bge-reranker-base (~280 MB, 동일 코드 경로).
# =============================================================================

import os
import logging
from typing import Optional

from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

logger = logging.getLogger(__name__)

USE_RERANKER   = os.getenv("USE_RERANKER",   "1") != "0"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


# =============================================================================
# 팩토리 함수
# =============================================================================

def build_reranker() -> Optional[SentenceTransformerRerank]:
    """
    SentenceTransformerRerank 인스턴스를 생성한다.

    USE_RERANKER=0 이면 None을 반환한다 (재순위 없이 벡터 검색 결과만 사용).

    top_n은 크게(10) 잡아 두고, 실제 최종 절삭은 tools._make_search_fn에서
    final_top_n으로 처리한다 (레인별로 3/2개로 다르게 자른다).

    Returns:
        SentenceTransformerRerank 인스턴스, 또는 USE_RERANKER=0일 때 None
    """
    if not USE_RERANKER:
        logger.info("[reranker] USE_RERANKER=0 — 재순위기 비활성화")
        return None

    logger.info(f"[reranker] 모델 로딩: {RERANKER_MODEL} (첫 실행 시 HuggingFace에서 다운로드)")
    return SentenceTransformerRerank(
        model=RERANKER_MODEL,
        top_n=10,  # 충분히 크게 설정; 실제 절삭은 _make_search_fn에서 처리
    )
