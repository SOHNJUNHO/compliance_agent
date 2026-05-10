# =============================================================================
# reranker.py
# -----------------------------------------------------------------------------
# 역할: Qwen3-Reranker 기반 cross-encoder 재순위 모듈.
#
# 동작 원리:
#   벡터 유사도 검색(bi-encoder)은 쿼리와 청크를 각각 독립적으로 임베딩해
#   코사인 유사도로 순위를 매긴다. 속도는 빠르지만 두 텍스트를 함께 보지 않아
#   정밀도가 떨어질 수 있다.
#
#   Cross-encoder(재순위기)는 (쿼리, 청크) 쌍을 하나의 입력으로 받아
#   관련성 점수를 직접 예측한다. 느리지만 정밀도가 높다.
#
# 파이프라인:
#   벡터 검색 (top_k=10) → ComplianceReranker → 상위 3개 반환
#
# Qwen3-Reranker 특이사항:
#   이 모델은 쿼리 앞에 도메인 지시문(task instruction)을 요구한다.
#   ComplianceReranker가 부모 클래스 호출 전에 지시문을 자동으로 붙인다.
#   지시문 없이 사용하면 일반 문서 검색 기준으로 점수를 매겨 정밀도가 낮아진다.
#
# 환경변수:
#   USE_RERANKER  : "0" 이면 비활성화 (None 반환) → 기존 top_k=3 검색으로 fallback
#   RERANKER_MODEL: HuggingFace 모델 ID (기본 Qwen/Qwen3-Reranker-0.6B)
#
# 첫 실행:
#   모델은 HuggingFace Hub에서 자동 다운로드된다 (~1.5 GB).
#   ~/.cache/huggingface/hub/ 에 캐시되어 이후 실행부터는 로컬에서 로드한다.
# =============================================================================

import os
import logging
from typing import Optional

from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

logger = logging.getLogger(__name__)

# =============================================================================
# 환경변수 설정
# =============================================================================

USE_RERANKER   = os.getenv("USE_RERANKER",   "1") != "0"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")

# Qwen3-Reranker 전용 도메인 지시문.
# 일반 웹 검색 지시문 대신 한국 금융 컴플라이언스 도메인에 맞게 작성.
# 모델이 "이것은 법률 규정 검색 태스크"임을 인식하도록 한다.
TASK_INSTRUCTION = (
    "Given a compliance question about Korean financial regulations, "
    "retrieve the most relevant legal provision or case precedent that answers the question."
)


# =============================================================================
# ComplianceReranker
# =============================================================================

class ComplianceReranker(SentenceTransformerRerank):
    """
    Qwen3-Reranker용 LlamaIndex 재순위기.

    SentenceTransformerRerank를 상속해 쿼리 앞에 도메인 지시문을 자동으로 붙인다.
    지시문 형식: "Instruct: {instruction}\\nQuery: {original_query}"

    부모 클래스(SentenceTransformerRerank)가 (instructed_query, chunk_text) 쌍으로
    CrossEncoder를 호출해 관련성 점수를 계산한다.
    """

    def postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
        query_str: Optional[str] = None,
    ) -> list[NodeWithScore]:
        # 원본 쿼리 추출 (query_str 우선, 없으면 query_bundle에서)
        raw_q = query_str or (query_bundle.query_str if query_bundle else "")
        if not raw_q:
            logger.warning("[reranker] 쿼리 없음 — 재순위 생략")
            return nodes

        # 도메인 지시문 추가
        instructed_q = f"Instruct: {TASK_INSTRUCTION}\nQuery: {raw_q}"

        logger.info(f"[reranker] {len(nodes)}개 후보 재순위 중 (모델: {RERANKER_MODEL})")

        # 지시문이 붙은 쿼리로 부모 클래스 호출
        reranked = super().postprocess_nodes(
            nodes,
            query_bundle=QueryBundle(query_str=instructed_q),
        )

        logger.info(f"[reranker] 상위 {len(reranked)}개 선택 완료")
        return reranked


# =============================================================================
# 팩토리 함수
# =============================================================================

def build_reranker() -> Optional[ComplianceReranker]:
    """
    ComplianceReranker 인스턴스를 생성한다.

    USE_RERANKER=0 이면 None을 반환한다.
    None이 반환되면 tools.py의 _make_search_fn이 재순위 없이 동작한다
    (기존 top_k=3 벡터 검색만 사용).

    top_n=10으로 설정: 재순위 후 전체 점수 정렬 결과를 반환하고,
    각 lane별 최종 개수 절삭(top 3 / top 2)은 _make_search_fn에서 처리한다.

    Returns:
        ComplianceReranker 인스턴스, 또는 USE_RERANKER=0일 때 None
    """
    if not USE_RERANKER:
        logger.info("[reranker] USE_RERANKER=0 — 재순위기 비활성화")
        return None

    logger.info(f"[reranker] 모델 로딩: {RERANKER_MODEL} (첫 실행 시 HuggingFace에서 다운로드)")
    return ComplianceReranker(
        model=RERANKER_MODEL,
        top_n=10,  # 충분히 크게 설정; 실제 절삭은 _make_search_fn에서 처리
    )
