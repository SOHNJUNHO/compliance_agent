# =============================================================================
# tools.py
# -----------------------------------------------------------------------------
# 역할: 워크플로우가 사용하는 검색/조회 함수를 정의하고 ToolRegistry로 관리한다.
#
# 함수 4개:
#   regulation_search : source_type="사규" 고정 벡터 검색
#   law_search        : source_type="법규" 고정 벡터 검색
#   case_search       : source_type="분쟁사례" 고정 벡터 검색
#   article_lookup    : 표준 인용 ID exact match (Qdrant 페이로드 기반)
#
# 핵심 설계: source_type 필터 하드코딩
#   각 검색 함수는 자신의 source_type 필터가 코드에 고정되어 있다.
#   → 에이전트(LLM)가 "잘못된 필터"를 쓸 수 없음 (strict control)
#   → regulation_search는 항상 사규만, law_search는 항상 법규만 반환
# =============================================================================

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Optional

from llama_index.core import VectorStoreIndex

# MetadataFilter: 벡터 검색 시 메타데이터 조건 필터링
# FilterOperator.EQ: 등호 조건 (source_type == "사규")
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 벡터 검색 함수 팩토리
# =============================================================================

SearchFn = Callable[..., Awaitable[list[dict]]]
LookupFn = Callable[[str, str], Optional[dict]]


def _make_search_fn(
    index: VectorStoreIndex,
    source_type: str,         # 이 값이 하드코딩되어 필터로 사용됨
    top_k: int = 3,           # 초기 벡터 검색 후보 수 (재순위 시 더 크게 설정됨)
    reranker=None,            # ComplianceReranker 인스턴스 (없으면 None)
    final_top_n: int = None,  # 재순위 후 최종 반환 수 (None이면 top_k 그대로)
) -> SearchFn:
    """
    source_type 필터가 고정된 벡터 검색 함수를 생성한다.

    재순위(reranker) 유무에 따른 동작:
      reranker=None : 벡터 유사도 상위 top_k개 바로 반환 (기존 동작)
      reranker 있음 : 벡터 유사도 상위 top_k개를 먼저 가져온 뒤
                     cross-encoder로 재순위하여 상위 final_top_n개만 반환.
                     → 정밀도 향상: (쿼리, 청크) 쌍을 함께 보고 관련성 직접 예측

    반환 형식 (list[dict]):
      [
        {
          "text":        "조항 본문...",
          "source_name": "표준투자권유준칙",
          "article_no":  "제5조",
          "case_no":     "",
          "url":         "https://...",
        },
        ...
      ]
    """
    async def search_fn(query: str) -> list[dict]:
        """
        쿼리 텍스트로 관련 문서를 검색한다.

        Args:
            query: 검색할 질문 또는 키워드

        Returns:
            관련 문서 청크 목록 (최대 top_k개, source_type 필터 적용)
        """
        filter_items = [
            MetadataFilter(
                key="source_type",
                value=source_type,
                operator=FilterOperator.EQ,
            )
        ]

        retriever = index.as_retriever(
            similarity_top_k=top_k,
            filters=MetadataFilters(filters=filter_items),
        )

        # aretrieve(): 쿼리를 임베딩하고 저장된 벡터와 유사도 비교 (비동기 — 이벤트
        # 루프를 막지 않으므로 세 검색 레인이 실제로 겹쳐 실행된다)
        nodes = await retriever.aretrieve(query)

        # ── 재순위 (reranker가 있을 때만 실행) ──────────────────────────────
        # cross-encoder가 (query, chunk) 쌍을 함께 보고 관련성 재평가 → 정밀도 향상
        # postprocess_nodes는 동기 모델 추론이므로 to_thread로 루프 밖에서 실행한다.
        if reranker is not None and nodes:
            nodes = await asyncio.to_thread(
                reranker.postprocess_nodes, nodes, query_str=query
            )
            # reranker.top_n=10으로 설정되어 있으므로 여기서 최종 개수로 절삭
            cutoff = final_top_n if final_top_n is not None else top_k
            nodes = nodes[:cutoff]

        results = []
        for node in nodes:
            results.append({
                "text":        node.text,
                "source_type": node.metadata.get("source_type", ""),
                "source_name": node.metadata.get("source_name", ""),
                "citation_id": node.metadata.get("citation_id", ""),
                "article_no":  node.metadata.get("article_no", ""),
                "section_no": node.metadata.get("section_no", ""),
                "case_no":     node.metadata.get("case_no", ""),
            })
        return results

    return search_fn


# =============================================================================
# article_lookup: exact match 조회
# =============================================================================

def _make_lookup_fn() -> LookupFn:
    """
    표준 인용 ID + 출처명으로 원문을 exact match 조회하는 함수.

    validate_article_evidence 전용:
      검색 결과의 조항이 실제로 Qdrant 페이로드에 존재하는지 검증하는 데 사용.
      벡터 유사도가 아닌 딕셔너리 키 조회 → 정확한 존재 여부 확인 가능.

    Qdrant 페이로드 기반:
      별도 JSON 파일을 쓰지 않는다. 조항 원문·메타데이터는 이미 Qdrant 페이로드에
      존재하므로 ingest.load_lookup_table()이 컬렉션을 1회 scroll하여 동일한
      인메모리 딕셔너리를 구성한다. → 중복 저장(second source of truth) 제거.

    키 형식: "{source_name}||{citation_id}"
      예: "표준투자권유준칙||제5조", "금융투자회사표준내부통제기준||2.2.1"

    캐시: 첫 호출 시 1회 로드 후 클로저 변수에 보관한다.
    validate_article_evidence 단계에서 한 쿼리당 10~20회 호출되므로, 첫 호출에서만
    scroll하고 이후에는 인메모리 조회로 처리한다.
    """
    cache: dict[str, dict] | None = None

    def lookup_fn(source_name: str, citation_id: str) -> Optional[dict]:
        """
        규정명과 표준 인용 ID로 원문을 조회한다.

        Returns:
            조항 원문 딕셔너리, 없으면 None
        """
        nonlocal cache
        if cache is None:
            from data.ingest import load_lookup_table
            cache = load_lookup_table()

        key = f"{source_name}||{citation_id}"
        result = cache.get(key)
        if not result:
            logger.warning(f"조항 미존재: {key}")
        return result

    return lookup_fn


# =============================================================================
# ToolRegistry: 워크플로우가 사용하는 검색/조회 함수의 단일 관리 지점
# =============================================================================

class ToolRegistry:
    """
    4개 함수를 보관하고 워크플로우에 제공하는 레지스트리.

    사용 흐름:
      1. ingest.py가 VectorStoreIndex를 생성
      2. tool_registry.build(index) 호출 → 4개 함수 초기화
      3. ComplianceWorkflow(registry=tool_registry) 로 워크플로우에 전달
      4. 각 Step이 self.registry.regulation_search(...) 로 함수 호출

    싱글턴 패턴:
      모듈 하단에 tool_registry = ToolRegistry() 인스턴스를 생성.
      워크플로우에서 이 인스턴스를 import해서 사용.
    """

    def __init__(self):
        # 검색 함수는 build() 호출 전까지 None (인덱스가 준비되어야 초기화 가능)
        self.regulation_search: Optional[SearchFn] = None
        self.law_search:        Optional[SearchFn] = None
        self.case_search:       Optional[SearchFn] = None
        self.article_lookup:    Optional[LookupFn] = None

    def build(self, index: VectorStoreIndex, reranker=None) -> None:
        """
        VectorStoreIndex를 받아 4개 함수를 초기화한다.
        main.py에서 ingest() 후 이 메서드를 호출한다.

        Args:
            index:    VectorStoreIndex (Qdrant 또는 인메모리)
            reranker: ComplianceReranker 인스턴스 (없으면 None → 재순위 생략)
                      reranker가 있으면 초기 top_k를 크게 잡아 더 많은 후보를 수집하고,
                      cross-encoder로 재순위한 뒤 최종 개수(3/2)로 절삭한다.
        """
        # reranker 유무에 따라 초기 후보 수 결정
        # 재순위 시: 더 많은 후보(10/5)에서 cross-encoder가 최적 결과 선택
        # 재순위 없음: 기존 top_k(3/2) 그대로 사용
        initial_reg_law_k = 10 if reranker else 3
        initial_case_k    = 5  if reranker else 2

        # 사규 검색 함수: source_type="사규" 고정 필터
        self.regulation_search = _make_search_fn(
            index=index,
            source_type="사규",
            top_k=initial_reg_law_k,
            reranker=reranker,
            final_top_n=3,
        )
        # 법규 검색 함수: source_type="법규" 고정 필터
        self.law_search = _make_search_fn(
            index=index,
            source_type="법규",
            top_k=initial_reg_law_k,
            reranker=reranker,
            final_top_n=3,
        )
        # 분쟁사례 검색 함수: source_type="분쟁사례" 고정 필터
        # 사례는 본문이 길어 컨텍스트 절약을 위해 최종 2개 반환
        self.case_search = _make_search_fn(
            index=index,
            source_type="분쟁사례",
            top_k=initial_case_k,
            reranker=reranker,
            final_top_n=2,
        )
        # exact match 조회 함수: 벡터 인덱스 불필요 (Qdrant 페이로드 기반)
        self.article_lookup = _make_lookup_fn()

        mode = "재순위 활성화" if reranker else "벡터 검색만"
        logger.info(f"ToolRegistry 초기화 완료 (검색/조회 함수 4개, {mode})")


# 싱글턴 인스턴스
# compliance_workflow.py에서: from tools import tool_registry
tool_registry = ToolRegistry()
