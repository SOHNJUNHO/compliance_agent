# =============================================================================
# tools_lc.py
# -----------------------------------------------------------------------------
# 역할: LangChain 기반 벡터 검색 함수를 정의하고 ToolRegistry로 관리한다.
#       tools.py(LlamaIndex)의 LangChain 대응 구현.
#
# LlamaIndex → LangChain 매핑:
#   VectorStoreIndex.as_retriever(filters=MetadataFilters(...))
#     → QdrantVectorStore.as_retriever(search_kwargs={"k":..., "filter": Filter(...)})
#   reranker.postprocess_nodes(nodes, query_str=...)
#     → ContextualCompressionRetriever(base_compressor=CrossEncoderReranker, ...)
#   TextNode.text / .metadata  →  Document.page_content / .metadata
#
# 핵심 설계:
#   - source_type 필터가 코드에 하드코딩됨 (LLM이 잘못된 필터를 쓸 수 없음).
#   - 반환 형식(list[dict])을 tools.py와 동일하게 유지 → evidence.py가 변경 없이 재사용.
#   - qdrant Filter를 직접 사용해 metadata 중첩 키(metadata.source_type)를 필터링.
#     langchain_qdrant는 페이로드를 {"page_content": ..., "metadata": {...}} 로 저장하므로
#     필터 키는 반드시 "metadata.source_type" 형식이어야 한다.
# =============================================================================

import logging
from collections.abc import Awaitable, Callable
from typing import Optional

from langchain_qdrant import QdrantVectorStore

logger = logging.getLogger(__name__)

SearchFn = Callable[..., Awaitable[list[dict]]]


# =============================================================================
# 벡터 검색 함수 팩토리
# =============================================================================

def _make_search_fn(
    store: QdrantVectorStore,
    source_type: str,          # 하드코딩 필터 값 ("사규" | "법규" | "분쟁사례")
    top_k: int = 3,            # 초기 벡터 검색 후보 수
    compressor=None,           # CrossEncoderReranker 인스턴스 (없으면 None)
    final_top_n: int = None,   # 재순위 후 최종 반환 수 (None이면 top_k)
) -> SearchFn:
    """
    source_type 필터가 고정된 LangChain 기반 벡터 검색 함수를 생성한다.

    재순위(compressor) 유무에 따른 동작:
      compressor=None  : 벡터 유사도 상위 top_k개 바로 반환
      compressor 있음  : ContextualCompressionRetriever로 래핑 →
                         벡터 검색 top_k개를 먼저 가져온 뒤 cross-encoder로 재순위,
                         CrossEncoderReranker.top_n으로 절삭

    반환 형식 (tools.py와 동일):
      [{"text", "source_type", "source_name", "citation_id",
        "article_no", "section_no", "case_no"}, ...]
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from langchain.retrievers import ContextualCompressionRetriever

    # source_type 메타데이터 필터 (langchain_qdrant는 metadata.*로 중첩 접근)
    qdrant_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.source_type",
                match=MatchValue(value=source_type),
            )
        ]
    )

    base_retriever = store.as_retriever(
        search_kwargs={"k": top_k, "filter": qdrant_filter}
    )

    if compressor is not None:
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
    else:
        retriever = base_retriever

    async def search_fn(query: str) -> list[dict]:
        """
        쿼리 텍스트로 관련 문서를 검색한다.

        Args:
            query: 검색할 질문 또는 키워드

        Returns:
            관련 문서 청크 목록 (최대 top_k/final_top_n개, source_type 필터 적용)
        """
        docs = await retriever.ainvoke(query)

        # compressor 없을 때 final_top_n으로 수동 절삭 (ContextualCompression은 자체 절삭)
        if compressor is None and final_top_n is not None:
            docs = docs[:final_top_n]

        results = []
        for doc in docs:
            meta = doc.metadata
            results.append({
                "text":        doc.page_content,
                "source_type": meta.get("source_type", ""),
                "source_name": meta.get("source_name", ""),
                "citation_id": meta.get("citation_id", ""),
                "article_no":  meta.get("article_no", ""),
                "section_no":  meta.get("section_no", ""),
                "case_no":     meta.get("case_no", ""),
            })
        return results

    return search_fn


# =============================================================================
# ToolRegistry: 워크플로우가 사용하는 검색 함수의 단일 관리 지점
# =============================================================================

class ToolRegistry:
    """
    3개 검색 함수를 보관하고 워크플로우에 제공하는 레지스트리.
    tools.py의 ToolRegistry와 동일한 인터페이스를 유지한다.

    사용 흐름:
      1. ingest_lc.py가 QdrantVectorStore를 생성
      2. tool_registry_lc.build(store) 호출 → 3개 함수 초기화
      3. 그래프 노드에서 tool_registry_lc.regulation_search(...) 로 호출
    """

    def __init__(self):
        self.regulation_search: Optional[SearchFn] = None
        self.law_search:        Optional[SearchFn] = None
        self.case_search:       Optional[SearchFn] = None

    def build(self, store: QdrantVectorStore, reranker=None) -> None:
        """
        QdrantVectorStore를 받아 3개 검색 함수를 초기화한다.

        Args:
            store:    QdrantVectorStore (langchain_qdrant)
            reranker: LaneRerankers 인스턴스 (없으면 None → 재순위 생략)
                      재순위 시 레인별 CrossEncoderReranker가 compressor로 전달된다.
        """
        # reranker 유무에 따라 초기 후보 수 결정 (tools.py와 동일 비율)
        initial_reg_law_k = 10 if reranker else 3
        initial_case_k    = 5  if reranker else 2

        self.regulation_search = _make_search_fn(
            store=store,
            source_type="사규",
            top_k=initial_reg_law_k,
            compressor=reranker.regulation if reranker else None,
            final_top_n=3,
        )
        self.law_search = _make_search_fn(
            store=store,
            source_type="법규",
            top_k=initial_reg_law_k,
            compressor=reranker.law if reranker else None,
            final_top_n=3,
        )
        self.case_search = _make_search_fn(
            store=store,
            source_type="분쟁사례",
            top_k=initial_case_k,
            compressor=reranker.case if reranker else None,
            final_top_n=2,
        )

        mode = "재순위 활성화" if reranker else "벡터 검색만"
        logger.info(f"ToolRegistry(LC) 초기화 완료 (검색 함수 3개, {mode})")


# 싱글턴 인스턴스
tool_registry_lc = ToolRegistry()
