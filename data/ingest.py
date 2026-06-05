# =============================================================================
# ingest.py
# -----------------------------------------------------------------------------
# 역할: ParsedChunk를 LlamaIndex TextNode로 변환하고 Vector DB에 적재한다.
#       이 파일은 데이터 파이프라인의 마지막 단계이다.
#
# 데이터 흐름:
#   scraper.py → parser.py → [ingest.py] → Vector DB
#                                         → article_lookup.json
#
# 두 가지 인덱스를 생성한다:
#   1. VectorStoreIndex: 벡터 유사도 검색용 (tools.py의 search 툴들이 사용)
#   2. article_lookup.json: 표준 인용 ID exact match 검색용 (factcheck_step이 사용)
#
# 교체 지점:
#   EMBEDDING_MODEL: Ollama 임베딩 모델 (기본 qwen3-embedding:0.6b)
#   QDRANT_URL:      Qdrant 서버 주소 (기본 http://localhost:6333)
#   QDRANT_COLLECTION: Qdrant 컬렉션명 (기본 compliance_agents)
#   USE_QDRANT:      Qdrant 사용 여부 (0이면 인메모리 VectorStore 사용)
# =============================================================================

import os
import json
import logging
import uuid
from pathlib import Path

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode

# OllamaEmbedding: Ollama에서 실행 중인 Qwen embedding 모델을 사용
from llama_index.embeddings.ollama import OllamaEmbedding

from parser import ParsedChunk

logger = logging.getLogger(__name__)

# =============================================================================
# 교체 지점
# =============================================================================
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
EMBEDDING_MODEL = OllamaEmbedding(model_name=EMBEDDING_MODEL_NAME)

QDRANT_URL        = os.getenv("QDRANT_URL",        "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "compliance_agents")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY",    "")   # required for Qdrant Cloud; empty = local (no auth)
USE_QDRANT        = os.getenv("USE_QDRANT",        "1") != "0"
QDRANT_VECTOR_DIM = int(os.getenv("QDRANT_VECTOR_DIM", "1024"))  # qwen3-embedding:0.6b output dim

# Fields used in MetadataFilter — Qdrant Cloud requires a payload index on each
PAYLOAD_INDEX_FIELDS = ["source_type", "source_name", "citation_id", "article_no", "case_no", "category"]

# article_lookup.json 저장 경로
LOOKUP_INDEX_PATH = Path("data/article_lookup.json")


# =============================================================================
# ParsedChunk → TextNode 변환
# =============================================================================

def chunk_to_node(chunk: ParsedChunk) -> TextNode:
    """
    ParsedChunk를 LlamaIndex TextNode로 변환한다.

    TextNode 구조:
      text:     임베딩 대상 본문 (벡터 검색의 실질 내용)
      id_:      고유 ID (중복 적재 방지)
      metadata: 필터링·인용에 사용할 메타데이터

    metadata 설계 포인트:
      - source_type: tools.py의 MetadataFilter가 이 값으로 청크를 필터링
        예) regulation_search → source_type="사규" 인 청크만 반환
      - keywords: 리스트를 문자열로 직렬화 (Vector DB 호환성)
        일부 DB는 메타데이터 값으로 리스트를 지원하지 않음

    excluded_embed_metadata_keys:
      임베딩 시 메타데이터 일부를 텍스트에 포함시키지 않을 필드 지정
      url, article_no, case_no는 검색 정확도와 무관하므로 제외
      → 임베딩 품질 향상

    excluded_llm_metadata_keys:
      LLM에 전달할 때 메타데이터에서 제외할 필드
      url은 LLM이 직접 방문할 수 없으므로 컨텍스트에서 제외
    """
    return TextNode(
        text=chunk.text,        # 벡터화 대상 본문
        id_=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.doc_id)),  # UUID (Qdrant requires UUID or uint)

        metadata={
            # ── 에이전트 라우팅 핵심 필드 ──
            "source_type": chunk.source_type,   # "사규" | "법규" | "분쟁사례"

            # ── 답변 인용에 사용되는 필드 ──
            "source_name": chunk.source_name,   # "표준투자권유준칙"
            "citation_id": chunk.citation_id or "",  # "제5조" | "2.2.1" | 사건번호
            "article_no":  chunk.article_no or "",  # "제5조" (없으면 빈 문자열)
            "article_title": chunk.article_title or "",
            "section_no": chunk.section_no or "",
            "section_title": chunk.section_title or "",
            "case_no":     chunk.case_no or "",     # "2022-증권-031"
            "url":         chunk.url,
            "chunk_id":     chunk.doc_id,
            "verified":     chunk.verified,

            # ── 검색 품질 보조 필드 ──
            "category":    chunk.category,                   # "적합성원칙"
            "keywords":    ", ".join(chunk.keywords),         # 리스트→쉼표 구분 문자열
        },

        # 임베딩 시 텍스트에 포함하지 않을 메타데이터 키
        excluded_embed_metadata_keys=[
            "url", "citation_id", "article_no", "case_no", "chunk_id", "verified",
        ],

        # LLM 컨텍스트에 전달하지 않을 메타데이터 키
        excluded_llm_metadata_keys=["url"],
    )


# =============================================================================
# VectorStoreIndex 구성
# =============================================================================

def build_vector_index(chunks: list[ParsedChunk]) -> VectorStoreIndex:
    """
    ParsedChunk 리스트로 VectorStoreIndex를 구성한다.

    VectorStoreIndex 동작:
      1. 각 TextNode의 text를 EMBEDDING_MODEL로 임베딩 (벡터화)
      2. 벡터와 메타데이터를 Qdrant 또는 인메모리 VectorStore에 저장
      3. 검색 시 쿼리 벡터와 저장된 벡터의 코사인 유사도를 계산

    기본은 Qdrant(localhost:6333)이며, qdrant 관련 패키지나 서버가 없으면
    인메모리 저장소로 fallback한다.
    """
    # 모든 청크를 TextNode로 변환
    nodes = [chunk_to_node(c) for c in chunks]
    logger.info(f"{len(nodes)}개 노드 인덱싱 시작")

    vector_store = None
    if USE_QDRANT:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client import models as qdrant_models
            from llama_index.vector_stores.qdrant import QdrantVectorStore

            client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY or None,  # None = unauthenticated (local Docker)
            )
            logger.info(f"Qdrant 사용: {QDRANT_URL} / collection={QDRANT_COLLECTION}")

            # Pre-create collection with INT8 scalar quantization if it does not exist.
            # LlamaIndex respects a pre-created collection and will not overwrite it.
            # To apply quantization to an existing collection, delete it in the Qdrant
            # console first, then re-run ingest.
            if not client.collection_exists(QDRANT_COLLECTION):
                client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=qdrant_models.VectorParams(
                        size=QDRANT_VECTOR_DIM,
                        distance=qdrant_models.Distance.COSINE,
                    ),
                    quantization_config=qdrant_models.ScalarQuantization(
                        scalar=qdrant_models.ScalarQuantizationConfig(
                            type=qdrant_models.ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True,
                        ),
                    ),
                )
                logger.info(f"[qdrant] 컬렉션 생성 완료 (INT8 스칼라 양자화): {QDRANT_COLLECTION}")
            else:
                logger.info(f"[qdrant] 기존 컬렉션 사용: {QDRANT_COLLECTION}")

            _ensure_payload_indexes(client, QDRANT_COLLECTION, qdrant_models)

            vector_store = QdrantVectorStore(
                client=client,
                collection_name=QDRANT_COLLECTION,
            )
        except Exception as e:
            logger.warning(f"Qdrant 초기화 실패, 인메모리 VectorStore 사용: {e}")

    if vector_store is not None:
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_ctx,
            embed_model=EMBEDDING_MODEL,
        )
    else:
        # 인메모리 (개발/데모용): StorageContext 불필요
        index = VectorStoreIndex(nodes, embed_model=EMBEDDING_MODEL)

    logger.info("VectorStoreIndex 구성 완료")
    return index


# =============================================================================
# article_lookup.json 구성
# =============================================================================

def build_lookup_index(chunks: list[ParsedChunk]) -> None:
    """
    factcheck_step 전용 exact-match 조회 인덱스를 JSON 파일로 저장한다.

    왜 별도 인덱스가 필요한가:
      factcheck_step은 "표준투자권유준칙 제5조가 실제로 존재하는가"를
      확인해야 한다. 벡터 유사도 검색은 "비슷한" 문서를 찾는 것이므로
      존재 여부 확인에 적합하지 않다.
      → 키-값 딕셔너리로 O(1) exact match 조회

    키 형식: "{source_name}||{citation_id}"
      예: "표준투자권유준칙||제5조", "금융투자회사표준내부통제기준||2.2.1"
      || 구분자: source_name이나 citation_id에 _가 포함될 수 있으므로
                겹치지 않는 구분자 사용

    citation_id가 있는 모든 청크를 포함한다.
    """
    lookup: dict[str, dict] = {}

    for chunk in chunks:
        if chunk.citation_id:
            key = f"{chunk.source_name}||{chunk.citation_id}"
            lookup[key] = {
                "doc_id":      chunk.doc_id,
                "source_type": chunk.source_type,
                "source_name": chunk.source_name,
                "citation_id": chunk.citation_id,
                "article_no": chunk.article_no,
                "article_title": chunk.article_title,
                "section_no": chunk.section_no,
                "section_title": chunk.section_title,
                "case_no": chunk.case_no,
                "url": chunk.url,
                "chunk_id": chunk.doc_id,
                "verified": True,
                "text": chunk.text,
            }

    # JSON 파일로 저장
    LOOKUP_INDEX_PATH.parent.mkdir(exist_ok=True)
    with open(LOOKUP_INDEX_PATH, "w", encoding="utf-8") as f:
        # ensure_ascii=False: 한글을 유니코드 이스케이프 없이 저장
        # indent=2: 사람이 읽기 좋은 형태로 저장
        json.dump(lookup, f, ensure_ascii=False, indent=2)

    logger.info(f"article_lookup 인덱스 저장: {len(lookup)}개 항목 → {LOOKUP_INDEX_PATH}")


# =============================================================================
# 전체 적재 파이프라인 진입점
# =============================================================================

def _ensure_payload_indexes(client, collection_name: str, qdrant_models) -> None:
    """
    MetadataFilter에 사용되는 모든 필드에 Qdrant 페이로드 인덱스를 생성한다.

    Qdrant Cloud는 필터 대상 필드에 인덱스가 없으면 HTTP 400을 반환한다.
    이미 존재하는 인덱스는 건너뛰므로 반복 호출해도 안전하다.
    """
    info = client.get_collection(collection_name)
    existing = set(info.payload_schema.keys()) if info.payload_schema else set()
    for field in PAYLOAD_INDEX_FIELDS:
        if field not in existing:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
            logger.info(f"[qdrant] 페이로드 인덱스 생성: {field}")


def load_index() -> VectorStoreIndex:
    """
    기존 Qdrant 컬렉션에서 VectorStoreIndex를 로드한다. 재임베딩 없음.

    query 모드 전용. python run_ingest.py를 먼저 실행해야 한다.
    LlamaIndex가 쿼리 텍스트만 임베딩하고, 저장된 벡터는 그대로 사용한다.
    """
    if not USE_QDRANT:
        raise RuntimeError(
            "USE_QDRANT=0: query 모드는 Qdrant가 필요합니다. "
            "먼저 python run_ingest.py를 실행하거나 USE_QDRANT=1로 설정하세요."
        )

    from qdrant_client import QdrantClient
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)

    if not client.collection_exists(QDRANT_COLLECTION):
        raise RuntimeError(
            f"Qdrant 컬렉션 '{QDRANT_COLLECTION}'이 없습니다. "
            "python run_ingest.py를 먼저 실행하세요."
        )

    count = client.count(QDRANT_COLLECTION).count
    if count == 0:
        raise RuntimeError(
            f"Qdrant 컬렉션 '{QDRANT_COLLECTION}'이 비어 있습니다. "
            "python run_ingest.py를 실행하세요."
        )

    logger.info(f"[qdrant] 기존 컬렉션 로드: {QDRANT_COLLECTION} ({count}개 벡터)")
    vector_store = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=EMBEDDING_MODEL)
    logger.info("[qdrant] VectorStoreIndex 로드 완료 (재임베딩 없음)")
    return index


def ingest(chunks: list[ParsedChunk]) -> VectorStoreIndex:
    """
    파이프라인의 최종 단계. 두 인덱스를 모두 구성한다.

    Args:
        chunks: parser.parse_all()의 반환값

    Returns:
        VectorStoreIndex (tools.py의 ToolRegistry.build()에 전달)
    """
    index = build_vector_index(chunks)   # 1. 벡터 인덱스
    build_lookup_index(chunks)           # 2. exact-match 인덱스
    return index
