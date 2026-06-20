# =============================================================================
# ingest_lc.py
# -----------------------------------------------------------------------------
# 역할: ParsedChunk를 LangChain Document로 변환하고 Qdrant에 적재한다.
#       ingest.py(LlamaIndex)의 LangChain 대응 구현.
#
# LlamaIndex → LangChain 페이로드 비호환:
#   LlamaIndex: {"text": ..., "source_type": ..., ...}  (flat)
#   LangChain:  {"page_content": ..., "metadata": {"source_type": ..., ...}}  (nested)
#   → 기존 컬렉션(compliance_agent)을 그대로 쓸 수 없다. 별도 컬렉션 사용 필수.
#
# 컬렉션명 기본값: compliance_agent_lc
#   (QDRANT_COLLECTION_LC 환경변수로 오버라이드 가능)
#
# 메타데이터 필터 키 주의:
#   langchain_qdrant는 필드를 metadata.* 하위에 저장한다.
#   따라서 페이로드 인덱스도 "metadata.source_type" 등으로 생성해야 한다.
#
# 임베딩 전략:
#   page_content(chunk.text)만 임베딩한다 (클린 텍스트 전략).
#   검색 품질이 LlamaIndex 버전보다 낮으면 page_content에 source_name 헤더를 추가한다.
# =============================================================================

import logging
import os
import uuid

from langchain_core.documents import Document

from .parser import ParsedChunk

logger = logging.getLogger(__name__)

# =============================================================================
# 교체 지점
# =============================================================================

EMBEDDING_MODEL_NAME   = os.getenv("EMBEDDING_MODEL",      "qwen3-embedding:0.6b")
QDRANT_URL             = os.getenv("QDRANT_URL",           "http://localhost:6333")
QDRANT_COLLECTION_LC   = os.getenv("QDRANT_COLLECTION_LC", "compliance_agent_lc")
QDRANT_API_KEY         = os.getenv("QDRANT_API_KEY",       "")
USE_QDRANT             = os.getenv("USE_QDRANT",           "1") != "0"
QDRANT_VECTOR_DIM      = int(os.getenv("QDRANT_VECTOR_DIM", "1024"))

# langchain_qdrant가 메타데이터를 metadata.* 하위에 저장하므로 인덱스 키도 동일하게
PAYLOAD_INDEX_FIELDS = [
    "metadata.source_type",
    "metadata.source_name",
    "metadata.citation_id",
    "metadata.article_no",
    "metadata.case_no",
]


# =============================================================================
# ParsedChunk → LangChain Document 변환
# =============================================================================

def chunk_to_document(chunk: ParsedChunk) -> Document:
    """
    ParsedChunk를 LangChain Document로 변환한다.

    page_content:
      chunk.text 만 사용 (클린 텍스트 임베딩 전략).
      LlamaIndex는 source_type/source_name/section_no를 임베딩 텍스트에 자동 주입하지만
      LangChain은 page_content를 그대로 임베딩한다.
      → 검색 품질 비교 후 필요하면 "{source_name}\\n{chunk.text}" 로 교체.

    metadata:
      tools_lc.py의 필터(metadata.source_type)와 evidence.py의 dict 계약이
      동일한 필드명을 사용하도록 맞춘다.
    """
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.doc_id))  # 결정적 UUID
    return Document(
        id=doc_id,
        page_content=chunk.text,
        metadata={
            "source_type": chunk.source_type,
            "source_name": chunk.source_name,
            "citation_id": chunk.citation_id or "",
            "article_no":  chunk.article_no  or "",
            "section_no":  chunk.section_no  or "",
            "case_no":     chunk.case_no     or "",
            "url":         chunk.url,
        },
    )


# =============================================================================
# Qdrant 유틸
# =============================================================================

def _ensure_collection(client, collection_name: str, models) -> None:
    """
    컬렉션이 없으면 TurboQuantization BITS4 + 코사인 거리로 생성한다.
    ingest.py의 build_vector_index와 동일한 설정 (재사용).
    """
    if client.collection_exists(collection_name):
        logger.info(f"[qdrant] 기존 컬렉션 사용: {collection_name}")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=QDRANT_VECTOR_DIM,
            distance=models.Distance.COSINE,
        ),
        quantization_config=models.TurboQuantization(
            turbo=models.TurboQuantQuantizationConfig(
                always_ram=True,
                bits=models.TurboQuantBitSize.BITS4,
            )
        ),
    )
    logger.info(f"[qdrant] 컬렉션 생성 완료 (TurboQuant BITS4): {collection_name}")


def _ensure_payload_indexes(client, collection_name: str, models) -> None:
    """
    MetadataFilter에 사용되는 모든 필드에 Qdrant 페이로드 인덱스를 생성한다.
    langchain_qdrant는 "metadata.*" 경로를 사용하므로 인덱스도 동일한 경로로 생성.
    """
    info = client.get_collection(collection_name)
    existing = set(info.payload_schema.keys()) if info.payload_schema else set()
    for field in PAYLOAD_INDEX_FIELDS:
        if field not in existing:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info(f"[qdrant] 페이로드 인덱스 생성: {field}")


# =============================================================================
# 전체 적재 파이프라인
# =============================================================================

def ingest(chunks: list[ParsedChunk]) -> None:
    """
    ParsedChunk 리스트를 LangChain Document로 변환하고 Qdrant에 적재한다.

    langchain_qdrant.QdrantVectorStore.add_documents()가 임베딩+적재를 담당한다.
    ingest.py와 달리 VectorStoreIndex를 반환하지 않는다 — 쿼리 경로는 load_store()로 로드.
    """
    from qdrant_client import QdrantClient
    from qdrant_client import models
    from langchain_qdrant import QdrantVectorStore
    from langchain_ollama import OllamaEmbeddings

    docs = [chunk_to_document(c) for c in chunks]
    ids  = [doc.id for doc in docs]
    logger.info(f"{len(docs)}개 문서 적재 시작 → 컬렉션: {QDRANT_COLLECTION_LC}")

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
    )

    _ensure_collection(client, QDRANT_COLLECTION_LC, models)
    _ensure_payload_indexes(client, QDRANT_COLLECTION_LC, models)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_LC,
        embedding=embeddings,
    )

    # add_documents: 임베딩 + 업서트. ids를 전달하면 결정적 point ID로 저장된다.
    store.add_documents(docs, ids=ids)
    logger.info(f"[qdrant] {len(docs)}개 문서 적재 완료: {QDRANT_COLLECTION_LC}")


# =============================================================================
# 쿼리 경로: 기존 컬렉션 로드
# =============================================================================

def load_store() -> "QdrantVectorStore":
    """
    기존 Qdrant 컬렉션에서 QdrantVectorStore를 로드한다. 재임베딩 없음.

    query 모드 전용. run_ingest_lc.py를 먼저 실행해야 한다.
    """
    from qdrant_client import QdrantClient
    from langchain_qdrant import QdrantVectorStore
    from langchain_ollama import OllamaEmbeddings

    if not USE_QDRANT:
        raise RuntimeError(
            "USE_QDRANT=0: query 모드는 Qdrant가 필요합니다. "
            "먼저 run_ingest_lc.py를 실행하거나 USE_QDRANT=1로 설정하세요."
        )

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)

    if not client.collection_exists(QDRANT_COLLECTION_LC):
        raise RuntimeError(
            f"Qdrant 컬렉션 '{QDRANT_COLLECTION_LC}'이 없습니다. "
            "python data/run_ingest_lc.py를 먼저 실행하세요."
        )

    count = client.count(QDRANT_COLLECTION_LC).count
    if count == 0:
        raise RuntimeError(
            f"Qdrant 컬렉션 '{QDRANT_COLLECTION_LC}'이 비어 있습니다. "
            "python data/run_ingest_lc.py를 실행하세요."
        )

    logger.info(f"[qdrant] 기존 컬렉션 로드: {QDRANT_COLLECTION_LC} ({count}개 벡터)")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_LC,
        embedding=embeddings,
    )
    logger.info("[qdrant] QdrantVectorStore 로드 완료 (재임베딩 없음)")
    return store
