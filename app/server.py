# =============================================================================
# app/server.py
# -----------------------------------------------------------------------------
# 역할: ECS Fargate 프로덕션 서빙 레이어.
#
# 엔드포인트:
#   POST /query        — 컴플라이언스 Q&A 실행; FinalAnswer JSON 반환
#   GET  /healthz      — 컨테이너 liveness (ALB + ECS 헬스체크)
#   GET  /readyz       — 애플리케이션 readiness (인덱스·vLLM·Qdrant 확인)
#
# 실행:
#   uv run uvicorn app.server:app --host 0.0.0.0 --port 8000
#
# 설계 원칙:
#   - 인덱스 로드 / 재순위기 / 도구 레지스트리 / LLM 인스턴스는 lifespan 시작 시
#     1회만 초기화하고 모든 요청이 재사용한다. (main.py와 달리 요청마다 재초기화하지 않음)
#   - Langfuse @observe + propagate_attributes 패턴은 main.py와 동일하게 유지한다.
#   - secrets은 ECS 태스크 정의에서 환경변수로 주입된다 (AWS Secrets Manager 연동).
#
# 환경변수 (필수):
#   LLM_API_BASE      — vLLM chat 서버 URL (예: http://10.0.1.5:8000/v1)
#   LLM_MODEL         — vLLM served-model-name (예: Qwen3-8B)
#   LLM_API_KEY       — vLLM API 키 (기본 "EMPTY")
#   EMBED_API_BASE    — vLLM/TEI 임베딩 서버 URL (예: http://10.0.1.5:8001/v1)
#   EMBED_MODEL       — 임베딩 모델명 (예: qwen3-embedding-0.6B)
#   EMBED_API_KEY     — 임베딩 API 키 (기본 "EMPTY")
#   QDRANT_URL        — Qdrant Cloud URL
#   QDRANT_API_KEY    — Qdrant Cloud API 키 (Secrets Manager)
#   QDRANT_COLLECTION — Qdrant 컬렉션명
#   LANGFUSE_PUBLIC_KEY  — Langfuse 공개 키
#   LANGFUSE_SECRET_KEY  — Langfuse 비밀 키 (Secrets Manager)
#   LANGFUSE_BASE_URL    — Langfuse 서버 주소
# =============================================================================

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()  # .env 파일에서 환경변수 로드 (없으면 무시)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Langfuse + auto-instrumentation (main.py와 동일)
from langfuse import observe, get_client, propagate_attributes
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
LlamaIndexInstrumentor().instrument()


# =============================================================================
# 요청/응답 스키마
# =============================================================================

class QueryRequest(BaseModel):
    """POST /query 요청 본문."""
    query: str
    user_id: str = "anonymous"


class QueryResponse(BaseModel):
    """POST /query 응답 — FinalAnswer의 직렬화 가능한 필드."""
    reasoning:         str
    cited_ids:         list[str]
    cited_passages:    list[dict]
    agents_used:       list[str]
    cited_agents:      list[str]
    routing_reasoning: str


# =============================================================================
# 애플리케이션 상태 (요청 간 공유)
# =============================================================================

class _AppState:
    """lifespan에서 초기화되고 요청 핸들러가 읽는 싱글턴 컨테이너."""
    llm = None
    workflow_cls = None
    registry = None
    ready: bool = False


_state = _AppState()


# =============================================================================
# Lifespan: 시작 시 1회 초기화
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan 이벤트 핸들러.

    startup:
      1. LLM 인스턴스 생성 (OpenAILike → vLLM)
      2. Qdrant에서 VectorStoreIndex 로드 (재임베딩 없음)
      3. 재순위기 초기화 (bge-reranker-v2-m3 로딩 ~2.2 GB)
      4. ToolRegistry 빌드
      5. Langfuse 프롬프트 동기화 (LANGFUSE_SYNC_PROMPTS=1 시에만)

    shutdown:
      Langfuse flush
    """
    logger.info("[startup] 애플리케이션 초기화 시작")

    # ── 1. LLM 인스턴스 ─────────────────────────────────────────────────────
    llm_api_base = os.getenv("LLM_API_BASE", "")
    llm_model    = os.getenv("LLM_MODEL",    "Qwen3-8B")
    llm_api_key  = os.getenv("LLM_API_KEY",  "EMPTY")

    if not llm_api_base:
        raise RuntimeError(
            "LLM_API_BASE 환경변수가 설정되지 않았습니다. "
            "vLLM 서버 URL을 설정하거나, 로컬 개발은 main.py를 사용하세요."
        )

    from llama_index.llms.openai_like import OpenAILike
    _state.llm = OpenAILike(
        model=llm_model,
        api_base=llm_api_base,
        api_key=llm_api_key,
        is_chat_model=True,
        is_function_calling_model=True,  # astructured_predict → tool-calling 경로
        request_timeout=120.0,
    )
    logger.info(f"[startup] LLM 초기화 완료: {llm_model} @ {llm_api_base}")

    # ── 2. VectorStoreIndex 로드 ─────────────────────────────────────────────
    from data.ingest import load_index
    index = load_index()
    logger.info("[startup] VectorStoreIndex 로드 완료")

    # ── 3. 재순위기 초기화 ───────────────────────────────────────────────────
    from workflow.reranker import build_reranker
    reranker = build_reranker()
    logger.info(f"[startup] 재순위기 초기화 완료: {'활성' if reranker else '비활성'}")

    # ── 4. ToolRegistry 빌드 ─────────────────────────────────────────────────
    from workflow.tools import tool_registry
    tool_registry.build(index, reranker=reranker)
    _state.registry = tool_registry
    logger.info("[startup] ToolRegistry 빌드 완료")

    # ── 5. Langfuse 프롬프트 동기화 (선택적) ─────────────────────────────────
    if os.getenv("LANGFUSE_SYNC_PROMPTS") == "1":
        from workflow.langfuse_setup import sync_prompts
        sync_prompts()
        logger.info("[startup] Langfuse 프롬프트 동기화 완료")

    # ── 워크플로우 클래스 준비 ───────────────────────────────────────────────
    from workflow.compliance_workflow import ComplianceWorkflow
    _state.workflow_cls = ComplianceWorkflow
    _state.ready = True
    logger.info("[startup] 초기화 완료 — 요청 수신 준비")

    yield  # ── 서비스 실행 구간 ──────────────────────────────────────────────

    # ── Shutdown: Langfuse flush ─────────────────────────────────────────────
    logger.info("[shutdown] Langfuse 이벤트 flush")
    get_client().flush()


# =============================================================================
# FastAPI 앱
# =============================================================================

app = FastAPI(
    title="Compliance Q&A API",
    version="1.0.0",
    description="증권사 컴플라이언스 Q&A 멀티에이전트 워크플로우 — REST API",
    lifespan=lifespan,
)


# =============================================================================
# 헬스체크 엔드포인트
# =============================================================================

@app.get("/healthz", tags=["health"])
async def healthz():
    """
    컨테이너 liveness 프로브.

    ECS 태스크 정의 및 ALB 타겟 그룹 헬스체크가 이 엔드포인트를 사용한다.
    프로세스가 살아 있으면 200을 반환한다. 무거운 검사는 /readyz에서 수행한다.
    """
    return {"status": "ok"}


@app.get("/readyz", tags=["health"])
async def readyz():
    """
    애플리케이션 readiness 프로브.

    모든 컴포넌트(인덱스·LLM·Qdrant)가 준비된 경우에만 200을 반환한다.
    lifespan 초기화가 완료되지 않으면 503을 반환한다.
    """
    if not _state.ready:
        raise HTTPException(status_code=503, detail="initializing")

    # vLLM 헬스체크: /health 엔드포인트 확인
    import httpx
    llm_api_base = os.getenv("LLM_API_BASE", "")
    if llm_api_base:
        vllm_health_url = llm_api_base.rstrip("/v1").rstrip("/") + "/health"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(vllm_health_url)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=503,
                    detail=f"vLLM 서버 응답 비정상: {resp.status_code}",
                )
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"vLLM 서버 연결 실패: {e}")

    return {"status": "ready"}


# =============================================================================
# POST /query — 핵심 엔드포인트
# =============================================================================

@observe(name="compliance_query", as_type="span", capture_input=False, capture_output=False)
async def _run_query(query: str, user_id: str) -> QueryResponse:
    """
    워크플로우를 실행하고 QueryResponse를 반환한다.

    Langfuse @observe 데코레이터는 이 함수를 루트 span으로 래핑한다.
    main.py:run_query와 동일한 관측 패턴을 사용한다.
    """
    client = get_client()
    client.update_current_span(input={"query": query})

    with propagate_attributes(
        user_id=user_id,
        session_id=user_id,
        trace_name=f"compliance: {query[:60]}",
        tags=["compliance-query", "api"],
    ):
        wf = _state.workflow_cls(
            llm=_state.llm,
            registry=_state.registry,
            timeout=300,
            verbose=False,
        )

        handler = wf.run(query=query)
        result = await handler

        if not hasattr(result, "reasoning"):
            raise HTTPException(status_code=500, detail=f"워크플로우 결과 이상: {result}")

        response = QueryResponse(
            reasoning=result.reasoning,
            cited_ids=result.cited_ids,
            cited_passages=result.cited_passages,
            agents_used=result.agents_used,
            cited_agents=result.cited_agents,
            routing_reasoning=result.routing_reasoning,
        )

        client.update_current_span(
            output={
                "answer": result.reasoning,
                "cited_ids": result.cited_ids,
            },
            metadata={
                "agents_used": str(result.agents_used),
            },
        )

        return response


@app.post("/query", response_model=QueryResponse, tags=["compliance"])
async def query_endpoint(req: QueryRequest):
    """
    컴플라이언스 Q&A 실행.

    요청:
      { "query": "설명의무 위반으로 손해배상 책임이 인정된 판례가 있나요?", "user_id": "cust-001" }

    응답:
      {
        "reasoning": "...",
        "cited_ids": ["사규-표준투자권유준칙-제5조", ...],
        "cited_passages": [{"evidence_id": "...", "text": "..."}],
        "agents_used": ["법규", "사례"],
        "cited_agents": ["사례"],
        "routing_reasoning": "판례 검색이 필요하여 사례 에이전트를 활성화했습니다."
      }
    """
    return await _run_query(req.query, req.user_id)
