# =============================================================================
# graph.py
# -----------------------------------------------------------------------------
# 역할: LangGraph 기반 증권사 컴플라이언스 Q&A 멀티에이전트 그래프 본체.
#       compliance_workflow.py(LlamaIndex)의 LangGraph 대응 구현.
#
# LlamaIndex Workflow vs LangGraph 핵심 차이:
#   LlamaIndex: Event 서브클래스로 Step 간 라우팅 (이벤트 타입 = DAG 엣지)
#               ctx.collect_events()로 fan-in 동기화
#               ctx.store로 Step 간 공유 상태
#   LangGraph:  단일 TypedDict(ComplianceState)가 노드 간을 흐른다.
#               operator.add reducer로 fan-in 자동 병합 (collect_events 불필요)
#               Send API로 동적 fan-out (skipped=True 센티널 불필요)
#               BSP(superstep) 실행 모델: 같은 슈퍼스텝의 노드가 모두 완료돼야
#               다음 슈퍼스텝이 시작 → synthesize 노드가 자연스럽게 대기
#
# 그래프 구조:
#   START
#     → classify        (LLM 라우팅 → agent_list, routing_reasoning)
#     → route_to_lanes  (Send 기반 동적 fan-out: 활성 레인 수만큼 search 태스크 생성)
#     → search ×N       (병렬: HyDE → 검색+재순위 → 검증 → evidence 누적)
#     → synthesize      (정렬 → per-passage LLM map → 코드 reduce → 최종 답변)
#     → END
#
# 설계 원칙 (LlamaIndex 버전과 동일):
#   금융 컴플라이언스 도메인은 LLM의 자율적 판단이 위험하다.
#   Rule이 통제하고 LLM은 해석에만 집중 = "Rule-based & LLM Hybrid"
# =============================================================================

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Optional

import httpx
import ollama
from pydantic import ValidationError

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from .state import (
    ComplianceState,
    LaneWork,
    PassageAnswer,
    ClassifyResponse,
    AgentName,
)
from .tools_lc import ToolRegistry
from .evidence import (
    validate_article_evidence,
    validate_case_evidence,
    evidence_article_labels,
    format_single_passage,
    format_cited_block,
)

logger = logging.getLogger(__name__)

# source_type → 정렬 우선순위 (규정→법규→사례)
_SOURCE_RANK = {"사규": 0, "법규": 1, "분쟁사례": 2}


# =============================================================================
# 레인 기술자(descriptor)
# =============================================================================

@dataclass(frozen=True)
class _Lane:
    name:        str   # "규정" | "법규" | "사례"
    search_attr: str   # registry 메서드명
    is_case:     bool  # 사례 레인은 case_no + validate_case_evidence 사용
    hyde_prompt: str   # 레인별 HyDE 프롬프트 이름


_LANES: dict[str, _Lane] = {
    "규정": _Lane("규정", "regulation_search", False, "hyde_regulation"),
    "법규": _Lane("법규", "law_search",        False, "hyde_law"),
    "사례": _Lane("사례", "case_search",       True,  "hyde_case"),
}


# =============================================================================
# 프롬프트 로더 (langfuse_setup 위임, 변경 없음)
# =============================================================================

def _load_prompt(name: str) -> str:
    from .langfuse_setup import get_langfuse_prompt
    return get_langfuse_prompt(name)


# =============================================================================
# 구조화 출력 + 복구·전송 재시도 헬퍼
# =============================================================================

async def _structured_invoke_with_repair(
    structured_llm,
    prompt_text: str,
    *,
    max_repair: int = 1,
    max_transport_retry: int = 2,
):
    """
    LangChain structured_llm.ainvoke 래퍼 — 두 가지 실패를 자동으로 처리한다.

    LlamaIndex의 _structured_predict_with_repair에 대응.

    ValidationError (스키마 실패):
      오류 메시지를 프롬프트에 되먹여 최대 max_repair회 재요청.

    httpx / ollama 전송 오류 (네트워크·타임아웃):
      지수 백오프(0.5s → 1s)로 최대 max_transport_retry회 재시도.
    """
    repair_prompt, repairs, transport = prompt_text, 0, 0
    while True:
        try:
            return await structured_llm.ainvoke([HumanMessage(content=repair_prompt)])
        except ValidationError as e:
            if repairs >= max_repair:
                raise
            repairs += 1
            logger.warning(
                f"[repair] 스키마 검증 실패 → 재요청 {repairs}/{max_repair}: {e}"
            )
            repair_prompt = (
                f"{prompt_text}\n\n"
                f"[직전 응답이 스키마 검증에 실패했습니다]\n"
                f"오류: {e}\n"
                f"동일한 JSON 스키마로 수정해 다시 출력하세요."
            )
        except (
            httpx.TimeoutException,
            httpx.ConnectError,
            ollama.ResponseError,
        ) as e:
            if transport >= max_transport_retry:
                raise
            transport += 1
            wait = 0.5 * 2 ** (transport - 1)
            logger.warning(
                f"[transport] Ollama 통신 실패 → {wait}s 후 재시도 "
                f"{transport}/{max_transport_retry}: {e}"
            )
            await asyncio.sleep(wait)


# =============================================================================
# HyDE 변환 헬퍼
# =============================================================================

async def _hyde_transform(llm, query: str, prompt_name: str) -> str:
    """
    HyDE (Hypothetical Document Embedding): 원본 질문을 레인별 코퍼스 형식의 가상 문서로 변환한다.

    compliance_workflow.py의 _hyde_transform과 동일한 로직.
    실패 시 원본 쿼리를 반환하므로 워크플로우가 중단되지 않는다.
    """
    try:
        prompt = _load_prompt(prompt_name).replace("{query}", query)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        parsed = json.loads(response.content)
        hypothesis = parsed.get("hypothesis", "").strip()
        if len(hypothesis) >= 20:
            logger.info(f"[hyde] 가상 조항 생성 완료: {hypothesis[:80]}...")
            return hypothesis
        logger.warning("[hyde] 생성된 가상 조항이 너무 짧음 → 원본 쿼리 사용")
    except Exception as e:
        logger.warning(f"[hyde] 생성 실패, 원본 쿼리 사용: {e}")
    return query


# =============================================================================
# 그래프 빌더
# =============================================================================

def build_graph(llm, registry: ToolRegistry):
    """
    ComplianceState StateGraph를 구성하고 컴파일한다.

    Args:
        llm:      ChatOllama 인스턴스
        registry: ToolRegistry(LC) — 3개 검색 함수 보유

    Returns:
        CompiledGraph — graph.ainvoke(initial_state, config=...) 로 실행
    """

    # =========================================================================
    # 노드 1: classify
    # -------------------------------------------------------------------------
    # 입력: ComplianceState (query만 유의미)
    # 출력: {"agent_list": [...], "routing_reasoning": "..."}
    # =========================================================================

    async def classify(state: ComplianceState) -> dict:
        """
        LLM이 질문을 분석해 활성화할 검색 레인 목록을 결정한다.

        안전 경계:
          LLM은 레인 활성화(넓고 복구 가능한 결정)만 담당한다.
          source_type 필터는 tools_lc.py에서 레인별로 고정되어 있다.
          → LLM의 잘못된 추론이 하드 필터로 전파될 수 없다.

        fallback:
          LLM 출력이 파싱 불가하거나 유효한 에이전트가 없으면 3개 레인 전체를 활성화한다.
        """
        query: str = state["query"]
        prompt = _load_prompt("classify_agent")
        structured_llm = llm.with_structured_output(
            ClassifyResponse, method="json_schema"
        )

        try:
            parsed: ClassifyResponse = await _structured_invoke_with_repair(
                structured_llm,
                f"{prompt}\n\n질문: {query}",
            )
            agent_list = list(parsed.agents)
            routing_reasoning = parsed.reasoning
        except ValidationError as e:
            logger.warning(f"[classify] 구조화 출력 검증 실패 → 전체 에이전트 fallback: {e}")
            agent_list = []
            routing_reasoning = ""

        if not agent_list:
            agent_list = ["규정", "법규", "사례"]
            routing_reasoning += " [fallback: 전체 활성화]"
            logger.warning("[classify] 유효 에이전트 없음 → 전체 에이전트 활성화")

        logger.info(f"[classify] 활성화 에이전트: {agent_list} | 근거: {routing_reasoning}")
        return {"agent_list": agent_list, "routing_reasoning": routing_reasoning}

    # =========================================================================
    # 조건부 엣지: route_to_lanes
    # -------------------------------------------------------------------------
    # classify → Send("search", LaneWork) 목록을 반환해 병렬 fan-out을 구성한다.
    # LlamaIndex의 세 @step 래퍼 + if lane.name not in ev.agent_list 를 대체한다.
    # =========================================================================

    def route_to_lanes(state: ComplianceState) -> list[Send]:
        """
        agent_list에 있는 레인마다 Send를 생성한다.
        빈 리스트이면 (방어적으로) 전체를 열어 누락을 방지한다.
        """
        lanes = state.get("agent_list") or ["규정", "법규", "사례"]
        sends = [
            Send("search", {"query": state["query"], "lane": lane})
            for lane in lanes
        ]
        logger.info(f"[route] {len(sends)}개 레인 파견: {[s.node for s in sends]}")
        return sends

    # =========================================================================
    # 노드 2: search
    # -------------------------------------------------------------------------
    # 입력: LaneWork (query + lane)  — Send API로 전달된 per-task 슬라이스
    # 출력: {"evidence": [...]}      — operator.add reducer가 누적
    # =========================================================================

    async def search(state: LaneWork) -> dict:
        """
        단일 레인의 HyDE → 검색+재순위 → 검증 파이프라인.

        LlamaIndex의 search_규정/search_법규/search_사례 + _search_lane을 하나의
        노드로 통합한다. LaneWork의 lane 필드로 어느 레인인지 구분한다.

        실패 시 빈 리스트를 반환(sentinel 없음) — synthesize가 근거 0건을 처리한다.
        """
        query: str = state["query"]
        lane_name: str = state["lane"]
        lane = _LANES.get(lane_name)
        if lane is None:
            logger.error(f"[search] 알 수 없는 레인: {lane_name}")
            return {"evidence": []}

        try:
            hyde_query = await _hyde_transform(llm, query, lane.hyde_prompt)
            search_fn = getattr(registry, lane.search_attr)
            raw_results: list[dict] = await search_fn(query=hyde_query)
            logger.info(f"[search_{lane_name}] 검색 완료: {len(raw_results)}개 결과")
        except Exception as e:
            logger.warning(f"[search_{lane_name}] 검색 실패: {e}")
            return {"evidence": []}

        if lane.is_case:
            evidence = validate_case_evidence(raw_results)
        else:
            evidence = validate_article_evidence(raw_results)

        if lane.is_case:
            case_nos = [item["case_no"] for item in evidence if item.get("case_no")]
            logger.info(f"[search_{lane_name}] 검증된 사례: {case_nos}")
        else:
            logger.info(
                f"[search_{lane_name}] 검증된 조항: {evidence_article_labels(evidence)}"
            )

        return {"evidence": evidence}

    # =========================================================================
    # 노드 3: synthesize
    # -------------------------------------------------------------------------
    # 입력: ComplianceState (evidence 누적 완료, agent_list, routing_reasoning)
    # 출력: 최종 답변 필드 전체
    # =========================================================================

    async def synthesize(state: ComplianceState) -> dict:
        """
        검색 레인 결과를 합성해 근거 기반 답변을 생성한다.

        LlamaIndex 버전과 동일한 3단계:
          Phase 1 (코드): evidence 정렬 (규정→법규→사례), retrieved_ids 구성
          Phase 2 (LLM map): 근거별 PassageAnswer 병렬 호출
          Phase 3 (코드 reduce): 관련 있는 근거만 인용 블록으로 조합
        """
        query: str = state["query"]
        evidence_list: list[dict] = state.get("evidence", [])
        agent_list: list[str] = state.get("agent_list", [])
        routing_reasoning: str = state.get("routing_reasoning", "")

        # ── Phase 1: 규정→법규→사례 순 정렬 ────────────────────────────────
        # operator.add reducer는 도착 순서를 보장하지 않으므로 명시적으로 정렬한다.
        # LlamaIndex의 collect_evidence() 순서 보장을 대체한다.
        evidence_list = sorted(
            evidence_list,
            key=lambda x: _SOURCE_RANK.get(x.get("source_type", ""), 99),
        )

        # agents_used: 실제 파견된 레인 목록 (skipped 없이 Send로 파견된 레인만 실행됨)
        agents_used = [lane for lane in ["규정", "법규", "사례"] if lane in agent_list]

        retrieved_ids = [
            item["evidence_id"]
            for item in evidence_list
            if item.get("evidence_id")
        ]

        # ── 근거 0건: LLM 호출 없이 답변 보류 ───────────────────────────────
        if not evidence_list:
            logger.info("[synthesize] 검증된 근거 0건 → LLM 호출 생략, 답변 보류")
            return {
                "reasoning": "검색된 근거가 없어 답변할 수 없습니다.",
                "cited_ids": [],
                "cited_passages": [],
                "agents_used": agents_used,
                "cited_agents": [],
            }

        # ── Phase 2: per-passage LLM map ────────────────────────────────────
        prompt = _load_prompt("synthesize_agent")
        structured_llm = llm.with_structured_output(
            PassageAnswer, method="json_schema"
        )

        async def _answer_passage(item: dict) -> tuple[dict, Optional[PassageAnswer]]:
            """단일 근거에 대한 PassageAnswer를 반환한다. 실패 시 None."""
            passage_block = format_single_passage(item)
            user_msg = (
                f"질문: {query}\n\n"
                f"근거:\n{passage_block}\n\n"
                f"위 근거만 보고 JSON 형식으로 답하십시오."
            )
            try:
                pa = await _structured_invoke_with_repair(
                    structured_llm,
                    f"{prompt}\n\n{user_msg}",
                )
                return item, pa
            except ValidationError as e:
                logger.warning(
                    f"[synthesize] PassageAnswer 검증 실패 "
                    f"(evidence_id={item.get('evidence_id', '?')}): {e}"
                )
                return item, None

        # abatch로 병렬 실행 (Ollama 단일 인스턴스이므로 실질적 직렬이나 코드 구조는 동일)
        results = await asyncio.gather(*[_answer_passage(item) for item in evidence_list])

        # ── Phase 3: 코드 reduce ─────────────────────────────────────────────
        cited_ids: list[str] = []
        cited_passages: list[dict] = []
        cited_source_types: set[str] = set()
        blocks: list[str] = []

        for item, pa in results:
            if not (pa and pa.relevant and pa.answer.strip()):
                continue
            eid = item.get("evidence_id", "")
            if not eid:
                continue
            cited_ids.append(eid)
            cited_passages.append({"evidence_id": eid, "text": item.get("text", "")})
            cited_source_types.add(item.get("source_type", ""))
            blocks.append(format_cited_block(item, pa.answer))

        # cited_agents: 실제 인용된 source_type을 에이전트명으로 매핑, 고정 순서
        cited_agents = [
            agent
            for stype, agent in (("사규", "규정"), ("법규", "법규"), ("분쟁사례", "사례"))
            if stype in cited_source_types
        ]

        if not blocks:
            reasoning = "검색된 근거 중 질문에 직접 답할 수 있는 내용이 없습니다."
            logger.info("[synthesize] 관련 근거 없음 → 모든 passage relevant=false")
        else:
            reasoning = "\n\n".join(blocks)
            logger.info(f"[synthesize] 인용 블록 {len(blocks)}개 생성 완료")

        return {
            "reasoning": reasoning,
            "cited_ids": cited_ids,
            "cited_passages": cited_passages,
            "agents_used": agents_used,
            "cited_agents": cited_agents,
        }

    # =========================================================================
    # 그래프 구성 + 컴파일
    # =========================================================================

    builder = StateGraph(ComplianceState)
    builder.add_node("classify", classify)
    builder.add_node("search", search)
    builder.add_node("synthesize", synthesize)

    # 정적 엣지
    builder.add_edge(START, "classify")
    # classify → route_to_lanes → Send("search", ...) × N (동적 fan-out)
    builder.add_conditional_edges("classify", route_to_lanes, ["search"])
    # 모든 search 태스크 완료(같은 슈퍼스텝) → synthesize (BSP fan-in)
    builder.add_edge("search", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()
