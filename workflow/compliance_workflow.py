# =============================================================================
# compliance_workflow.py
# -----------------------------------------------------------------------------
# 역할: 증권사 컴플라이언스 Q&A 멀티에이전트 워크플로우 본체.
#       이 파일이 프로젝트의 핵심이다.
#
# LlamaIndex Workflow 동작 원리:
#   - Workflow 클래스를 상속하고 @step 데코레이터로 Step을 정의한다.
#   - 각 Step의 입력/출력 타입 어노테이션으로 LlamaIndex가 DAG를 자동 구성한다.
#   - async/await 기반이므로 Step 2a/b/c는 실제로 동시에 실행된다.
#     (vLLM 서버는 병렬 요청을 지원하므로 실질적 병렬 실행이 가능하다)
#
# Step 구성:
#   Step 1: classify_step   — LLM 기반 라우팅 (constrained JSON output)
#   Step 2a: search_규정    — 사규 검색 + 조항 exact-match 검증
#   Step 2b: search_법규    — 법규 검색 + 조항 exact-match 검증
#   Step 2c: search_사례    — 분쟁사례 검색 + 사건번호 metadata 검증
#   Step 3: synthesize_step — LLM 합성 (3개 결과 수집 후 실행) → StopEvent 직접 반환
#
# 검색 Step 구조:
#   - 코드가 lane별 검색 함수를 직접 호출한다.
#   - LLM은 검색/검증 lane에서 사용하지 않고 최종 합성에만 사용한다.
#   - strict control: 어떤 검색 함수를 쓸지 코드에 하드코딩됨
#
# 이 설계의 정당성:
#   금융 컴플라이언스 도메인은 LLM의 자율적 판단이 위험하다.
#   Rule이 통제하고 LLM은 해석에만 집중 = "Rule-based & LLM Hybrid"
# =============================================================================

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Optional

import httpx
import openai

from pydantic import ValidationError

from langfuse import observe, get_client

from llama_index.core.workflow import (
    Workflow,      # 워크플로우 기반 클래스
    StartEvent,    # 워크플로우 시작 이벤트
    StopEvent,     # 워크플로우 종료 이벤트
    Context,       # Step 간 공유 상태 저장소
    step,          # Step 정의 데코레이터
)
from llama_index.core.llms import LLM  # LLM 기반 클래스 (타입 힌트용)
from llama_index.core.prompts import PromptTemplate  # 구조화 출력(structured_predict) 호출용

from .events import (
    ClassifiedEvent,
    RegulationResultEvent,
    LawResultEvent,
    CaseResultEvent,
    FinalAnswer,
    PassageAnswer,       # per-passage LLM 출력 계약 (relevant + answer)
    ClassifyResponse,
)
from .tools import ToolRegistry
from .evidence import (
    validate_article_evidence,
    validate_case_evidence,
    evidence_article_labels,
    collect_evidence,
    format_single_passage,
    format_cited_block,
)
logger = logging.getLogger(__name__)

_PASSTHROUGH_PROMPT = PromptTemplate("{__prompt__}")


# =============================================================================
# 검색 레인 기술자(descriptor)
# -----------------------------------------------------------------------------
# 사규/법규/사례 세 검색 Step은 본문이 거의 동일하다. 차이를 데이터로 표현해
# _search_lane() 하나로 합치고, @step 메서드는 얇은 래퍼로만 남긴다.
# =============================================================================

@dataclass(frozen=True)
class _Lane:
    name: str           # agent_list 키: "규정" / "법규" / "사례"
    search_attr: str    # registry 메서드명: regulation_search / law_search / case_search
    event_cls: type     # RegulationResultEvent / LawResultEvent / CaseResultEvent
    is_case: bool       # 사례 레인은 case_no + validate_case_evidence 사용
    hyde_prompt: str    # 레인별 HyDE 프롬프트 이름 (코퍼스 형식에 맞는 가상 문서 생성)


_REGULATION_LANE = _Lane("규정", "regulation_search", RegulationResultEvent, False, "hyde_regulation")
_LAW_LANE        = _Lane("법규", "law_search",        LawResultEvent,        False, "hyde_law")
_CASE_LANE       = _Lane("사례", "case_search",       CaseResultEvent,       True,  "hyde_case")


# =============================================================================
# 프롬프트 파일 로더
# =============================================================================


def load_prompt(name: str) -> str:
    """
    Langfuse 프롬프트 레지스트리에서 프롬프트를 가져온다.

    Langfuse 우선:
      - Langfuse에 등록된 프롬프트를 가져와 compiled 문자열을 반환한다.
      - Langfuse 미설정 / 미연결 시 prompts/*.txt 파일로 자동 fallback한다.
      → 프롬프트 등록은 manage_prompts.py 로 별도 수행한다 (LANGFUSE_SYNC_PROMPTS=1 도 가능).
        서빙 경로는 사용 시점에 lazy하게 가져오고 Langfuse 미연결 시 로컬 파일로 fallback한다.

    Langfuse 기반 버전 관리:
      Langfuse 대시보드에서 프롬프트를 수정하면 다음 run부터 반영된다.
      코드 변경 없이 프롬프트 A/B 테스트 및 버전 롤백이 가능하다.

    Args:
        name: 프롬프트 이름 (예: "synthesize_agent", "classify_agent")

    Returns:
        프롬프트 텍스트 문자열
    """
    from .langfuse_setup import get_langfuse_prompt
    return get_langfuse_prompt(name)


# =============================================================================
# Step 1 분류: LLM 기반 에이전트 라우팅
# =============================================================================
# classify_agent.txt 프롬프트로 LLM이 어떤 검색 레인을 열지 결정한다.
# 출력은 {"규정", "법규", "사례"} 집합으로 제한되며, 허용되지 않은 값은 코드에서 제거된다.
# LLM 출력이 파싱 불가하거나 빈 리스트이면 3개 레인을 모두 여는 fallback이 동작한다.


# =============================================================================
# ComplianceWorkflow: 메인 워크플로우 클래스
# =============================================================================

class ComplianceWorkflow(Workflow):
    """
    증권사 컴플라이언스 Q&A 멀티에이전트 워크플로우.

    초기화:
      llm:      LLM 인스턴스 (main.py 또는 app/server.py에서 주입 — Ollama 또는 OpenAILike)
      registry: ToolRegistry (검색/조회 함수 보유)
      timeout:  LlamaIndex 내장 전체 타임아웃 (초)
                → 이 시간 초과 시 워크플로우 강제 종료

    실행:
      handler = wf.run(query="질문")
      result = await handler  # FinalAnswer 반환
    """

    def __init__(self, llm: LLM, registry: ToolRegistry, **kwargs):
        # **kwargs에 timeout, verbose 등이 포함됨
        # Workflow.__init__에 그대로 전달
        super().__init__(**kwargs)
        self.llm = llm            # LLM 인스턴스 (Ollama qwen3:8b-q4_K_M)
        self.registry = registry  # 검색/조회 함수를 보유한 레지스트리

    # =========================================================================
    # 내부 헬퍼: 구조화 출력 + 복구·전송 재시도
    # =========================================================================

    async def _structured_predict_with_repair(
        self,
        output_cls,
        prompt_text: str,
        *,
        max_repair: int = 1,
        max_transport_retry: int = 2,
    ):
        """
        astructured_predict 래퍼 — 두 가지 실패를 자동으로 처리한다.

        ValidationError (스키마 실패):
          오류 메시지를 프롬프트에 되먹여 최대 max_repair회 재요청.
          한도 초과 시 ValidationError를 그대로 올려 호출부의 폴백 정책을 유지한다.

        httpx / openai 전송 오류 (네트워크·타임아웃):
          지수 백오프(0.5s → 1s)로 최대 max_transport_retry회 재시도.
          한도 초과 시 예외를 그대로 올려 fail-loud 정책을 유지한다.
        """
        repair_prompt, repairs, transport = prompt_text, 0, 0
        while True:
            try:
                return await self.llm.astructured_predict(
                    output_cls, _PASSTHROUGH_PROMPT, __prompt__=repair_prompt
                )
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
                openai.APIConnectionError,
                openai.APITimeoutError,
            ) as e:
                if transport >= max_transport_retry:
                    raise
                transport += 1
                wait = 0.5 * 2 ** (transport - 1)
                logger.warning(
                    f"[transport] LLM 통신 실패 → {wait}s 후 재시도 "
                    f"{transport}/{max_transport_retry}: {e}"
                )
                await asyncio.sleep(wait)

    # =========================================================================
    # Step 1: classify_step
    # -------------------------------------------------------------------------
    # 입력: StartEvent (query 포함)
    # 출력: ClassifiedEvent (query + agent_list)
    # LLM: 있음 (구조화 출력 라우팅)
    # =========================================================================

    @step
    @observe(name="classify_step", as_type="span", capture_input=False, capture_output=False)
    async def classify_step(
        self,
        ctx: Context,
        ev: StartEvent  # LlamaIndex가 자동으로 이 Step을 시작점으로 인식
    ) -> ClassifiedEvent:
        """
        LLM이 질문을 분석해 활성화할 검색 에이전트 목록을 결정한다.

        LLM 사용 이유:
          하드코딩된 키워드는 동의어, 신조어, 우회 표현을 처리할 수 없어 확장성이 없다.
          LLM은 의미 기반으로 레인을 선택하며, 출력은 {"규정","법규","사례"} 집합으로 제한된다.

        안전 경계:
          LLM은 레인 활성화(넓고 복구 가능한 결정)만 담당한다.
          source_type 필터는 tools.py에서 레인별로 고정되어 있다.
          → LLM의 잘못된 추론이 하드 필터로 전파될 수 없다.

        fallback:
          LLM 출력이 파싱 불가하거나 유효한 에이전트가 없으면 3개 레인 전체를 활성화한다.

        ctx 초기화:
          모든 Step이 공유하는 ctx에 초기값을 설정한다.
          이 Step이 항상 첫 번째로 실행되므로 초기화 위치로 적합하다.
        """
        query: str = ev.get("query", "")

        # Langfuse: Step 입력 기록
        get_client().update_current_span(input={"query": query})

        # ctx 초기값 설정 (이후 Step들이 읽고 업데이트함)

        # ── LLM 라우팅 (구조화 출력) ──────────────────────────────────────────
        # ClassifyResponse 스키마를 Ollama format= 으로 강제 → agents가 허용 3종으로
        # 제한되므로 코드 측 화이트리스트 필터가 불필요하다. 스키마 위반 시
        # ValidationError → 빈 결과로 두고 아래 fallback이 전체 레인을 연다.
        prompt = load_prompt("classify_agent")
        try:
            parsed = await self._structured_predict_with_repair(
                ClassifyResponse,
                f"{prompt}\n\n질문: {query}",
            )
            agent_list = list(parsed.agents)
            routing_reasoning = parsed.reasoning
        except ValidationError as e:
            logger.warning(f"[classify] 구조화 출력 검증 실패 → 전체 에이전트 fallback: {e}")
            agent_list = []
            routing_reasoning = ""

        # ── 안전 fallback ────────────────────────────────────────────────────
        # LLM이 유효한 에이전트를 하나도 반환하지 못한 경우 전체 레인을 열어
        # 관련 근거를 누락하는 것보다 불필요한 검색이 낫다는 원칙을 적용한다.
        if not agent_list:
            agent_list = ["규정", "법규", "사례"]
            routing_reasoning += " [fallback: 전체 활성화]"
            logger.warning("[classify] 유효 에이전트 없음 → 전체 에이전트 활성화")

        logger.info(f"[classify] 활성화 에이전트: {agent_list} | 근거: {routing_reasoning}")

        # routing_reasoning을 ctx에 보관 → synthesize_step이 읽어 FinalAnswer에 싣는다.
        await ctx.store.set("routing_reasoning", routing_reasoning)

        # Langfuse: routing_reasoning을 출력으로 기록 → 대시보드에서 LLM 판단 근거 확인 가능
        get_client().update_current_span(
            output={"agent_list": agent_list, "routing_reasoning": routing_reasoning}
        )

        return ClassifiedEvent(query=query, agent_list=agent_list)

    # =========================================================================
    # Step 2a/2b/2c: 검색 레인 (search_규정 / search_법규 / search_사례)
    # -------------------------------------------------------------------------
    # 입력: ClassifiedEvent → 출력: 각 레인의 ResultEvent
    # 세 Step은 lane 기술자만 다르고 본문이 동일하므로 _search_lane에 위임한다.
    # @step / @observe는 래퍼에 그대로 남겨 DAG 라우팅과 Langfuse span 이름을 보존한다.
    # 검색은 Rule-based(LLM 없음)이며 source_type은 registry에서 고정된다.
    # =========================================================================

    @step
    @observe(name="search_규정", as_type="span", capture_input=False, capture_output=False)
    async def search_규정(
        self, ctx: Context, ev: ClassifiedEvent
    ) -> RegulationResultEvent:
        """사규 검색 에이전트 (→ _search_lane)."""
        return await self._search_lane(ctx, ev, _REGULATION_LANE)

    @step
    @observe(name="search_법규", as_type="span", capture_input=False, capture_output=False)
    async def search_법규(
        self, ctx: Context, ev: ClassifiedEvent
    ) -> LawResultEvent:
        """법규 검색 에이전트 (→ _search_lane)."""
        return await self._search_lane(ctx, ev, _LAW_LANE)

    @step
    @observe(name="search_사례", as_type="span", capture_input=False, capture_output=False)
    async def search_사례(
        self, ctx: Context, ev: ClassifiedEvent
    ) -> CaseResultEvent:
        """분쟁사례 검색 에이전트 (→ _search_lane)."""
        return await self._search_lane(ctx, ev, _CASE_LANE)

    @staticmethod
    def _skip_event(lane: _Lane, query: str):
        """skipped=True 결과 이벤트를 생성한다."""
        return lane.event_cls(query=query, evidence=[], skipped=True)

    async def _search_lane(
        self, ctx: Context, ev: ClassifiedEvent, lane: _Lane
    ):
        """
        사규/법규/사례 검색 레인의 공통 본문.

        agent_list에 없으면 스킵 이벤트를 반환한다(synthesize_step이 집계하므로
        워크플로우는 중단되지 않는다). 검색 실패 시에도 스킵 이벤트로 폴백한다.
        """
        # ── 스킵: agent_list에 이 레인이 없으면 검색하지 않음 ──────────────
        if lane.name not in ev.agent_list:
            logger.info(f"[search_{lane.name}] 스킵 (agent_list에 없음)")
            get_client().update_current_span(
                input={"query": ev.query},
                output={"skipped": True},
            )
            return self._skip_event(lane, ev.query)

        # ── 검색 실행 (Rule-based, LLM 없음) ───────────────────────────────
        try:
            # HyDE: 원본 쿼리를 가상 법령 조항으로 변환해 임베딩 정렬을 개선한다.
            hyde_query = await self._hyde_transform(ev.query, lane.hyde_prompt)
            get_client().update_current_span(
                input={
                    "query": ev.query,
                    "hyde_query": hyde_query,
                },
            )
            search_fn = getattr(self.registry, lane.search_attr)
            raw_results: list[dict] = await search_fn(query=hyde_query)
            logger.info(f"[search_{lane.name}] 검색 완료: {len(raw_results)}개 결과")
        except Exception as e:
            # 검색 자체가 실패하면 (DB 오류 등) 스킵 처리
            logger.warning(f"[search_{lane.name}] 검색 실패: {e}")
            get_client().update_current_span(
                output={"error": str(e), "skipped": True},
                level="ERROR",
            )
            return self._skip_event(lane, ev.query)

        # ── 검증 ──────────────────────────────────────────────────────────
        if lane.is_case:
            evidence = validate_case_evidence(raw_results)
        else:
            evidence = validate_article_evidence(
                raw_results, self.registry.article_lookup
            )

        # ── 결과 이벤트 구성 ───────────────────────────────────────────────
        # case_nos / articles 라벨은 Langfuse 관찰용으로만 남기고, 이벤트에는
        # 검증된 evidence 객체만 싣는다.
        if lane.is_case:
            case_nos = [item["case_no"] for item in evidence if item.get("case_no")]
            get_client().update_current_span(
                output={"evidence_count": len(evidence), "case_nos": case_nos},
            )
        else:
            get_client().update_current_span(
                output={
                    "evidence_count": len(evidence),
                    "articles": evidence_article_labels(evidence),
                },
            )
        return lane.event_cls(query=ev.query, evidence=evidence)

    # =========================================================================
    # Step 3: synthesize_step
    # -------------------------------------------------------------------------
    # 입력: RegulationResultEvent | LawResultEvent | CaseResultEvent (3개 모두)
    # 출력: StopEvent(result=FinalAnswer) → 워크플로우 종료
    # LLM: 근거 passage 개수(N)만큼 호출 — per-passage map → 코드 reduce
    # 특이사항: collect_events()로 3개 이벤트가 모두 도착할 때까지 대기
    # =========================================================================

    @step
    @observe(name="synthesize_step", as_type="span", capture_input=False, capture_output=False)
    async def synthesize_step(
        self,
        ctx: Context,
        # 유니온 타입: 세 이벤트 중 어느 것이든 이 Step을 트리거할 수 있음
        # LlamaIndex가 세 타입을 모두 처리하는 Step으로 자동 인식
        ev: RegulationResultEvent | LawResultEvent | CaseResultEvent,
    ) -> Optional[StopEvent]:
        """
        3개 검색 에이전트의 결과를 합성해 근거 기반 답변을 생성하고 StopEvent로 종료한다.

        collect_events() 동작:
          세 이벤트 중 하나가 도착할 때마다 이 Step이 호출됨.
          하지만 collect_events()가 3개 모두 모일 때까지 None을 반환.
          3개 모두 도착하면 [RegulationResultEvent, LawResultEvent, CaseResultEvent]
          리스트를 반환하고 실제 처리를 시작함.

          → None 반환 시 LlamaIndex는 이 Step이 아직 완료되지 않았다고 인식
          → 모든 Step이 None을 반환하면 대기 상태가 됨
          → 마지막 이벤트 도착 시 collect_events()가 리스트를 반환하고 처리 시작

        이 패턴이 병렬 처리를 안전하게 동기화한다.
        """
        # collect_events(): 지정된 타입의 이벤트가 모두 도착하면 리스트 반환
        # 아직 다 안 모였으면 None 반환
        collected = ctx.collect_events(
            ev,  # 현재 도착한 이벤트
            [RegulationResultEvent, LawResultEvent, CaseResultEvent],  # 기다릴 타입들
        )

        if collected is None:
            # 아직 3개 모두 도착하지 않음 → 대기
            return None

        # 3개 모두 도착: 언패킹
        # LlamaIndex는 타입 선언 순서대로 반환을 보장함
        reg_ev, law_ev, case_ev = collected

        # agents_used: 수집된 이벤트에서 직접 파생한다. 각 레인은 skip/실패 시
        # skipped=True로 표시하므로 not skipped == 실제 실행 및 완료된 레인과 동치.
        # ctx.store 누적 방식의 read-modify-write 경쟁 조건을 제거하고 결정적 순서를 보장.
        agents_used = [
            name for name, lane_ev in (("규정", reg_ev), ("법규", law_ev), ("사례", case_ev))
            if not lane_ev.skipped
        ]
        routing_reasoning = await ctx.store.get("routing_reasoning", "")

        # ── Phase 1 (코드): 3개 레인 근거를 단일 리스트로 합산 ─────────────────
        # collect_evidence가 규정→법규→사례 순서를 보존하며 skipped 레인을 건넌다.
        evidence_list = collect_evidence(reg_ev, law_ev, case_ev)
        retrieved_ids = [item["evidence_id"] for item in evidence_list if item.get("evidence_id")]

        # ── 근거 0건: LLM 호출 없이 답변 보류 ─────────────────────────────────
        # 검증된 근거가 하나도 없으면 합성할 원문이 없다.
        if not evidence_list:
            logger.info("[synthesize] 검증된 근거 0건 → LLM 호출 생략, 답변 보류")
            get_client().update_current_span(
                output={"reasoning": "no_evidence", "cited_count": 0},
                metadata={"agents_used": str(agents_used)},
            )
            return StopEvent(result=FinalAnswer(
                reasoning="검색된 근거가 없어 답변할 수 없습니다.",
                cited_ids=[],
                cited_passages=[],
                agents_used=agents_used,
                routing_reasoning=routing_reasoning,
            ))

        # Langfuse: Step 입력 기록 (LLM 호출 전)
        get_client().update_current_span(
            input={
                "query": reg_ev.query,
                "evidence_ids": retrieved_ids,
            },
        )

        # ── Phase 2 (LLM map): 근거별 1회 호출 → PassageAnswer ────────────────
        # 각 근거 passage를 독립적으로 LLM에 넘긴다. LLM은 출처를 명명하지 않고
        # {relevant, answer}만 반환한다. 코드가 evidence_id를 citation으로 부착한다.
        # → hallucination 경로가 스키마 수준에서 차단된다.
        prompt = load_prompt("synthesize_agent")
        query = reg_ev.query

        async def _answer_passage(item: dict) -> tuple[dict, Optional[PassageAnswer]]:
            """단일 근거에 대한 PassageAnswer를 반환한다. 실패 시 None."""
            passage_block = format_single_passage(item)
            user_msg = (
                f"질문: {query}\n\n"
                f"근거:\n{passage_block}\n\n"
                f"위 근거만 보고 JSON 형식으로 답하십시오."
            )
            try:
                pa = await self._structured_predict_with_repair(
                    PassageAnswer,
                    f"{prompt}\n\n{user_msg}",
                )
                return item, pa
            except ValidationError as e:
                logger.warning(
                    f"[synthesize] PassageAnswer 검증 실패 "
                    f"(evidence_id={item.get('evidence_id', '?')}): {e}"
                )
                return item, None

        # asyncio.gather: 근거를 병렬로 평가한다.
        # vLLM 서버는 병렬 요청을 지원하므로 다중 근거 처리 시 실질적 성능 향상이 있다.
        results = await asyncio.gather(*[_answer_passage(item) for item in evidence_list])

        # ── Phase 3 (코드 reduce): 관련 있는 근거만 인용 블록으로 조합 ──────────
        cited_ids: list[str] = []
        cited_passages: list[dict] = []  # [인용 근거] 출력용: evidence_id + text
        cited_source_types: set[str] = set()  # 인용된 근거의 source_type → cited_agents 도출용
        blocks: list[str] = []

        for item, pa in results:
            if not (pa and pa.relevant and pa.answer.strip()):
                continue
            eid = item.get("evidence_id", "")
            if not eid:
                continue  # 검증된 근거엔 항상 evidence_id가 있으나, 없으면 인용 바인딩 불가
            cited_ids.append(eid)
            cited_passages.append({"evidence_id": eid, "text": item.get("text", "")})
            cited_source_types.add(item.get("source_type", ""))
            blocks.append(format_cited_block(item, pa.answer))

        # cited_agents: 실제 인용된 source_type을 에이전트명으로 매핑, 규정→법규→사례 순 고정
        cited_agents = [
            agent
            for stype, agent in (("사규", "규정"), ("법규", "법규"), ("분쟁사례", "사례"))
            if stype in cited_source_types
        ]

        if not blocks:
            # 근거는 검색됐으나 관련 passage가 없는 경우.
            reasoning = "검색된 근거 중 질문에 직접 답할 수 있는 내용이 없습니다."
            logger.info("[synthesize] 관련 근거 없음 → 모든 passage relevant=false")
        else:
            reasoning = "\n\n".join(blocks)
            logger.info(f"[synthesize] 인용 블록 {len(blocks)}개 생성 완료")

        # Langfuse: Step 출력 기록
        get_client().update_current_span(
            output={
                "cited_count": len(cited_ids),
                "relevant_count": len(blocks),
                "cited_ids": cited_ids,
                "cited_agents": cited_agents,
                "retrieved_ids": retrieved_ids,
            },
            metadata={"agents_used": str(agents_used)},
        )

        return StopEvent(result=FinalAnswer(
            reasoning=reasoning,
            cited_ids=cited_ids,          # 코드가 부착한 인용 ID (hallucination 불가)
            cited_passages=cited_passages,  # [인용 근거] 출력용 (evidence_id + text)
            agents_used=agents_used,
            cited_agents=cited_agents,    # agents_used의 부분집합: 실제 인용된 에이전트
            routing_reasoning=routing_reasoning,
        ))

    # =========================================================================
    # 헬퍼 메서드: 포맷터 / 파서
    # =========================================================================

    async def _hyde_transform(self, query: str, prompt_name: str) -> str:
        """
        HyDE (Hypothetical Document Embedding): 원본 질문을 레인별 코퍼스 형식의 가상 문서로 변환한다.

        왜 필요한가:
          질문("65세 고객에게 레버리지 ETF 권유 가능한가요?")과 코퍼스("금융투자업자는
          고령투자자에 대해 강화된 적합성 확인 절차를 적용하여야 한다")는 언어 형식과
          어휘가 달라 임베딩 벡터 공간에서 멀리 위치할 수 있다.
          LLM이 레인 코퍼스와 같은 문체로 가상 문서를 생성하면 임베딩 공간에서
          실제 문서와 가까워져 검색 정밀도가 높아진다.

        레인별 프롬프트 (prompt_name):
          hyde_law        → 법령 조문 문체 ("금융투자업자는 ... 하여야 한다")
          hyde_regulation → 내규·표준규정 조문 문체 ("회사는 / 임직원등은 ... 하여야 한다")
          hyde_case       → 법원 판례 판시사항·판결요지 문체 ("... 하는지 여부(적극)")

        안전성:
          가상 문서는 임베딩에만 사용되고 최종 답변에는 노출되지 않는다.
          조항 번호·사건번호 인용을 금지하므로 잘못된 citation이 생성되지 않는다.
          실패 시 원본 쿼리를 반환하므로 워크플로우가 중단되지 않는다.

        Reference: Gao et al. (2022) "Precise Zero-Shot Dense Retrieval without
                   Relevance Labels", arXiv:2212.10496
        """
        try:
            prompt = load_prompt(prompt_name).replace("{query}", query)
            response = await self.llm.acomplete(prompt)
            # vLLM with guided JSON (or Ollama json_mode) returns a JSON object;
            # extract the hypothesis text from it.
            parsed = json.loads(response.text)
            hypothesis = parsed.get("hypothesis", "").strip()
            if len(hypothesis) >= 20:
                logger.info(f"[hyde] 가상 조항 생성 완료: {hypothesis[:80]}...")
                return hypothesis
            logger.warning("[hyde] 생성된 가상 조항이 너무 짧음 → 원본 쿼리 사용")
        except Exception as e:
            logger.warning(f"[hyde] 생성 실패, 원본 쿼리 사용: {e}")
        return query

    # 검색 근거 추출·검증·포맷 로직은 evidence.py로 분리했다 (순수 함수).

    # ── LLM 응답 JSON 파서들 ────────────────────────────────────────────────
    # vLLM guided JSON (또는 Ollama json_mode)으로 LLM이 순수 JSON만 반환하도록 강제했으므로
    # 방어적 추출(_safe_json)은 더 이상 필요 없다. 직접 json.loads로 파싱한다.
    # (롤백 대비를 위해 _safe_json 구현은 삭제하지 않고 주석으로 보존한다.)

    # @staticmethod
    # def _safe_json(text: str) -> dict:
    #     """
    #     LLM 응답 텍스트에서 첫 번째 유효한 JSON 객체를 안전하게 추출한다.
    #
    #     기존 방식의 문제:
    #       re.search(r'\\{.*\\}', text, re.DOTALL) — greedy 매칭.
    #       LLM이 JSON 뒤에 설명 문장을 추가하면 마지막 }까지 통째로 잡아
    #       json.loads()가 실패하고 빈 dict를 반환한다.
    #
    #     개선 방식:
    #       json.JSONDecoder().raw_decode()는 첫 번째 완전한 JSON 객체만
    #       파싱하고 나머지 텍스트(마크다운 설명 등)는 무시한다.
    #       JSON 앞에 텍스트가 있으면 { 위치를 찾아 거기서부터 시도한다.
    #     """
    #     text = text.strip()
    #     # { 가 처음 등장하는 위치부터 순서대로 시도
    #     for start in range(len(text)):
    #         if text[start] != "{":
    #             continue
    #         try:
    #             obj, _ = json.JSONDecoder().raw_decode(text, start)
    #             return obj
    #         except json.JSONDecodeError:
    #             continue  # 이 위치는 유효한 JSON 시작점이 아님 → 다음 시도
    #
    #     logger.warning(f"JSON 없음: {text[:100]}")
    #     return {}
