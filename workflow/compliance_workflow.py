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
#     (단, Ollama는 직렬 처리이므로 실질적 병렬 실행은 아님 — README에 명시 필요)
#
# 5개 Step:
#   Step 1: classify_step   — LLM 기반 라우팅 (constrained JSON output)
#   Step 2a: search_규정    — 사규 검색 + 조항 exact-match 검증
#   Step 2b: search_법규    — 법규 검색 + 조항 exact-match 검증
#   Step 2c: search_사례    — 분쟁사례 검색 + 사건번호 metadata 검증
#   Step 3: synthesize_step — LLM 합성 (3개 결과 수집 후 실행)
#   Step 4: factcheck_step  — 조항 exact-match 재검증 + LLM 존재 여부 판정
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
import ollama

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
    CitedArticle,
    ClassifiedEvent,
    RegulationResultEvent,
    LawResultEvent,
    CaseResultEvent,
    SynthesizedEvent,
    FinalAnswer,
    SynthesisResponse,   # LLM 구조화 출력 스키마
    ClassifyResponse,
    FactcheckResponse,
)
from .tools import ToolRegistry
from .evidence import (
    precision_filters,
    validate_article_evidence,
    validate_case_evidence,
    evidence_article_labels,
    format_synthesis_input,
    format_factcheck_input,
)
logger = logging.getLogger(__name__)

# factcheck_step이 SynthesizedEvent를 재emit하는 최대 횟수.
# 이 값을 초과하면 partial 결과로 강제 종료 (무한 루프 방지).
MAX_FACTCHECK_RETRY = 1

# astructured_predict는 PromptTemplate에 .format()을 적용한다. 프롬프트 본문의
# 리터럴 중괄호({ }, 예: 출력 형식 예시)가 재해석되지 않도록 전체 프롬프트를 단일
# 치환 슬롯으로 통째로 전달한다 — 치환되는 값 안의 중괄호는 str.format이 다시 해석하지 않는다.
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
    is_case: bool       # 사례 레인은 case_nos + validate_case_evidence 사용
    summary_label: str  # summary 문구용 표기 (규정 레인은 "사규"로 표기)
    hyde_prompt: str    # 레인별 HyDE 프롬프트 이름 (코퍼스 형식에 맞는 가상 문서 생성)


_REGULATION_LANE = _Lane("규정", "regulation_search", RegulationResultEvent, False, "사규", "hyde_regulation")
_LAW_LANE        = _Lane("법규", "law_search",        LawResultEvent,        False, "법규", "hyde_law")
_CASE_LANE       = _Lane("사례", "case_search",       CaseResultEvent,       True,  "사례", "hyde_case")


# =============================================================================
# 프롬프트 파일 로더
# =============================================================================


def load_prompt(name: str) -> str:
    """
    Langfuse 프롬프트 레지스트리에서 프롬프트를 가져온다.

    Langfuse 우선:
      - Langfuse에 등록된 프롬프트를 가져와 compiled 문자열을 반환한다.
      - Langfuse 미설정 / 미연결 시 prompts/*.txt 파일로 자동 fallback한다.
      → langfuse_setup.sync_prompts()가 앱 시작 시 로컬 파일을 Langfuse에 업로드하므로
        일반적으로 Langfuse에서 가져오는 경로가 동작한다.

    Langfuse 기반 버전 관리:
      Langfuse 대시보드에서 프롬프트를 수정하면 다음 run부터 반영된다.
      코드 변경 없이 프롬프트 A/B 테스트 및 버전 롤백이 가능하다.

    Args:
        name: 프롬프트 이름 (예: "synthesize_agent", "factcheck_agent")

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
      llm:      LLM 인스턴스 (main.py에서 Ollama로 설정)
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
        self.llm = llm            # LLM 인스턴스 (Ollama Qwen2.5)
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

        httpx / ollama 전송 오류 (네트워크·타임아웃):
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

    # =========================================================================
    # Step 1: classify_step
    # -------------------------------------------------------------------------
    # 입력: StartEvent (query 포함)
    # 출력: ClassifiedEvent (query + agent_list)
    # LLM: 없음 (순수 Rule-based)
    # =========================================================================

    @step
    @observe(name="classify_step", as_type="span")
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
          source_name·citation_id 정밀 필터는 evidence.precision_filters()에서 regex로만 처리한다.
          → LLM의 잘못된 추론이 하드 AND 필터로 전파될 수 없다.

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
        await ctx.store.set("retry_count", 0)    # factcheck 재시도 카운터
        await ctx.store.set("agents_used", [])   # 실제 실행된 에이전트 기록

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
    @observe(name="search_규정", as_type="span")
    async def search_규정(
        self, ctx: Context, ev: ClassifiedEvent
    ) -> RegulationResultEvent:
        """사규 검색 에이전트 (→ _search_lane)."""
        return await self._search_lane(ctx, ev, _REGULATION_LANE)

    @step
    @observe(name="search_법규", as_type="span")
    async def search_법규(
        self, ctx: Context, ev: ClassifiedEvent
    ) -> LawResultEvent:
        """법규 검색 에이전트 (→ _search_lane)."""
        return await self._search_lane(ctx, ev, _LAW_LANE)

    @step
    @observe(name="search_사례", as_type="span")
    async def search_사례(
        self, ctx: Context, ev: ClassifiedEvent
    ) -> CaseResultEvent:
        """분쟁사례 검색 에이전트 (→ _search_lane)."""
        return await self._search_lane(ctx, ev, _CASE_LANE)

    @staticmethod
    def _skip_event(lane: _Lane, query: str, summary: str):
        """레인 타입에 맞는 skipped=True 결과 이벤트를 생성한다."""
        if lane.is_case:
            return lane.event_cls(
                query=query, case_nos=[], summary=summary,
                confidence=0.0, evidence=[], skipped=True,
            )
        return lane.event_cls(
            query=query, articles=[], summary=summary,
            confidence=0.0, evidence=[], skipped=True,
        )

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
            return self._skip_event(lane, ev.query, "(스킵됨)")

        # ── 검색 실행 (Rule-based, LLM 없음) ───────────────────────────────
        try:
            filters = precision_filters(ev.query, lane.name)
            # HyDE: 원본 쿼리를 가상 법령 조항으로 변환해 임베딩 정렬을 개선한다.
            # (precision_filters는 원본 쿼리 기준으로 추출 — 명시된 조항번호·문서명 regex)
            hyde_query = await self._hyde_transform(ev.query, lane.hyde_prompt)
            get_client().update_current_span(
                input={
                    "query": ev.query,
                    "hyde_query": hyde_query,
                    "precision_filters": filters,
                },
            )
            search_fn = getattr(self.registry, lane.search_attr)
            raw_results: list[dict] = await search_fn(query=hyde_query, **filters)
            logger.info(f"[search_{lane.name}] 검색 완료: {len(raw_results)}개 결과")
        except Exception as e:
            # 검색 자체가 실패하면 (DB 오류 등) 스킵 처리
            logger.warning(f"[search_{lane.name}] 검색 실패: {e}")
            get_client().update_current_span(
                output={"error": str(e), "skipped": True},
                level="ERROR",
            )
            return self._skip_event(lane, ev.query, "검색 오류")

        # ── 검증 + agents_used 기록 ────────────────────────────────────────
        if lane.is_case:
            evidence = validate_case_evidence(raw_results)
        else:
            evidence = validate_article_evidence(
                raw_results, self.registry.article_lookup
            )
        agents_used = await ctx.store.get("agents_used", [])
        await ctx.store.set("agents_used", agents_used + [lane.name])

        summary = f"검증된 {lane.summary_label} 근거 {len(evidence)}개"
        confidence = 1.0 if evidence else 0.0

        # ── 결과 이벤트 구성 (타입별 필드 분기) ────────────────────────────
        if lane.is_case:
            case_nos = [item["case_no"] for item in evidence if item.get("case_no")]
            get_client().update_current_span(
                output={"evidence_count": len(evidence), "case_nos": case_nos},
            )
            return lane.event_cls(
                query=ev.query, case_nos=case_nos, summary=summary,
                confidence=confidence, evidence=evidence,
            )

        labels = evidence_article_labels(evidence)
        get_client().update_current_span(
            output={"evidence_count": len(evidence), "articles": labels},
        )
        return lane.event_cls(
            query=ev.query, articles=labels, summary=summary,
            confidence=confidence, evidence=evidence,
        )

    # =========================================================================
    # Step 3: synthesize_step
    # -------------------------------------------------------------------------
    # 입력: RegulationResultEvent | LawResultEvent | CaseResultEvent (3개 모두)
    # 출력: SynthesizedEvent
    # LLM: 1회 호출
    # 특이사항: collect_events()로 3개 이벤트가 모두 도착할 때까지 대기
    # =========================================================================

    @step
    @observe(name="synthesize_step", as_type="span")
    async def synthesize_step(
        self,
        ctx: Context,
        # 유니온 타입: 세 이벤트 중 어느 것이든 이 Step을 트리거할 수 있음
        # LlamaIndex가 세 타입을 모두 처리하는 Step으로 자동 인식
        ev: RegulationResultEvent | LawResultEvent | CaseResultEvent,
    ) -> Optional[SynthesizedEvent]:
        """
        3개 검색 에이전트의 결과를 합성해 최종 판정 초안을 생성한다.

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

        prompt = load_prompt("synthesize_agent")

        # ── Phase 1 (코드): 검증된 근거 목록을 evidence_id 기준으로 색인 ────
        # LLM이 cited_articles를 직접 생성하면 citation_id 형식이 달라져
        # article_lookup 테이블 키와 불일치할 수 있다.
        # → 검증된 evidence 객체를 코드에서 직접 색인하고,
        #   LLM에게는 evidence_id 문자열(키)만 선택하도록 위임한다.
        all_evidence: dict[str, dict] = {
            item["evidence_id"]: item
            for _ev in [reg_ev, law_ev, case_ev]
            if not _ev.skipped
            for item in _ev.evidence
            if item.get("evidence_id")
        }

        # Langfuse: Step 입력 기록 (LLM 호출 전)
        get_client().update_current_span(
            input={
                "query": reg_ev.query,
                "evidence_ids": list(all_evidence.keys()),
            },
        )

        # ── Phase 2 (LLM): 판정·근거·risk_level + cited_evidence_ids 선택 ──
        context_str = format_synthesis_input(reg_ev, law_ev, case_ev)
        user_msg = (
            f"질문: {reg_ev.query}\n\n"
            f"수집된 검색 결과:\n{context_str}\n\n"
            f"종합 판정을 JSON 형식으로 출력하세요."
        )

        # 구조화 출력: SynthesisResponse 스키마를 Ollama format= 으로 강제하고
        # 응답을 Pydantic으로 검증해 객체로 받는다. 스키마를 벗어나면
        # ValidationError → "판정 불가" 폴백.
        try:
            parsed = await self._structured_predict_with_repair(
                SynthesisResponse,
                f"{prompt}\n\n{user_msg}",
            )
        except ValidationError as e:
            logger.warning(f"[synthesize] 구조화 출력 검증 실패 → 판정 불가 폴백: {e}")
            return SynthesizedEvent(
                query=reg_ev.query,
                verdict="판정 불가",
                reasoning="LLM 출력이 스키마를 위반하여 판정 보류",
                cited_articles=[],
                risk_level=3,
            )

        # ── Phase 1 (코드): cited_evidence_ids → cited_articles 재구성 ────
        # LLM이 선택한 evidence_id가 실제 all_evidence에 없으면 무시한다.
        # → LLM이 evidence_id를 잘못 기재해도 hallucination이 결과에 반영되지 않는다.
        cited_articles = [
            CitedArticle(
                source_name=all_evidence[eid]["source_name"],
                citation_id=all_evidence[eid]["citation_id"],
            )
            for eid in parsed.cited_evidence_ids
            if eid in all_evidence
        ]

        # Langfuse: Step 출력 기록 (LLM 응답 후)
        get_client().update_current_span(
            output={
                "verdict": parsed.verdict,
                "risk_level": parsed.risk_level,
                "cited_count": len(cited_articles),
                "cited_evidence_ids": parsed.cited_evidence_ids,
            },
        )

        # verdict·risk_level은 스키마로 보장되므로 정규화·클램프가 불필요하다.
        return SynthesizedEvent(
            query=reg_ev.query,
            verdict=parsed.verdict,
            reasoning=parsed.reasoning,
            cited_articles=cited_articles,   # 코드가 재구성한 검증된 목록
            risk_level=parsed.risk_level,
        )

    # =========================================================================
    # Step 4: factcheck_step
    # -------------------------------------------------------------------------
    # 입력: SynthesizedEvent
    # 출력: StopEvent(result=FinalAnswer) 또는 SynthesizedEvent (재시도)
    # Phase 1: article_lookup으로 인용 조항 원문 조회 (Rule-based)
    # Phase 2: LLM이 인용 조항 존재 여부 판정
    # 재시도: 검증 실패 시 SynthesizedEvent 재emit → synthesize_step 재실행
    # =========================================================================

    @step
    @observe(name="factcheck_step", as_type="span")
    async def factcheck_step(
        self,
        ctx: Context,
        ev: SynthesizedEvent
    ) -> StopEvent | SynthesizedEvent:
        """
        합성된 초안의 인용 조항이 실제로 존재하는지 검증한다.

        반환 타입이 StopEvent | SynthesizedEvent인 이유:
          - 검증 통과: StopEvent → 워크플로우 종료
          - 검증 실패: SynthesizedEvent → synthesize_step 재실행 (최대 1회)
          LlamaIndex가 이 유니온 반환 타입을 자동으로 처리한다.

        재시도 루프 방지:
          ctx.store("retry_count")가 MAX_FACTCHECK_RETRY(=1) 이상이면
          더 이상 SynthesizedEvent를 재emit하지 않고 partial 결과로 종료한다.
        """
        # ── Phase 1: article_lookup으로 조항 존재 여부 확인 ────────────────
        # LLM이 "표준투자권유준칙 제5조"를 인용했다고 해서
        # 실제로 존재하는 조항인지는 보장할 수 없음 (hallucination 가능)
        # article_lookup의 exact match로 존재 여부를 확인

        # Langfuse: Step 입력 기록
        get_client().update_current_span(
            input={
                "cited_articles": [c.model_dump() for c in ev.cited_articles],
                "verdict_draft": ev.verdict,
            },
        )

        lookup_results = []
        for cited in ev.cited_articles:
            # 각 인용 조항에 대해 article_lookup 호출
            result = self.registry.article_lookup(
                source_name=cited.source_name,
                citation_id=cited.citation_id,
            )
            lookup_results.append({
                "cited":  cited,
                "found":  result,
                "exists": result is not None,
            })

        if not lookup_results:
            agents_used = await ctx.store.get("agents_used", [])
            return StopEvent(result=FinalAnswer(
                query=ev.query,
                verdict=ev.verdict,
                reasoning=ev.reasoning + " [검증 가능한 인용 없음]",
                cited_articles=[],
                risk_level=max(ev.risk_level, 3),
                factcheck_passed=False,
                agents_used=agents_used,
            ))

        deterministic_failed = [
            f"{item['cited'].source_name}||{item['cited'].citation_id}"
            for item in lookup_results
            if not item["exists"]
        ]

        # ── Phase 2: LLM 존재 여부 판정 ───────────────────────────────────────
        prompt = load_prompt("factcheck_agent")

        # LLM에게 인용 조항 목록과 조회 결과(존재 여부)를 제공
        # 구조화 출력: 존재하지 않는 citation_id 목록을 받는다.
        # 스키마 위반 시 LLM 실패목록은 무시하고 결정론적 검증만 적용한다.
        context_str = format_factcheck_input(ev.verdict, ev.reasoning, lookup_results)
        try:
            parsed = await self._structured_predict_with_repair(
                FactcheckResponse,
                f"{prompt}\n\n{context_str}",
            )
            llm_failed = parsed.failed_items
        except ValidationError as e:
            logger.warning(f"[factcheck] 구조화 출력 검증 실패 → LLM 실패목록 무시: {e}")
            llm_failed = []

        failed_items = sorted(set(llm_failed + deterministic_failed))

        if not failed_items:
            # ── 검증 통과: 워크플로우 종료 ────────────────────────────
            agents_used = await ctx.store.get("agents_used", [])
            get_client().update_current_span(
                output={"factcheck_passed": True, "failed_items": []},
                metadata={"agents_used": str(agents_used)},
            )
            return StopEvent(result=FinalAnswer(
                query=ev.query,
                verdict=ev.verdict,
                reasoning=ev.reasoning,
                cited_articles=ev.cited_articles,
                risk_level=ev.risk_level,
                factcheck_passed=True,
                agents_used=agents_used,
            ))

        # ── 검증 실패: 실패 조항만 제거하고 factcheck_step 재실행 ────
        # SynthesizedEvent를 다시 emit하면 LlamaIndex가 synthesize_step이
        # 아니라 factcheck_step(SynthesizedEvent를 받는 유일한 step)을
        # 다시 트리거한다. 즉 합성은 한 번만, 검증만 재시도하는 구조.
        retry_count: int = await ctx.store.get("retry_count", default=0)
        if retry_count < MAX_FACTCHECK_RETRY:
            await ctx.store.set("retry_count", retry_count + 1)
            logger.warning(f"[factcheck] 검증 실패, 재시도: 실패 항목={failed_items}")
            return SynthesizedEvent(
                query=ev.query,
                verdict=ev.verdict,
                reasoning=ev.reasoning + f" [재시도: {failed_items} 조항 불일치]",
                cited_articles=[
                    c for c in ev.cited_articles
                    if f"{c.source_name}||{c.citation_id}" not in failed_items
                ],
                risk_level=ev.risk_level,
            )

        # 재시도 한도 초과: partial 결과로 강제 종료
        logger.warning("[factcheck] 재시도 한도 초과 → 강제 종료")
        agents_used = await ctx.store.get("agents_used", [])
        get_client().update_current_span(
            output={"factcheck_passed": False, "failed_items": failed_items, "reason": "retry_exceeded"},
            metadata={"agents_used": str(agents_used)},
            level="WARNING",
        )
        return StopEvent(result=FinalAnswer(
            query=ev.query,
            verdict=ev.verdict,
            reasoning=ev.reasoning + " [일부 조항 검증 실패]",
            cited_articles=ev.cited_articles,
            risk_level=ev.risk_level,
            factcheck_passed=False,
            agents_used=agents_used,
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
            # json_mode=True wraps the response in JSON; extract the hypothesis text
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
    # json_mode=True + thinking=False로 LLM이 순수 JSON만 반환하도록 강제했으므로
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
