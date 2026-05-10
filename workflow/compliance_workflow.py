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
#   (신한투자증권 공고의 핵심 요구사항과 일치)
# =============================================================================

import logging
import re
from typing import Optional

from langfuse import observe, get_client

from llama_index.core.workflow import (
    Workflow,      # 워크플로우 기반 클래스
    StartEvent,    # 워크플로우 시작 이벤트 (LlamaIndex 내장)
    StopEvent,     # 워크플로우 종료 이벤트 (LlamaIndex 내장)
    Context,       # Step 간 공유 상태 저장소
    step,          # Step 정의 데코레이터
)
from llama_index.core.llms import LLM  # LLM 기반 클래스 (타입 힌트용)

from events import (
    ClassifiedEvent,
    RegulationResultEvent,
    LawResultEvent,
    CaseResultEvent,
    SynthesizedEvent,
    FinalAnswer,
)
from tools import ToolRegistry
from circuit_breaker import (
    token_guard,
    check_retry,
    record_token_usage,
    BudgetExceeded,
    RetryExceeded,
)

logger = logging.getLogger(__name__)

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
    from langfuse_setup import get_langfuse_prompt
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
          source_name·citation_id 정밀 필터는 _precision_filters()에서 regex로만 처리한다.
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
        await ctx.store.set("token_used", 0)     # 누적 토큰 카운터
        await ctx.store.set("retry_count", 0)    # factcheck 재시도 카운터
        await ctx.store.set("agents_used", [])   # 실제 실행된 에이전트 기록

        # ── LLM 라우팅 ────────────────────────────────────────────────────────
        prompt = load_prompt("classify_agent")
        response = await self.llm.acomplete(f"{prompt}\n\n질문: {query}")
        parsed = self._safe_json(response.text)

        # LLM 출력에서 허용된 값만 통과 (hallucination 차단)
        raw_agents = parsed.get("agents", [])
        agent_list = [a for a in raw_agents if a in {"규정", "법규", "사례"}]
        routing_reasoning = parsed.get("reasoning", "")

        # ── 안전 fallback ────────────────────────────────────────────────────
        # LLM이 유효한 에이전트를 하나도 반환하지 못한 경우 전체 레인을 열어
        # 관련 근거를 누락하는 것보다 불필요한 검색이 낫다는 원칙을 적용한다.
        if not agent_list:
            agent_list = ["규정", "법규", "사례"]
            routing_reasoning += " [fallback: 전체 활성화]"
            logger.warning("[classify] LLM 출력 파싱 실패 → 전체 에이전트 활성화")

        logger.info(f"[classify] 활성화 에이전트: {agent_list} | 근거: {routing_reasoning}")

        # Langfuse: routing_reasoning을 출력으로 기록 → 대시보드에서 LLM 판단 근거 확인 가능
        get_client().update_current_span(
            output={"agent_list": agent_list, "routing_reasoning": routing_reasoning}
        )

        return ClassifiedEvent(query=query, agent_list=agent_list)

    # =========================================================================
    # Step 2a: search_규정
    # -------------------------------------------------------------------------
    # 입력: ClassifiedEvent
    # 출력: RegulationResultEvent
    # regulation_search 호출 (Rule-based, 고정)
    # =========================================================================

    @step
    @observe(name="search_규정", as_type="span")
    async def search_규정(
        self,
        ctx: Context,
        ev: ClassifiedEvent  # classify_step의 출력을 받음
    ) -> RegulationResultEvent:
        """
        사규 검색 에이전트.

        "규정"이 agent_list에 없으면 스킵 → skipped=True 이벤트 반환.
        synthesize_step은 skipped 이벤트도 집계하므로 워크플로우가 중단되지 않는다.
        """
        # agent_list에 "규정"이 없으면 이 에이전트는 필요 없음
        if "규정" not in ev.agent_list:
            logger.info("[search_규정] 스킵 (agent_list에 없음)")
            get_client().update_current_span(
                input={"query": ev.query},
                output={"skipped": True},
            )
            return RegulationResultEvent(
                query=ev.query, articles=[], summary="(스킵됨)",
                confidence=0.0, evidence=[], skipped=True,
            )

        # ── 검색 함수 실행 (Rule-based, LLM 없음) ────────────────────────
        # regulation_search는 source_type="사규" 필터가 하드코딩됨
        # → 법규나 분쟁사례가 섞일 수 없음
        try:
            precision_filters = self._precision_filters(ev.query, "규정")
            # HyDE: 원본 쿼리를 가상 법령 조항으로 변환해 벡터 공간 정렬 개선
            # precision_filters는 원본 쿼리 기준으로 추출 (명시된 조항번호·문서명 regex)
            hyde_query = await self._hyde_transform(ev.query)
            get_client().update_current_span(
                input={
                    "query": ev.query,
                    "hyde_query": hyde_query,
                    "precision_filters": precision_filters,
                },
            )
            raw_results: list[dict] = self.registry.regulation_search(
                query=hyde_query,   # 임베딩에는 가상 조항 사용
                **precision_filters,
            )
            logger.info(f"[search_규정] 검색 완료: {len(raw_results)}개 결과")
        except Exception as e:
            # 검색 자체가 실패하면 (DB 오류 등) 스킵 처리
            logger.warning(f"[search_규정] 검색 실패: {e}")
            get_client().update_current_span(
                output={"error": str(e), "skipped": True},
                level="ERROR",
            )
            return RegulationResultEvent(
                query=ev.query, articles=[], summary="검색 오류",
                confidence=0.0, evidence=[], skipped=True,
            )

        evidence = self._validate_article_evidence(raw_results)
        agents_used = await ctx.store.get("agents_used", [])
        await ctx.store.set("agents_used", agents_used + ["규정"])

        get_client().update_current_span(
            output={
                "evidence_count": len(evidence),
                "articles": self._evidence_article_labels(evidence),
            },
        )
        return RegulationResultEvent(
            query=ev.query,
            articles=self._evidence_article_labels(evidence),
            summary=f"검증된 사규 근거 {len(evidence)}개",
            confidence=1.0 if evidence else 0.0,
            evidence=evidence,
        )

    # =========================================================================
    # Step 2b: search_법규
    # -------------------------------------------------------------------------
    # Step 2a와 동일한 구조. 검색 함수만 law_search로 다름.
    # =========================================================================

    @step
    @observe(name="search_법규", as_type="span")
    async def search_법규(
        self,
        ctx: Context,
        ev: ClassifiedEvent
    ) -> LawResultEvent:
        """법규 검색 에이전트. search_규정과 동일한 구조."""

        if "법규" not in ev.agent_list:
            get_client().update_current_span(
                input={"query": ev.query},
                output={"skipped": True},
            )
            return LawResultEvent(
                query=ev.query, articles=[], summary="(스킵됨)",
                confidence=0.0, evidence=[], skipped=True,
            )

        # law_search 호출 (source_type="법규" 고정)
        try:
            precision_filters = self._precision_filters(ev.query, "법규")
            hyde_query = await self._hyde_transform(ev.query)
            get_client().update_current_span(
                input={
                    "query": ev.query,
                    "hyde_query": hyde_query,
                    "precision_filters": precision_filters,
                },
            )
            raw_results = self.registry.law_search(query=hyde_query, **precision_filters)
        except Exception as e:
            logger.warning(f"[search_법규] 검색 실패: {e}")
            get_client().update_current_span(
                output={"error": str(e), "skipped": True},
                level="ERROR",
            )
            return LawResultEvent(
                query=ev.query, articles=[], summary="검색 오류",
                confidence=0.0, evidence=[], skipped=True,
            )

        evidence = self._validate_article_evidence(raw_results)
        agents_used = await ctx.store.get("agents_used", [])
        await ctx.store.set("agents_used", agents_used + ["법규"])

        get_client().update_current_span(
            output={
                "evidence_count": len(evidence),
                "articles": self._evidence_article_labels(evidence),
            },
        )
        return LawResultEvent(
            query=ev.query,
            articles=self._evidence_article_labels(evidence),
            summary=f"검증된 법규 근거 {len(evidence)}개",
            confidence=1.0 if evidence else 0.0,
            evidence=evidence,
        )

    # =========================================================================
    # Step 2c: search_사례
    # -------------------------------------------------------------------------
    # Step 2a와 동일한 구조. 검색 함수만 case_search로 다름.
    # =========================================================================

    @step
    @observe(name="search_사례", as_type="span")
    async def search_사례(
        self,
        ctx: Context,
        ev: ClassifiedEvent
    ) -> CaseResultEvent:
        """분쟁사례 검색 에이전트. search_규정과 동일한 구조."""

        if "사례" not in ev.agent_list:
            get_client().update_current_span(
                input={"query": ev.query},
                output={"skipped": True},
            )
            return CaseResultEvent(
                query=ev.query, case_nos=[], summary="(스킵됨)",
                confidence=0.0, evidence=[], skipped=True,
            )

        # case_search 호출 (source_type="분쟁사례" 고정)
        try:
            precision_filters = self._precision_filters(ev.query, "사례")
            hyde_query = await self._hyde_transform(ev.query)
            get_client().update_current_span(
                input={
                    "query": ev.query,
                    "hyde_query": hyde_query,
                    "precision_filters": precision_filters,
                },
            )
            raw_results = self.registry.case_search(query=hyde_query, **precision_filters)
        except Exception as e:
            logger.warning(f"[search_사례] 검색 실패: {e}")
            get_client().update_current_span(
                output={"error": str(e), "skipped": True},
                level="ERROR",
            )
            return CaseResultEvent(
                query=ev.query, case_nos=[], summary="검색 오류",
                confidence=0.0, evidence=[], skipped=True,
            )

        evidence = self._validate_case_evidence(raw_results)
        agents_used = await ctx.store.get("agents_used", [])
        await ctx.store.set("agents_used", agents_used + ["사례"])

        case_nos = [item["case_no"] for item in evidence if item.get("case_no")]
        get_client().update_current_span(
            output={"evidence_count": len(evidence), "case_nos": case_nos},
        )
        return CaseResultEvent(
            query=ev.query,
            case_nos=case_nos,
            summary=f"검증된 사례 근거 {len(evidence)}개",
            confidence=1.0 if evidence else 0.0,
            evidence=evidence,
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

        estimated_tokens = 1_000  # 합성은 가장 많은 토큰 사용

        try:
            async with token_guard(ctx, estimated_tokens):
                prompt = load_prompt("synthesize_agent")

                # ── Phase 1 (코드): 검증된 근거 목록을 evidence_id 기준으로 색인 ────
                # LLM이 cited_articles를 직접 생성하면 citation_id 형식이 달라져
                # article_lookup.json 키와 불일치할 수 있다.
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
                context_str = self._format_synthesis_input(reg_ev, law_ev, case_ev)
                user_msg = (
                    f"질문: {reg_ev.query}\n\n"
                    f"수집된 검색 결과:\n{context_str}\n\n"
                    f"종합 판정을 JSON 형식으로 출력하세요."
                )

                response = await self.llm.acomplete(f"{prompt}\n\n{user_msg}")
                parsed = self._parse_synthesis_response(response.text)
                await record_token_usage(ctx, estimated_tokens)

                # ── Phase 1 (코드): cited_evidence_ids → cited_articles 재구성 ────
                # LLM이 선택한 evidence_id가 실제 all_evidence에 없으면 무시한다.
                # → LLM이 evidence_id를 잘못 기재해도 hallucination이 결과에 반영되지 않는다.
                cited_articles = [
                    {
                        "source_name": all_evidence[eid]["source_name"],
                        "citation_id":  all_evidence[eid]["citation_id"],
                    }
                    for eid in parsed.get("cited_evidence_ids", [])
                    if eid in all_evidence
                ]

                # Langfuse: Step 출력 기록 (LLM 응답 후)
                token_used = await ctx.store.get("token_used", 0)
                get_client().update_current_span(
                    output={
                        "verdict": parsed.get("verdict"),
                        "risk_level": parsed.get("risk_level"),
                        "cited_count": len(cited_articles),
                        "cited_evidence_ids": parsed.get("cited_evidence_ids", []),
                    },
                    metadata={"token_used": str(token_used)},
                )

                return SynthesizedEvent(
                    query=reg_ev.query,
                    verdict=parsed.get("verdict", "판정 불가"),
                    reasoning=parsed.get("reasoning", ""),
                    cited_articles=cited_articles,   # 코드가 재구성한 검증된 목록
                    risk_level=parsed.get("risk_level", 2),
                )

        except BudgetExceeded:
            # 합성도 실패하면 partial 결과로 강제 종료 (무한 대기 방지)
            logger.warning("[synthesize] 토큰 초과 → 부분 결과로 진행")
            return SynthesizedEvent(
                query=reg_ev.query,
                verdict="판정 불가 (토큰 초과)",
                reasoning="토큰 예산 초과로 합성 생략",
                cited_articles=[],
                risk_level=3,   # 불확실하므로 최고 위험 등급으로 보수적 처리
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
          check_retry()가 MAX_RETRY(=1) 초과 시 RetryExceeded 발생
          → 재시도 없이 partial 결과로 종료
        """
        # ── Phase 1: article_lookup으로 조항 존재 여부 확인 ────────────────
        # LLM이 "표준투자권유준칙 제5조"를 인용했다고 해서
        # 실제로 존재하는 조항인지는 보장할 수 없음 (hallucination 가능)
        # article_lookup의 exact match로 존재 여부를 확인

        # Langfuse: Step 입력 기록
        get_client().update_current_span(
            input={
                "cited_articles": ev.cited_articles,
                "verdict_draft": ev.verdict,
            },
        )

        lookup_results = []
        for cited in ev.cited_articles:
            # 각 인용 조항에 대해 article_lookup 호출
            # cited_articles는 synthesize_step에서 {source_name, citation_id}로 생성됨
            result = self.registry.article_lookup(
                source_name=cited.get("source_name", ""),
                citation_id=cited.get("citation_id", ""),
            )
            lookup_results.append({
                "cited":  cited,
                "found":  result,
                "exists": result is not None,
            })

        if not lookup_results:
            token_used = await ctx.store.get("token_used", 0)
            agents_used = await ctx.store.get("agents_used", [])
            return StopEvent(result=FinalAnswer(
                query=ev.query,
                verdict=ev.verdict,
                reasoning=ev.reasoning + " [검증 가능한 인용 없음]",
                cited_articles=[],
                risk_level=max(ev.risk_level, 3),
                factcheck_passed=False,
                agents_used=agents_used,
                token_used=token_used,
            ))

        deterministic_failed = [
            item["cited"].get("citation_id", "")
            for item in lookup_results
            if not item["exists"]
        ]

        # ── Phase 2: LLM 존재 여부 판정 ───────────────────────────────────────
        estimated_tokens = 600

        try:
            async with token_guard(ctx, estimated_tokens):
                prompt = load_prompt("factcheck_agent")

                # LLM에게 인용 조항 목록과 조회 결과(존재 여부)를 제공
                context_str = self._format_factcheck_input(ev, lookup_results)
                response = await self.llm.acomplete(f"{prompt}\n\n{context_str}")
                parsed = self._parse_factcheck_response(response.text)
                await record_token_usage(ctx, estimated_tokens)

                failed_items = sorted(set(parsed.get("failed_items", []) + deterministic_failed))

                if not failed_items:
                    # ── 검증 통과: 워크플로우 종료 ────────────────────────────
                    token_used = await ctx.store.get("token_used", 0)
                    agents_used = await ctx.store.get("agents_used", [])
                    get_client().update_current_span(
                        output={"factcheck_passed": True, "failed_items": []},
                        metadata={"agents_used": str(agents_used), "token_used": str(token_used)},
                    )
                    return StopEvent(result=FinalAnswer(
                        query=ev.query,
                        verdict=ev.verdict,
                        reasoning=ev.reasoning,
                        cited_articles=ev.cited_articles,
                        risk_level=ev.risk_level,
                        factcheck_passed=True,
                        agents_used=agents_used,
                        token_used=token_used,
                    ))

                else:
                    # ── 검증 실패: 실패 조항만 제거하고 factcheck_step 재실행 ────
                    # SynthesizedEvent를 다시 emit하면 LlamaIndex가 synthesize_step이
                    # 아니라 factcheck_step(SynthesizedEvent를 받는 유일한 step)을
                    # 다시 트리거한다. 즉 합성은 한 번만, 검증만 재시도하는 구조.
                    try:
                        # check_retry(): MAX_RETRY 초과 시 RetryExceeded 발생
                        await check_retry(ctx)
                        logger.warning(
                            f"[factcheck] 검증 실패, 재시도: 실패 항목={failed_items}"
                        )
                        return SynthesizedEvent(
                            query=ev.query,
                            verdict=ev.verdict,
                            reasoning=ev.reasoning + f" [재시도: {failed_items} 조항 불일치]",
                            cited_articles=[
                                c for c in ev.cited_articles
                                if c.get("citation_id") not in failed_items
                            ],
                            risk_level=ev.risk_level,
                        )

                    except RetryExceeded:
                        # 재시도 횟수 초과: partial 결과로 강제 종료
                        logger.warning("[factcheck] 재시도 한도 초과 → 강제 종료")
                        token_used = await ctx.store.get("token_used", 0)
                        agents_used = await ctx.store.get("agents_used", [])
                        get_client().update_current_span(
                            output={"factcheck_passed": False, "failed_items": failed_items, "reason": "retry_exceeded"},
                            metadata={"agents_used": str(agents_used), "token_used": str(token_used)},
                            level="WARNING",
                        )
                        return StopEvent(result=FinalAnswer(
                            query=ev.query,
                            verdict=ev.verdict,
                            reasoning=ev.reasoning + " [일부 조항 검증 실패]",
                            cited_articles=ev.cited_articles,
                            risk_level=ev.risk_level,
                            factcheck_passed=False,  # 검증 실패 표시
                            agents_used=agents_used,
                            token_used=token_used,
                        ))

        except BudgetExceeded:
            # 팩트체크 단계에서 토큰 초과: 검증 생략하고 종료
            token_used = await ctx.store.get("token_used", 0)
            agents_used = await ctx.store.get("agents_used", [])
            return StopEvent(result=FinalAnswer(
                query=ev.query,
                verdict=ev.verdict,
                reasoning=ev.reasoning + " [팩트체크 생략 - 토큰 초과]",
                cited_articles=ev.cited_articles,
                risk_level=ev.risk_level,
                factcheck_passed=False,
                agents_used=agents_used,
                token_used=token_used,
            ))

    # =========================================================================
    # 헬퍼 메서드: 포맷터 / 파서
    # =========================================================================

    async def _hyde_transform(self, query: str) -> str:
        """
        HyDE (Hypothetical Document Embedding): 원본 질문을 가상 법령 조항으로 변환한다.

        왜 필요한가:
          질문("65세 고객에게 레버리지 ETF 권유 가능한가요?")과 코퍼스("금융투자업자는
          고령투자자에 대해 강화된 적합성 확인 절차를 적용하여야 한다")는 언어 형식과
          어휘가 달라 임베딩 벡터 공간에서 멀리 위치할 수 있다.
          LLM이 코퍼스와 같은 법령 문체로 가상 조항을 생성하면 임베딩 공간에서
          실제 조항과 가까워져 검색 정밀도가 높아진다.

        안전성:
          가상 조항은 임베딩에만 사용되고 최종 답변에는 노출되지 않는다.
          조항 번호 인용을 금지하므로 잘못된 citation이 생성되지 않는다.
          실패 시 원본 쿼리를 반환하므로 워크플로우가 중단되지 않는다.

        Reference: Gao et al. (2022) "Precise Zero-Shot Dense Retrieval without
                   Relevance Labels", arXiv:2212.10496
        """
        try:
            prompt = load_prompt("hyde_agent").replace("{query}", query)
            response = await self.llm.acomplete(prompt)
            # json_mode=True wraps the response in JSON; extract the hypothesis text
            parsed = self._safe_json(response.text)
            hypothesis = parsed.get("hypothesis", "").strip()
            if len(hypothesis) >= 20:
                logger.info(f"[hyde] 가상 조항 생성 완료: {hypothesis[:80]}...")
                return hypothesis
            logger.warning("[hyde] 생성된 가상 조항이 너무 짧음 → 원본 쿼리 사용")
        except Exception as e:
            logger.warning(f"[hyde] 생성 실패, 원본 쿼리 사용: {e}")
        return query

    def _precision_filters(self, query: str, agent: str) -> dict:
        """
        쿼리 텍스트에서 고신뢰 신호만 추출해 메타데이터 precision filter로 변환한다.
        source_type은 tools.py에서 이미 강제되므로 여기서는 추가 필터만 반환한다.

        포함하는 필터:
          source_name — 사용자가 쿼리에 문서명을 명시한 경우 (확실한 신호)
          citation_id — 쿼리에 "제N조" 또는 "N.N.N" 섹션번호가 있는 경우 (확실한 신호)

        제외하는 필터:
          category   — _classify_category()의 키워드 분류는 오탐률이 높아
                       AND 조건으로 적용하면 유효한 청크를 통째로 걸러낼 수 있음.
                       예: 제47조가 category="기타"로 분류된 경우 "설명의무" 필터가
                           해당 조항을 검색 결과에서 제외한다.
        """
        filters: dict[str, str] = {}

        # ── source_name 필터: 쿼리에 문서명이 직접 언급된 경우만 적용 ──────────
        if agent == "규정":
            if "표준투자권유준칙" in query:
                filters["source_name"] = "표준투자권유준칙"
            elif "내부통제" in query or "준법감시" in query:
                filters["source_name"] = "금융투자회사표준내부통제기준"
        elif agent == "법규":
            if "자본시장법" in query or "금융투자업" in query:
                filters["source_name"] = "자본시장과금융투자업에관한법률"

        # ── citation_id 필터: 조항번호·섹션번호가 명시된 경우만 적용 ────────────
        article_match = re.search(r"제\d+조(?:의\d+)?(?:\([^)]*\))?", query)
        if article_match and agent in {"규정", "법규"}:
            # 괄호 제목 부분 제거: "제47조(설명의무)" → "제47조"
            filters["citation_id"] = re.sub(r"\([^)]*\)$", "", article_match.group(0))

        section_match = re.search(r"\b\d+(?:\.\d+)+\b", query)
        if section_match and agent == "규정":
            filters["citation_id"] = section_match.group(0)

        return filters

    def _validate_article_evidence(self, raw_results: list[dict]) -> list[dict]:
        """사규/법규 검색 결과를 exact lookup으로 즉시 검증한다."""
        evidence = []
        for item in raw_results:
            source_name = item.get("source_name", "")
            citation_id = item.get("citation_id") or item.get("article_no", "")
            if not source_name or not citation_id:
                continue
            lookup = self.registry.article_lookup(
                source_name=source_name,
                citation_id=citation_id,
            )
            if not lookup:
                continue
            verified = dict(item)
            verified["verified"] = True
            verified["text"] = lookup.get("text") or item.get("text", "")
            verified["citation_id"] = lookup.get("citation_id") or citation_id
            verified["article_no"] = lookup.get("article_no") or item.get("article_no", "")
            verified["section_no"] = lookup.get("section_no") or item.get("section_no", "")
            verified["case_no"] = lookup.get("case_no") or item.get("case_no", "")
            verified["evidence_id"] = f"{source_name}||{verified['citation_id']}"
            evidence.append(verified)
        return evidence

    def _validate_case_evidence(self, raw_results: list[dict]) -> list[dict]:
        """사례 검색 결과는 case_no metadata 존재 여부로 검증한다."""
        evidence = []
        for item in raw_results:
            case_no = item.get("case_no", "")
            if not case_no:
                continue
            verified = dict(item)
            verified["verified"] = True
            verified["citation_id"] = item.get("citation_id") or case_no
            verified["evidence_id"] = f"{item.get('source_name', '법원판례')}||{verified['citation_id']}"
            evidence.append(verified)
        return evidence

    @staticmethod
    def _evidence_article_labels(evidence: list[dict]) -> list[str]:
        return [
            f"{item.get('source_name', '')} {item.get('citation_id', '')}".strip()
            for item in evidence
            if item.get("source_name") and item.get("citation_id")
        ]

    def _format_synthesis_input(
        self,
        reg: RegulationResultEvent,
        law: LawResultEvent,
        case: CaseResultEvent,
    ) -> str:
        """
        3개 에이전트 결과를 synthesize LLM에게 전달할 형태로 포맷한다.
        skipped된 에이전트는 "(결과 없음)"으로 표시.
        """
        parts = []

        if not reg.skipped:
            parts.append("[사규 검증 근거]\n" + self._format_evidence_for_synthesis(reg.evidence))
        else:
            parts.append("[사규 검색 결과] (결과 없음)")

        if not law.skipped:
            parts.append("[법규 검증 근거]\n" + self._format_evidence_for_synthesis(law.evidence))
        else:
            parts.append("[법규 검색 결과] (결과 없음)")

        if not case.skipped:
            parts.append("[분쟁사례 검증 근거]\n" + self._format_evidence_for_synthesis(case.evidence))
        else:
            parts.append("[분쟁사례 검색 결과] (결과 없음)")

        return "\n\n".join(parts)

    def _format_evidence_for_synthesis(self, evidence: list[dict]) -> str:
        """검증된 evidence 객체를 최종 합성 LLM에 전달할 형식으로 변환한다."""
        if not evidence:
            return "(검증된 근거 없음)"
        lines = []
        for i, item in enumerate(evidence, 1):
            citation = item.get("citation_id") or item.get("article_no") or item.get("case_no", "")
            lines.append(
                f"[E{i}] evidence_id={item.get('evidence_id', '')}\n"
                f"source_type={item.get('source_type', '')}\n"
                f"source_name={item.get('source_name', '')}\n"
                f"citation={citation}\n"
                f"article_no={item.get('article_no', '')}\n"
                f"section_no={item.get('section_no', '')}\n"
                f"case_no={item.get('case_no', '')}\n"
                f"verified={item.get('verified', False)}\n"
                f"score={item.get('score', 0):.4f}\n"
                f"text={item.get('text', '')[:900]}"
            )
        return "\n\n".join(lines)

    def _format_factcheck_input(
        self,
        ev: SynthesizedEvent,
        lookups: list[dict],
    ) -> str:
        """
        factcheck LLM에게 전달할 컨텍스트를 포맷한다.
        인용 조항 목록과 조회 결과(존재/미존재)를 포함한다.
        """
        lines = [
            f"판정 초안: {ev.verdict}",
            f"근거: {ev.reasoning}",
            "",
            "인용 조항 검증 결과:",
        ]
        for item in lookups:
            status = "✓ 존재" if item["exists"] else "✗ 미존재"
            cited = item["cited"]
            lines.append(
                f"  - {cited.get('source_name','')} {cited.get('citation_id','')} : {status}"
            )
        return "\n".join(lines)

    # ── LLM 응답 JSON 파서들 ────────────────────────────────────────────────

    @staticmethod
    def _safe_json(text: str) -> dict:
        """
        LLM 응답 텍스트에서 첫 번째 유효한 JSON 객체를 안전하게 추출한다.

        기존 방식의 문제:
          re.search(r'\\{.*\\}', text, re.DOTALL) — greedy 매칭.
          LLM이 JSON 뒤에 설명 문장을 추가하면 마지막 }까지 통째로 잡아
          json.loads()가 실패하고 빈 dict를 반환한다.

        개선 방식:
          json.JSONDecoder().raw_decode()는 첫 번째 완전한 JSON 객체만
          파싱하고 나머지 텍스트(마크다운 설명 등)는 무시한다.
          JSON 앞에 텍스트가 있으면 { 위치를 찾아 거기서부터 시도한다.
        """
        import json

        text = text.strip()
        # { 가 처음 등장하는 위치부터 순서대로 시도
        for start in range(len(text)):
            if text[start] != "{":
                continue
            try:
                obj, _ = json.JSONDecoder().raw_decode(text, start)
                return obj
            except json.JSONDecodeError:
                continue  # 이 위치는 유효한 JSON 시작점이 아님 → 다음 시도

        logger.warning(f"JSON 없음: {text[:100]}")
        return {}

    def _parse_synthesis_response(self, text: str) -> dict:
        return self._safe_json(text)

    def _parse_factcheck_response(self, text: str) -> dict:
        return self._safe_json(text)
