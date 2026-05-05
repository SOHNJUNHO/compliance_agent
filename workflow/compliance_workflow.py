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
#   Step 1: classify_step   — Rule-based 분류, LLM 없음
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
from pathlib import Path
from typing import Optional

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

# 프롬프트 파일 디렉토리
PROMPT_DIR = Path("prompts")


def load_prompt(name: str) -> str:
    """
    prompts/ 디렉토리에서 에이전트별 시스템 프롬프트를 파일로 로드한다.

    Context Engineering 구현:
      각 에이전트마다 다른 역할/제약/출력형식을 별도 파일로 분리 관리.
      코드에 하드코딩하지 않고 파일로 분리하면:
        - 프롬프트 버전 관리가 가능 (git으로 변경 추적)
        - 코드 수정 없이 프롬프트만 튜닝 가능
        → 포트폴리오 요건 "프롬프트 설계 및 버전 관리" 충족

    Args:
        name: 프롬프트 파일명 (확장자 제외, 예: "synthesize_agent")

    Returns:
        프롬프트 텍스트 문자열
    """
    path = PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"프롬프트 파일 없음: {path}")
    return path.read_text(encoding="utf-8")


# =============================================================================
# Step 1 분류 규칙 (Rule-based)
# =============================================================================

# 키워드 → 에이전트 매핑 딕셔너리
# 질문에 이 키워드가 포함되면 해당 에이전트를 활성화한다
# 여러 에이전트가 동시에 활성화될 수 있음 (AND가 아닌 OR 조건)
CLASSIFY_RULES: dict[str, list[str]] = {
    # 사규 에이전트: 투자 행위와 관련된 키워드
    "규정": ["적합성", "투자권유", "설명의무", "권유", "고령", "파생", "ELS", "내부통제", "준법"],
    # 법규 에이전트: 법률/제도/처벌과 관련된 키워드
    "법규": ["법률", "조항", "자본시장법", "금융투자업", "위반", "제재", "과태료"],
    # 사례 에이전트: 사례/분쟁과 관련된 키워드
    "사례": ["사례", "분쟁", "판례", "피해", "손실보상", "조정"],
}


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
    async def classify_step(
        self,
        ctx: Context,
        ev: StartEvent  # LlamaIndex가 자동으로 이 Step을 시작점으로 인식
    ) -> ClassifiedEvent:
        """
        사용자 질문을 키워드로 분류해 필요한 에이전트 목록을 결정한다.

        LLM을 사용하지 않는 이유:
          라우팅 결정이 LLM에 맡겨지면 같은 질문에 다른 에이전트가 선택될 수 있다.
          워크플로우 진입점은 결정론적이어야 한다.

        ctx 초기화:
          모든 Step이 공유하는 ctx에 초기값을 설정한다.
          이 Step이 항상 첫 번째로 실행되므로 초기화 위치로 적합하다.
        """
        # ev.get(): StartEvent의 kwargs에서 값을 가져옴
        # wf.run(query="질문") 으로 전달된 값
        query: str = ev.get("query", "")

        # ctx 초기값 설정 (이후 Step들이 읽고 업데이트함)
        await ctx.set("token_used", 0)     # 누적 토큰 카운터
        await ctx.set("retry_count", 0)    # factcheck 재시도 카운터
        await ctx.set("agents_used", [])   # 실제 실행된 에이전트 기록

        # CLASSIFY_RULES의 키워드와 질문을 비교해 에이전트 목록 결정
        agent_list = []
        for agent_name, keywords in CLASSIFY_RULES.items():
            # 키워드 중 하나라도 질문에 포함되면 해당 에이전트 활성화
            if any(kw in query for kw in keywords):
                agent_list.append(agent_name)

        # 아무 키워드도 매칭되지 않으면 안전을 위해 전체 활성화
        if not agent_list:
            agent_list = ["규정", "법규", "사례"]
            logger.info("[classify] 키워드 미매칭 → 전체 에이전트 활성화")

        logger.info(f"[classify] 활성화 에이전트: {agent_list}")

        # ClassifiedEvent 반환 → LlamaIndex가 search_* Step들을 트리거
        return ClassifiedEvent(query=query, agent_list=agent_list)

    # =========================================================================
    # Step 2a: search_규정
    # -------------------------------------------------------------------------
    # 입력: ClassifiedEvent
    # 출력: RegulationResultEvent
    # regulation_search 호출 (Rule-based, 고정)
    # =========================================================================

    @step
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
            return RegulationResultEvent(
                query=ev.query, articles=[], summary="(스킵됨)",
                confidence=0.0, evidence=[], skipped=True,
            )

        # ── 검색 함수 실행 (Rule-based, LLM 없음) ────────────────────────
        # regulation_search는 source_type="사규" 필터가 하드코딩됨
        # → 법규나 분쟁사례가 섞일 수 없음
        try:
            precision_filters = self._precision_filters(ev.query, "규정")
            raw_results: list[dict] = self.registry.regulation_search(
                query=ev.query,
                **precision_filters,
            )
            logger.info(f"[search_규정] 검색 완료: {len(raw_results)}개 결과")
        except Exception as e:
            # 검색 자체가 실패하면 (DB 오류 등) 스킵 처리
            logger.warning(f"[search_규정] 검색 실패: {e}")
            return RegulationResultEvent(
                query=ev.query, articles=[], summary="검색 오류",
                confidence=0.0, evidence=[], skipped=True,
            )

        evidence = self._validate_article_evidence(raw_results)
        agents_used = await ctx.get("agents_used", [])
        await ctx.set("agents_used", agents_used + ["규정"])

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
    async def search_법규(
        self,
        ctx: Context,
        ev: ClassifiedEvent
    ) -> LawResultEvent:
        """법규 검색 에이전트. search_규정과 동일한 구조."""

        if "법규" not in ev.agent_list:
            return LawResultEvent(
                query=ev.query, articles=[], summary="(스킵됨)",
                confidence=0.0, evidence=[], skipped=True,
            )

        # law_search 호출 (source_type="법규" 고정)
        try:
            precision_filters = self._precision_filters(ev.query, "법규")
            raw_results = self.registry.law_search(query=ev.query, **precision_filters)
        except Exception as e:
            logger.warning(f"[search_법규] 검색 실패: {e}")
            return LawResultEvent(
                query=ev.query, articles=[], summary="검색 오류",
                confidence=0.0, evidence=[], skipped=True,
            )

        evidence = self._validate_article_evidence(raw_results)
        agents_used = await ctx.get("agents_used", [])
        await ctx.set("agents_used", agents_used + ["법규"])

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
    async def search_사례(
        self,
        ctx: Context,
        ev: ClassifiedEvent
    ) -> CaseResultEvent:
        """분쟁사례 검색 에이전트. search_규정과 동일한 구조."""

        if "사례" not in ev.agent_list:
            return CaseResultEvent(
                query=ev.query, case_nos=[], summary="(스킵됨)",
                confidence=0.0, evidence=[], skipped=True,
            )

        # case_search 호출 (source_type="분쟁사례" 고정)
        try:
            precision_filters = self._precision_filters(ev.query, "사례")
            raw_results = self.registry.case_search(query=ev.query, **precision_filters)
        except Exception as e:
            logger.warning(f"[search_사례] 검색 실패: {e}")
            return CaseResultEvent(
                query=ev.query, case_nos=[], summary="검색 오류",
                confidence=0.0, evidence=[], skipped=True,
            )

        evidence = self._validate_case_evidence(raw_results)
        agents_used = await ctx.get("agents_used", [])
        await ctx.set("agents_used", agents_used + ["사례"])

        return CaseResultEvent(
            query=ev.query,
            case_nos=[item["case_no"] for item in evidence if item.get("case_no")],
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

        lookup_results = []
        for cited in ev.cited_articles:
            # 각 인용 조항에 대해 article_lookup 호출
            result = self.registry.article_lookup(
                source_name=cited.get("source_name", ""),
                citation_id=cited.get("citation_id") or cited.get("article_no", ""),
            )
            lookup_results.append({
                "cited":  cited,           # 인용 정보 {"source_name": ..., "article_no": ...}
                "found":  result,          # DB에서 찾은 원문 (없으면 None)
                "exists": result is not None,  # 존재 여부 플래그
            })

        if not lookup_results:
            token_used = await ctx.get("token_used", 0)
            agents_used = await ctx.get("agents_used", [])
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
            item["cited"].get("citation_id") or item["cited"].get("article_no", "")
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
                    token_used = await ctx.get("token_used", 0)
                    agents_used = await ctx.get("agents_used", [])
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
                    # ── 검증 실패: 재시도 ──────────────────────────────────────
                    try:
                        # check_retry(): MAX_RETRY 초과 시 RetryExceeded 발생
                        await check_retry(ctx)
                        logger.warning(
                            f"[factcheck] 검증 실패, 재시도: 실패 항목={failed_items}"
                        )
                        # 실패한 조항을 제거하고 synthesize_step 재실행
                        # → SynthesizedEvent를 다시 emit하면 LlamaIndex가
                        #   synthesize_step이 아닌 factcheck_step을 트리거하지 않도록
                        #   주의: 이 경우 re-emit이 synthesize_step을 직접 트리거하지 않음
                        #   실제로는 조항을 제거한 SynthesizedEvent를 factcheck_step이 다시 받음
                        return SynthesizedEvent(
                            query=ev.query,
                            verdict=ev.verdict,
                            reasoning=ev.reasoning + f" [재시도: {failed_items} 조항 불일치]",
                            # 검증 실패한 조항을 제거한 리스트
                            cited_articles=[
                                c for c in ev.cited_articles
                                if (c.get("citation_id") or c.get("article_no")) not in failed_items
                            ],
                            risk_level=ev.risk_level,
                            retry_count=ev.retry_count + 1,  # 재시도 카운터 증가
                        )

                    except RetryExceeded:
                        # 재시도 횟수 초과: partial 결과로 강제 종료
                        logger.warning("[factcheck] 재시도 한도 초과 → 강제 종료")
                        token_used = await ctx.get("token_used", 0)
                        agents_used = await ctx.get("agents_used", [])
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
            token_used = await ctx.get("token_used", 0)
            agents_used = await ctx.get("agents_used", [])
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

    def _format_search_results(self, results: list[dict]) -> str:
        """
        검색 결과 리스트를 LLM이 읽기 좋은 텍스트로 포맷한다.
        본문은 300자로 잘라서 컨텍스트 토큰을 절약한다.
        """
        if not results:
            return "(검색 결과 없음)"

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] {r.get('source_name','')} {r.get('citation_id','')}\n"
                f"    {r.get('text','')[:300]}\n"   # 300자 제한으로 토큰 절약
                f"    (유사도: {r.get('score', 0):.2f})"
            )
        return "\n".join(lines)

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
            citation_id = cited.get("citation_id") or cited.get("article_no", "")
            lines.append(
                f"  - {cited.get('source_name','')} {citation_id} : {status}"
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
