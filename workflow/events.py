# =============================================================================
# events.py
# -----------------------------------------------------------------------------
# 역할: 워크플로우의 모든 Event 클래스를 정의한다.
#
# LlamaIndex Workflow의 핵심 원리:
#   Step 간의 유일한 통신 수단이 Event이다.
#   Step은 특정 타입의 Event를 받아서, 다른 타입의 Event를 반환한다.
#   LlamaIndex가 타입 어노테이션을 분석해서 실행 순서(DAG)를 자동으로 결정한다.
#
# 타입이 곧 라우팅 규칙:
#   @step
#   async def search_규정(self, ev: ClassifiedEvent) -> RegulationResultEvent:
#   → "ClassifiedEvent가 도착하면 이 Step을 실행하고 RegulationResultEvent를 반환"
#   → LLM이나 외부 설정 없이 코드 자체가 흐름을 보장한다
#
# 이벤트 흐름 (DAG):
#   StartEvent
#     ↓ classify_step
#   ClassifiedEvent
#     ↓ (병렬) search_규정 / search_법규 / search_사례
#   RegulationResultEvent + LawResultEvent + CaseResultEvent
#     ↓ synthesize_step (3개 모두 도착 후 실행)
#   SynthesizedEvent
#     ↓ factcheck_step
#   StopEvent(result=FinalAnswer)
# =============================================================================

from typing import Optional
from llama_index.core.workflow import Event  # LlamaIndex Event 기반 클래스


# =============================================================================
# classify_step 출력 → search_* steps 입력
# =============================================================================

class ClassifiedEvent(Event):
    """
    classify_step이 반환하는 이벤트.
    Rule-based 분류 결과를 담는다.

    agent_list 예시:
      ["규정"]           → 사규 에이전트만 실행
      ["규정", "법규"]   → 두 에이전트 병렬 실행
      ["규정", "법규", "사례"] → 전체 실행

    이 리스트는 각 search_* step이 자신을 실행해야 하는지 확인하는 데 사용된다.
    예: search_규정 step은 "규정" in ev.agent_list 로 실행 여부를 결정
    """
    query:      str         # 사용자의 원본 질문 (이후 모든 Step에 전달됨)
    agent_list: list[str]   # 활성화할 에이전트 이름 목록


# =============================================================================
# search_* steps 출력 → synthesize_step 입력
# (세 이벤트가 모두 도착해야 synthesize_step이 실행됨)
# =============================================================================

class RegulationResultEvent(Event):
    """
    search_규정 step(Step 2a)의 출력 이벤트.
    사규 검색 결과를 담는다.

    skipped=True인 경우:
      - agent_list에 "규정"이 없어서 실행 자체를 건너뜀
      - circuit breaker가 발동해서 중단됨
      → synthesize_step은 skipped 여부를 확인해서 "검색 결과 없음"으로 처리
    """
    query:      str
    articles:   list[str]   # 검색된 관련 조항 목록 (예: ["표준투자권유준칙 제5조"])
    summary:    str          # LLM이 생성한 한 문장 요약
    confidence: float        # LLM의 자신감 점수 (0.0~1.0)
    evidence:   list[dict]  # 검증된 검색 근거 객체 목록
    skipped:    bool = False # circuit breaker 발동 또는 agent_list에 없으면 True


class LawResultEvent(Event):
    """
    search_법규 step(Step 2b)의 출력 이벤트.
    법규 검색 결과를 담는다. RegulationResultEvent와 동일한 구조.
    """
    query:      str
    articles:   list[str]   # 예: ["자본시장법 제46조"]
    summary:    str
    confidence: float
    evidence:   list[dict]
    skipped:    bool = False


class CaseResultEvent(Event):
    """
    search_사례 step(Step 2c)의 출력 이벤트.
    분쟁사례 검색 결과를 담는다.

    case_nos: 사건번호 리스트 (articles 대신 case_nos 사용)
    """
    query:      str
    case_nos:   list[str]   # 예: ["2022-증권-031"]
    summary:    str
    confidence: float
    evidence:   list[dict]
    skipped:    bool = False


# =============================================================================
# synthesize_step 출력 → factcheck_step 입력
# =============================================================================

class SynthesizedEvent(Event):
    """
    synthesize_step(Step 3)의 출력 이벤트.
    3개 검색 결과를 합성한 초안을 담는다.

    cited_articles가 factcheck의 핵심 입력:
      factcheck_step은 이 리스트의 각 조항이 실제로 존재하는지 검증한다.
      예: [{"source_name": "표준투자권유준칙", "article_no": "제5조"}, ...]

    retry_count:
      factcheck 실패 시 이 이벤트가 재emit된다 (최대 1회).
      retry_count로 무한 루프를 방지한다.
    """
    query:           str
    verdict:         str          # "가능" | "불가" | "조건부 가능" (LLM이 생성)
    reasoning:       str          # 판정 근거 설명
    cited_articles:  list[dict]   # 인용된 조항 목록
                                  # 형식: [{"source_name": "...", "citation_id": "..."}, ...]
                                  # synthesize_step에서 코드가 evidence_id로부터 재구성한다
    risk_level:      int          # 위험 수준: 1(저), 2(중), 3(고)
    retry_count:     int = 0      # factcheck 재시도 횟수 (기본값 0)


# =============================================================================
# factcheck_step 출력 → StopEvent의 result
# =============================================================================

class FinalAnswer(Event):
    """
    워크플로우의 최종 결과.
    StopEvent(result=FinalAnswer(...))로 전달된다.

    main.py에서 이 객체의 필드를 출력한다.
    포트폴리오 데모 시 이 구조가 "에이전트가 구조화된 답변을 생성한다"는 것을 보여준다.
    """
    query:            str
    verdict:          str          # 최종 판정
    reasoning:        str          # 최종 근거
    cited_articles:   list[dict]   # 검증 완료된 인용 조항
    risk_level:       int          # 위험 수준
    factcheck_passed: bool         # 팩트체크 통과 여부 (신뢰도 지표)

    # 포트폴리오 설명용 메타 정보
    agents_used:      list[str]    # 실제 실행된 에이전트 목록 (예: ["규정", "법규"])
    token_used:       int          # 총 누적 토큰 사용량 (circuit breaker 효과 확인용)
