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

from typing import Annotated, Literal

from llama_index.core.workflow import Event  # LlamaIndex Event 기반 클래스
from pydantic import BaseModel, Field  # Event가 Pydantic BaseModel이므로 제약 조건을 그대로 사용 가능

AgentName = Literal["규정", "법규", "사례"]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]  # LLM 자신감 점수 (범위 강제)
LLMVerdict = Literal["가능", "불가", "조건부 가능"]    # LLM이 직접 고르는 판정 (단일 출처)
Verdict = LLMVerdict | Literal["판정 불가"]            # + 코드 폴백(토큰 초과·검증 실패 등) = 상위 집합


class CitedArticle(BaseModel):
    """
    검증 완료된 인용 조항.

    synthesize_step이 evidence_id로부터 코드로 재구성하고,
    factcheck_step이 존재 여부를 검증하며, FinalAnswer까지 그대로 전달된다.
    free dict 대신 타입을 부여해 source_name/citation_id 누락을 경계에서 차단한다.
    min_length=1로 빈 문자열까지 거부해 "누락 차단" 계약을 실제로 강제한다
    (정상 경로에서는 load_lookup_table()이 비어 있지 않음을 보장하므로, 위반은
    데이터 무결성 버그를 fail-loud로 드러낸다).
    """
    source_name: str = Field(min_length=1)
    citation_id: str = Field(min_length=1)


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
    agent_list: list[AgentName]   # 활성화할 에이전트 이름 목록


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
      - 검색 자체가 실패(DB 오류 등)하여 폴백됨
      → synthesize_step은 skipped 여부를 확인해서 "검색 결과 없음"으로 처리
    """
    query:      str
    articles:   list[str]   # 검색된 관련 조항 목록 (예: ["표준투자권유준칙 제5조"])
    summary:    str          # LLM이 생성한 한 문장 요약
    confidence: Confidence        # LLM의 자신감 점수 (0.0~1.0)
    evidence:   list[dict]  # 검증된 검색 근거 객체 목록
    skipped:    bool = False # agent_list에 없거나 검색 실패 시 True


class LawResultEvent(Event):
    """
    search_법규 step(Step 2b)의 출력 이벤트.
    법규 검색 결과를 담는다. RegulationResultEvent와 동일한 구조.
    """
    query:      str
    articles:   list[str]   # 예: ["자본시장법 제46조"]
    summary:    str
    confidence: Confidence
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
    confidence: Confidence
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
      예: [{"source_name": "표준투자권유준칙", "citation_id": "제5조"}, ...]

    재시도 카운터는 ctx.store("retry_count")에 보관된다 (factcheck_step 인라인 카운터).
    """
    query:           str
    verdict:         Verdict           # "가능" | "불가" | "조건부 가능" (LLM이 생성)
    reasoning:       str               # 판정 근거 설명
    cited_articles:  list[CitedArticle]  # 인용된 조항 목록 (synthesize가 evidence_id로 재구성)


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
    verdict:          Verdict            # 최종 판정
    reasoning:        str               # 최종 근거
    cited_articles:   list[CitedArticle]  # 검증 완료된 인용 조항
    factcheck_passed: bool         # 팩트체크 통과 여부 (신뢰도 지표)

    # 포트폴리오 설명용 메타 정보
    agents_used:      list[str]    # 실제 실행된 에이전트 목록 (예: ["규정", "법규"])


# =============================================================================
# LLM 구조화 출력(structured output) 스키마
# -----------------------------------------------------------------------------
# Event가 아니라 LLM 입출력 계약이다(Step 간 라우팅에 쓰이지 않으므로 Event 미상속).
# 위의 공유 타입(LLMVerdict)을 재사용하기 위해 같은 모듈에 둔다.
#
# astructured_predict()가 이 스키마를 Ollama format= 으로 전달하면 디코딩 자체가
# 스키마로 제약되고(잘못된 토큰 생성 불가), 응답은 Pydantic으로 검증된다.
# → verdict가 허용 집합 밖일 수 없어 코드 측 정규화가 불필요.
# =============================================================================

class SynthesisResponse(BaseModel):
    """
    synthesize_step의 LLM 출력 계약.

    verdict는 LLM이 고를 수 있는 3개(LLMVerdict)로 한정한다.
    "판정 불가"는 코드 폴백 전용이라 스키마에서 제외한다.
    """
    verdict:            LLMVerdict
    reasoning:          str
    cited_evidence_ids: list[str]


class ClassifyResponse(BaseModel):
    """
    classify_step의 LLM 출력 계약.

    agents는 허용 3종(AgentName)으로 제한된다. 스키마를 벗어나는 이름은 거부되며,
    호출부는 ValidationError 시 "전체 레인 활성화" fallback으로 안전하게 처리한다
    (verdict와 달리 거부해도 손실이 없다 — 최악의 경우 검색이 늘 뿐 누락은 없음).
    """
    agents:    list[AgentName]
    reasoning: str = ""


class FactcheckResponse(BaseModel):
    """
    factcheck_step의 LLM 출력 계약.

    failed_items: 인용 조항 중 실제로 존재하지 않는다고 LLM이 판단한 citation_id 목록.
    free-form 문자열이라 enum 제약은 없다. 누락 시 빈 목록으로 처리(코드의 결정론적
    검증 deterministic_failed가 별도로 적용되므로 안전).
    """
    failed_items: list[str] = []
