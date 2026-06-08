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
#     ↓ synthesize_step (3개 모두 도착 후 실행 → 워크플로우 종료)
#   StopEvent(result=FinalAnswer)
# =============================================================================

from typing import Literal

from llama_index.core.workflow import Event  # LlamaIndex Event 기반 클래스
from pydantic import BaseModel  # Event가 Pydantic BaseModel이므로 제약 조건을 그대로 사용 가능

AgentName = Literal["규정", "법규", "사례"]


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
    evidence:   list[dict]  # 검증된 검색 근거 객체 목록
    skipped:    bool = False # agent_list에 없거나 검색 실패 시 True


class LawResultEvent(Event):
    """
    search_법규 step(Step 2b)의 출력 이벤트.
    법규 검색 결과를 담는다. RegulationResultEvent와 동일한 구조.
    """
    query:      str
    evidence:   list[dict]
    skipped:    bool = False


class CaseResultEvent(Event):
    """
    search_사례 step(Step 2c)의 출력 이벤트.
    분쟁사례 검색 결과를 담는다. RegulationResultEvent와 동일한 구조.
    """
    query:      str
    evidence:   list[dict]
    skipped:    bool = False


# =============================================================================
# synthesize_step 출력 → StopEvent의 result
# =============================================================================

class FinalAnswer(Event):
    """
    워크플로우의 최종 결과.
    StopEvent(result=FinalAnswer(...))로 전달된다.

    main.py에서 이 객체의 필드를 출력한다.
    포트폴리오 데모 시 이 구조가 "에이전트가 구조화된 답변을 생성한다"는 것을 보여준다.
    """
    reasoning:        str               # 최종 답변 (evidence_id 라벨 + 답변 문장 블록)
    cited_ids:        list[str]     # 답변에 인용된 근거 ID (관련 판정 통과분)
    cited_passages:   list[dict]    # 인용된 근거의 evidence_id + text ([인용 근거] 출력용)

    # 포트폴리오 설명용 메타 정보
    agents_used:       list[str]   # 실제 실행된 에이전트 목록 (예: ["규정", "법규"])
    routing_reasoning: str = ""    # classify_step이 그 에이전트들을 선택한 이유


# =============================================================================
# LLM 구조화 출력(structured output) 스키마
# -----------------------------------------------------------------------------
# Event가 아니라 LLM 입출력 계약이다(Step 간 라우팅에 쓰이지 않으므로 Event 미상속).
# Event 클래스와 같은 계약 계층이므로 같은 모듈에 둔다.
#
# astructured_predict()가 이 스키마를 Ollama format= 으로 전달하면 디코딩 자체가
# 스키마로 제약되고(잘못된 토큰 생성 불가), 응답은 Pydantic으로 검증된다.
# =============================================================================

class PassageAnswer(BaseModel):
    """
    synthesize_step의 LLM 출력 계약 (단일 근거 → 단일 판단).

    LLM은 하나의 근거만 보고 관련성과 해석을 반환한다.
    출처 명명은 코드가 담당하므로 LLM은 evidence_id를 생성하지 않는다
    → citation hallucination이 스키마 수준에서 불가능해진다.
    """
    relevant: bool  # 이 근거가 질문에 답하는 데 관련 있는가
    answer:   str   # 이 근거에 한정한 1~2문장 해석 (relevant=False면 "")


# [rollback: 단일-호출 합성 방식으로 복구 시 아래 주석 해제]
# class SynthesisResponse(BaseModel):
#     """
#     synthesize_step의 LLM 출력 계약 (전체 근거 → 단일 답변).
#
#     reasoning은 인용 근거에 기반한 답변이고, cited_evidence_ids는 그 근거의
#     evidence_id 목록이다. 코드가 cited_evidence_ids를 all_evidence와 대조해
#     실재하는 ID만 cited_ids로 남긴다(hallucination 차단).
#     문제: LLM이 evidence_id를 자유 생성하므로 존재 검증은 되지만 attribution은 보장 안 됨.
#     """
#     reasoning:          str
#     cited_evidence_ids: list[str]


class ClassifyResponse(BaseModel):
    """
    classify_step의 LLM 출력 계약.

    agents는 허용 3종(AgentName)으로 제한된다. 스키마를 벗어나는 이름은 거부되며,
    호출부는 ValidationError 시 "전체 레인 활성화" fallback으로 안전하게 처리한다
    (거부해도 손실이 없다 — 최악의 경우 검색이 늘 뿐 누락은 없음).
    """
    agents:    list[AgentName]
    reasoning: str = ""


