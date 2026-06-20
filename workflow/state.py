# =============================================================================
# state.py
# -----------------------------------------------------------------------------
# 역할: LangGraph 워크플로우의 단일 상태 스키마 및 LLM 구조화 출력 계약을 정의한다.
#
# LlamaIndex 이벤트 방식과의 차이:
#   LlamaIndex Workflow: Event 서브클래스로 Step 간 라우팅 (이벤트 타입 = 라우팅 규칙)
#   LangGraph: 단일 TypedDict(ComplianceState)가 노드 간을 흐른다.
#              노드는 상태의 부분 딕셔너리를 반환 → reducer가 병합한다.
#
# 핵심 설계:
#   - evidence 필드에만 operator.add reducer를 적용한다.
#     → 병렬 search 노드들이 각자의 결과를 누적한다 (fan-in).
#     → collect_events() + skipped 플래그 패턴이 필요 없어진다.
#   - 나머지 필드는 last-write-wins (synthesize 노드만 쓴다).
#   - LaneWork는 Send API로 search 노드에 전달되는 per-task 슬라이스다.
#
# LLM 계약 (PassageAnswer, ClassifyResponse):
#   events.py에서 이동했다. 워크플로우 오케스트레이션에 의존하지 않는 순수 Pydantic 모델.
# =============================================================================

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel

# =============================================================================
# 에이전트 이름 타입
# =============================================================================

AgentName = Literal["규정", "법규", "사례"]


# =============================================================================
# 워크플로우 전역 상태
# =============================================================================

class ComplianceState(TypedDict):
    """
    LangGraph 워크플로우의 단일 상태 클래스.

    LlamaIndex의 5개 Event 클래스(ClassifiedEvent / *ResultEvent×3 / FinalAnswer)를
    하나의 TypedDict로 통합한다.

    필드 설명:
      query              : 사용자의 원본 질문 (StartEvent 역할)
      agent_list         : classify 노드가 결정한 활성 레인 목록
      routing_reasoning  : classify 노드의 LLM 판단 근거
      evidence           : 병렬 search 노드들의 누적 결과 (reducer: operator.add)
                           규정→법규→사례 순 정렬은 synthesize 노드에서 수행
      reasoning          : synthesize 노드의 최종 답변
      cited_ids          : 답변에 인용된 evidence_id 목록
      cited_passages     : [인용 근거] 출력용 (evidence_id + text)
      agents_used        : 실제 실행된 에이전트 목록
      cited_agents       : 근거가 인용된 에이전트 (agents_used의 부분집합)
    """
    query:             str
    agent_list:        list[AgentName]
    routing_reasoning: str

    # operator.add: 병렬 search 노드들이 각자 반환하는 list를 이어 붙인다.
    # LlamaIndex의 collect_events() + skipped=True 패턴을 대체한다.
    evidence:          Annotated[list[dict], operator.add]

    # synthesize 노드 출력 (last-write-wins)
    reasoning:         str
    cited_ids:         list[str]
    cited_passages:    list[dict]
    agents_used:       list[str]
    cited_agents:      list[str]


# =============================================================================
# Send API 페이로드 (search 노드 per-task 슬라이스)
# =============================================================================

class LaneWork(TypedDict):
    """
    route_to_lanes가 Send("search", LaneWork)로 search 노드에 전달하는 슬라이스.

    LangGraph Send API:
      graph.add_conditional_edges("classify", route_to_lanes)
      route_to_lanes가 [Send("search", {...}), ...] 을 반환하면
      LangGraph가 각 Send를 독립적인 search 태스크로 실행한다.
    """
    query: str
    lane:  str   # "규정" | "법규" | "사례"


# =============================================================================
# LLM 구조화 출력 계약 (events.py에서 이동, 동일 내용)
# =============================================================================

class PassageAnswer(BaseModel):
    """
    synthesize 노드의 LLM 출력 계약 (단일 근거 → 단일 판단).

    LLM은 하나의 근거만 보고 관련성과 해석을 반환한다.
    출처 명명은 코드가 담당하므로 LLM은 evidence_id를 생성하지 않는다
    → citation hallucination이 스키마 수준에서 불가능해진다.
    """
    relevant: bool  # 이 근거가 질문에 답하는 데 관련 있는가
    answer:   str   # 이 근거에 한정한 1~2문장 해석 (relevant=False면 "")


class ClassifyResponse(BaseModel):
    """
    classify 노드의 LLM 출력 계약.

    agents는 허용 3종(AgentName)으로 제한된다. 스키마를 벗어나는 이름은 거부되며,
    호출부는 ValidationError 시 "전체 레인 활성화" fallback으로 안전하게 처리한다
    (거부해도 손실이 없다 — 최악의 경우 검색이 늘 뿐 누락은 없음).
    """
    agents:    list[AgentName]
    reasoning: str = ""
