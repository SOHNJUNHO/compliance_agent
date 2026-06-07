# =============================================================================
# evidence.py
# -----------------------------------------------------------------------------
# 역할: 검색 근거(evidence)의 추출·검증·포맷을 담당하는 순수 함수 모음.
#
# 설계 경계:
#   이 모듈은 워크플로우 오케스트레이션(compliance_workflow.py)에서 분리된
#   순수 로직 계층이다. LlamaIndex Workflow나 ctx에 의존하지 않으며,
#   외부 의존(조항 조회)은 article_lookup 콜러블을 인자로 주입받는다.
#   → 단위 테스트가 쉽고, 워크플로우 클래스를 비대하게 만들지 않는다.
#
# 제공 함수:
#   precision_filters          — 쿼리에서 고신뢰 메타데이터 필터 추출 (regex)
#   validate_article_evidence  — 사규/법규 검색 결과를 exact lookup으로 검증
#   validate_case_evidence     — 사례 검색 결과를 case_no 존재로 검증
#   evidence_article_labels    — evidence → "문서명 조항" 라벨 목록
#   format_synthesis_input     — 3개 레인 결과를 synthesize LLM 입력으로 포맷
#   format_evidence_for_synthesis — 단일 evidence 목록을 LLM 입력으로 포맷
#   format_factcheck_input     — factcheck LLM 입력(인용 조항 + 조회 결과) 포맷
# =============================================================================

import re
from typing import Callable


def precision_filters(query: str, agent: str) -> dict:
    """
    쿼리 텍스트에서 고신뢰 신호만 추출해 메타데이터 precision filter로 변환한다.
    source_type은 tools.py에서 이미 강제되므로 여기서는 추가 필터만 반환한다.

    포함하는 필터:
      source_name — 사용자가 쿼리에 문서명을 명시한 경우 (확실한 신호)
      citation_id — 쿼리에 "제N조" 또는 "N.N.N" 섹션번호가 있는 경우 (확실한 신호)

    여기서는 쿼리에 명시된 "확실한 신호"만 필터로 쓴다. 키워드 기반 주제 분류
    (과거의 category 메타데이터)는 오탐률이 높아 유효한 청크를 통째로 걸러낼 수
    있으므로 도입하지 않는다 — 주제 라우팅은 classify_step(LLM)이 담당한다.
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


def validate_article_evidence(
    raw_results: list[dict],
    article_lookup: Callable[..., dict | None],
) -> list[dict]:
    """
    사규/법규 검색 결과를 exact lookup으로 즉시 검증한다.

    article_lookup: (source_name, citation_id) → 조항 dict | None.
        존재하지 않는 조항은 결과에서 제외된다(hallucination 차단).
    """
    evidence = []
    for item in raw_results:
        source_name = item.get("source_name", "")
        citation_id = item.get("citation_id") or item.get("article_no", "")
        if not source_name or not citation_id:
            continue
        lookup = article_lookup(
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


def validate_case_evidence(raw_results: list[dict]) -> list[dict]:
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


def evidence_article_labels(evidence: list[dict]) -> list[str]:
    """evidence 목록을 "문서명 조항" 형식의 라벨 리스트로 변환한다."""
    return [
        f"{item.get('source_name', '')} {item.get('citation_id', '')}".strip()
        for item in evidence
        if item.get("source_name") and item.get("citation_id")
    ]


def format_synthesis_input(reg, law, case) -> str:
    """
    3개 에이전트 결과를 synthesize LLM에게 전달할 형태로 포맷한다.
    skipped된 에이전트는 "(결과 없음)"으로 표시.

    reg/law/case는 .skipped / .evidence 속성을 가진 이벤트 객체이면 된다
    (duck typing — 이벤트 타입을 import하지 않는다).
    """
    parts = []

    if not reg.skipped:
        parts.append("[사규 검증 근거]\n" + format_evidence_for_synthesis(reg.evidence))
    else:
        parts.append("[사규 검색 결과] (결과 없음)")

    if not law.skipped:
        parts.append("[법규 검증 근거]\n" + format_evidence_for_synthesis(law.evidence))
    else:
        parts.append("[법규 검색 결과] (결과 없음)")

    if not case.skipped:
        parts.append("[분쟁사례 검증 근거]\n" + format_evidence_for_synthesis(case.evidence))
    else:
        parts.append("[분쟁사례 검색 결과] (결과 없음)")

    return "\n\n".join(parts)


def format_evidence_for_synthesis(evidence: list[dict]) -> str:
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


def format_factcheck_input(verdict: str, reasoning: str, lookups: list[dict]) -> str:
    """
    factcheck LLM에게 전달할 컨텍스트를 포맷한다.
    인용 조항 목록과 조회 결과(존재/미존재)를 포함한다.

    lookups의 각 항목: {"cited": CitedArticle, "found": ..., "exists": bool}
    """
    lines = [
        f"판정 초안: {verdict}",
        f"근거: {reasoning}",
        "",
        "인용 조항 검증 결과:",
    ]
    for item in lookups:
        status = "✓ 존재" if item["exists"] else "✗ 미존재"
        cited = item["cited"]
        eid = f"{cited.source_name}||{cited.citation_id}"
        lines.append(
            f"  - eid={eid} : {status}"
        )
    return "\n".join(lines)
