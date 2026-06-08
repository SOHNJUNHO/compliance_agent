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
#   validate_article_evidence  — 사규/법규 검색 결과를 exact lookup으로 검증
#   validate_case_evidence     — 사례 검색 결과를 case_no 존재로 검증
#   evidence_article_labels    — evidence → "문서명 조항" 라벨 목록
#   collect_evidence           — 3개 레인 결과를 규정→법규→사례 순으로 단일 리스트 합산
#   format_single_passage      — 단일 근거를 per-passage LLM 입력으로 포맷
#   format_cited_block         — per-passage 답변 fragment를 최종 출력 블록으로 포맷
# =============================================================================

from typing import Callable


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


def collect_evidence(reg, law, case) -> list[dict]:
    """
    3개 레인의 검증된 근거를 규정→법규→사례 순서로 하나의 리스트로 합친다.

    reg/law/case는 .skipped / .evidence 속성을 가진 이벤트 객체이면 된다
    (duck typing — 이벤트 타입을 import하지 않는다).

    synthesize_step에서 all_evidence dict 대신 이 리스트를 기반으로
    retrieved_ids를 구성하고 per-passage map을 실행한다.
    """
    result = []
    for ev in (reg, law, case):
        if not ev.skipped:
            result.extend(ev.evidence)
    return result


def format_single_passage(item: dict) -> str:
    """
    단일 근거를 LLM 입력용으로 포맷한다 (per-passage map 전용).

    LLM에게 출처를 명명하도록 요청하지 않는다 — citation 부착은 코드가 담당한다.
    """
    citation = item.get("citation_id") or item.get("article_no") or item.get("case_no", "")
    source_type = item.get("source_type", "")
    return (
        f"[{source_type}] {item.get('source_name', '')} {citation}\n"
        f"{item.get('text', '')[:900]}"
    )


def format_cited_block(item: dict, answer: str) -> str:
    """
    per-passage 답변 fragment를 최종 출력용 블록으로 포맷한다.

    라벨은 evidence_id를 그대로 사용한다 → [답변]에서 인용 ID가 답변 문장과 함께
    노출되고, [인용 근거] 섹션의 동일 evidence_id와 짝지어진다 (id가 의도적으로 2회 등장).

    레이아웃 예시:
      자본시장과금융투자업에관한법률||제46조
        → 적합성 원칙에 따라 투자자 성향을 먼저 확인해야 합니다.
    """
    eid = item.get("evidence_id", "")
    return f"{eid}\n  → {answer}"


# [rollback: 단일-호출 합성 방식으로 복구 시 아래 주석 해제]
#
# def format_synthesis_input(reg, law, case) -> str:
#     """
#     3개 에이전트 결과를 synthesize LLM에게 전달할 형태로 포맷한다.
#     skipped된 에이전트는 "(결과 없음)"으로 표시.
#     """
#     parts = []
#     if not reg.skipped:
#         parts.append("[사규 검증 근거]\n" + format_evidence_for_synthesis(reg.evidence))
#     else:
#         parts.append("[사규 검색 결과] (결과 없음)")
#     if not law.skipped:
#         parts.append("[법규 검증 근거]\n" + format_evidence_for_synthesis(law.evidence))
#     else:
#         parts.append("[법규 검색 결과] (결과 없음)")
#     if not case.skipped:
#         parts.append("[분쟁사례 검증 근거]\n" + format_evidence_for_synthesis(case.evidence))
#     else:
#         parts.append("[분쟁사례 검색 결과] (결과 없음)")
#     return "\n\n".join(parts)
#
#
# def format_evidence_for_synthesis(evidence: list[dict]) -> str:
#     """검증된 evidence 객체를 최종 합성 LLM에 전달할 형식으로 변환한다."""
#     if not evidence:
#         return "(검증된 근거 없음)"
#     lines = []
#     for i, item in enumerate(evidence, 1):
#         citation = item.get("citation_id") or item.get("article_no") or item.get("case_no", "")
#         lines.append(
#             f"[E{i}] evidence_id={item.get('evidence_id', '')}\n"
#             f"source_type={item.get('source_type', '')}\n"
#             f"source_name={item.get('source_name', '')}\n"
#             f"citation={citation}\n"
#             f"article_no={item.get('article_no', '')}\n"
#             f"section_no={item.get('section_no', '')}\n"
#             f"case_no={item.get('case_no', '')}\n"
#             f"score={item.get('score', 0):.4f}\n"
#             f"text={item.get('text', '')[:900]}"
#         )
#     return "\n\n".join(lines)


