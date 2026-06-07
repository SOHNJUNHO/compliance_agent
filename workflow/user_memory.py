# =============================================================================
# user_memory.py
# -----------------------------------------------------------------------------
# 역할: user_id 기준으로 "사용자 메모리"를 SQLite에 영속화하는 참조 모듈.
#
# 두 가지 메모리를 정의한다:
#   1. profile      - 고객 프로필 (나이, 투자성향 등). 적합성/적정성 판정의 입력.
#   2. interaction  - 과거 질의·판정 이력. 감사 추적 + 맥락 제공.
#
# PoC 범위 참고:
#   현재 main.py는 이 모듈을 import하지 않는다 (컨텍스트 주입은 PoC 범위 외).
#   설계 의도는 다음과 같다: 이 모듈은 "애플리케이션 계층"에 속하며, 워크플로우는
#   SQLite를 직접 알지 못한다. main.py가 메모리를 읽어 synthesize_step에 주입하고
#   결과를 다시 이 모듈로 저장하는 방식으로 연결한다.
#   → 워크플로우가 순수 함수를 유지하면서 사용자 맥락을 활용할 수 있다.
#
# ctx.store는 단일 run 동안만 유효한 휘발성 저장소이므로, run을 넘는 "기억"은
# 반드시 이런 외부 영속 저장소가 필요하다.
# =============================================================================

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# DB 경로 (USER_MEMORY_DB 환경변수로 재정의 가능)
DB_PATH = Path(os.getenv("USER_MEMORY_DB", "data/user_memory.db"))

# upsert_profile이 수정을 허용하는 컬럼 화이트리스트
# (임의 키 주입을 막아 SQL 조립을 안전하게 유지)
_PROFILE_COLUMNS = ("age", "investment_grade", "risk_appetite", "notes")


def _now() -> str:
    """ISO 8601 UTC 타임스탬프."""
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    """
    연결을 열고 스키마를 보장한다 (없으면 생성).
    row_factory를 dict형으로 설정해 컬럼명으로 접근 가능하게 한다.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profile (
            user_id          TEXT PRIMARY KEY,
            age              INTEGER,
            investment_grade TEXT,
            risk_appetite    TEXT,
            notes            TEXT,
            updated_at       TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS interaction (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id          TEXT NOT NULL,
            created_at       TEXT NOT NULL,
            query            TEXT NOT NULL,
            verdict          TEXT,
            reasoning        TEXT,
            factcheck_passed INTEGER
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_interaction_user "
        "ON interaction(user_id, created_at DESC)"
    )
    conn.commit()
    return conn

# =============================================================================
# 프로필 (customer profile)
# =============================================================================

def upsert_profile(user_id: str, **fields) -> None:
    """
    프로필을 생성/갱신한다. _PROFILE_COLUMNS에 있는 키만 반영된다.

    예: upsert_profile("cust-001", age=65, investment_grade="일반", risk_appetite="낮음")
    """
    cols = {k: v for k, v in fields.items() if k in _PROFILE_COLUMNS}
    if not cols:
        raise ValueError(f"갱신할 유효한 컬럼이 없습니다. 허용: {_PROFILE_COLUMNS}")

    with _connect() as conn:
        # 행이 없으면 INSERT, 있으면 지정된 컬럼만 UPDATE (나머지는 보존)
        conn.execute(
            "INSERT INTO profile (user_id, updated_at) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO NOTHING",
            (user_id, _now()),
        )
        assignments = ", ".join(f"{c} = ?" for c in cols)
        params = list(cols.values()) + [_now(), user_id]
        conn.execute(
            f"UPDATE profile SET {assignments}, updated_at = ? WHERE user_id = ?",
            params,
        )


def get_profile(user_id: str) -> dict | None:
    """프로필 dict 반환 (없으면 None)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM profile WHERE user_id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


# =============================================================================
# 상호작용 이력 (interaction history)
# =============================================================================

def add_interaction(
    user_id: str,
    query: str,
    verdict: str | None = None,
    reasoning: str | None = None,
    factcheck_passed: bool | None = None,
) -> None:
    """한 건의 질의·판정 결과를 이력에 추가한다."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO interaction "
            "(user_id, created_at, query, verdict, reasoning, factcheck_passed) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                user_id, _now(), query, verdict, reasoning,
                None if factcheck_passed is None else int(factcheck_passed),
            ),
        )


def get_recent_interactions(user_id: str, limit: int = 5) -> list[dict]:
    """최근 상호작용을 최신순으로 반환한다."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM interaction WHERE user_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def save_interaction(user_id: str, final_answer) -> None:
    """
    워크플로우 결과(FinalAnswer)를 이력으로 저장하는 편의 함수.
    main.py가 run() 종료 후 호출한다.
    """
    add_interaction(
        user_id=user_id,
        query=getattr(final_answer, "query", ""),
        verdict=getattr(final_answer, "verdict", None),
        reasoning=getattr(final_answer, "reasoning", None),
        factcheck_passed=getattr(final_answer, "factcheck_passed", None),
    )


# =============================================================================
# 조회 + 프롬프트 주입용 포맷
# =============================================================================

def load_user_memory(user_id: str, history_limit: int = 5) -> dict:
    """프로필 + 최근 이력을 한 번에 조회한다."""
    return {
        "profile": get_profile(user_id),
        "history": get_recent_interactions(user_id, history_limit),
    }


def format_memory_for_prompt(memory: dict) -> str:
    """
    load_user_memory() 결과를 LLM 프롬프트에 주입할 한국어 텍스트로 렌더링한다.
    메모리가 비어 있으면 빈 문자열을 반환한다(→ 호출부가 주입을 생략).

    이 함수는 순수 함수(DB 접근 없음)이므로 워크플로우에서 import해도 안전하다.
    """
    profile = memory.get("profile")
    history = memory.get("history") or []
    if not profile and not history:
        return ""

    lines: list[str] = []

    if profile:
        parts = []
        if profile.get("age") is not None:
            parts.append(f"나이 {profile['age']}세")
        if profile.get("investment_grade"):
            parts.append(f"투자자등급 {profile['investment_grade']}")
        if profile.get("risk_appetite"):
            parts.append(f"위험성향 {profile['risk_appetite']}")
        if profile.get("notes"):
            parts.append(f"비고: {profile['notes']}")
        if parts:
            lines.append("■ 고객 프로필: " + ", ".join(parts))

    if history:
        lines.append("■ 최근 상담 이력:")
        for h in history:
            lines.append(
                f"  - [{h.get('created_at', '')[:10]}] \"{h.get('query', '')}\" "
                f"→ {h.get('verdict', '?')}"
            )

    return "\n".join(lines)
