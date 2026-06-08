# =============================================================================
# langfuse_setup.py
# -----------------------------------------------------------------------------
# 역할: Langfuse 클라이언트 싱글턴 + 프롬프트 동기화를 담당한다.
#
# 기능 3가지:
#   1. get_client()        : Langfuse 클라이언트 싱글턴 반환 (env vars 자동 읽기)
#   2. sync_prompts()      : Langfuse에 프롬프트가 없으면 로컬 파일에서 업로드
#   3. get_langfuse_prompt(): Langfuse에서 프롬프트를 가져와 컴파일된 문자열 반환
#                             (Langfuse 미연결 시 로컬 파일 fallback)
#
# 환경변수 (자동 읽기):
#   LANGFUSE_PUBLIC_KEY  : Langfuse 공개 키
#   LANGFUSE_SECRET_KEY  : Langfuse 비밀 키
#   LANGFUSE_BASE_URL    : Langfuse 서버 주소 (기본 https://cloud.langfuse.com)
#
# Langfuse가 미설정이면:
#   - get_client()는 비활성화(disabled) 클라이언트를 반환한다.
#   - 모든 @observe 호출이 조용히 no-op으로 처리된다.
#   - 워크플로우 실행에는 영향 없음.
#
# 프롬프트 관리:
#   - 프로비저닝(신규 등록/버전 업)은 manage_prompts.py 또는 LANGFUSE_SYNC_PROMPTS=1 로 수행.
#   - 서빙 경로는 get_langfuse_prompt()가 사용 시점에 lazy하게 가져오고,
#     Langfuse 미연결 시 로컬 prompts/*.txt 로 fallback한다.
# =============================================================================

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 로컬 프롬프트 파일 디렉토리 (fallback 경로)
PROMPT_DIR = Path("prompts")

# Langfuse에 등록/조회할 프롬프트 이름 목록
PROMPT_NAMES = ["synthesize_agent", "classify_agent",
                "hyde_regulation", "hyde_law", "hyde_case"]

# 싱글턴 클라이언트 캐시
_client = None

# 프롬프트 동기화가 이 프로세스에서 이미 실행되었는지 여부 (프로세스당 1회 보장)
_prompts_synced = False


def get_client():
    """
    Langfuse 클라이언트 싱글턴을 반환한다.

    langfuse.get_client()는 환경변수(LANGFUSE_PUBLIC_KEY 등)를 자동으로 읽는다.
    env가 설정되지 않으면 비활성화 클라이언트가 반환되어 모든 호출이 no-op이 된다.
    → Langfuse 없이도 워크플로우가 정상 실행됨.
    """
    global _client
    if _client is None:
        from langfuse import get_client as _langfuse_get_client
        _client = _langfuse_get_client()
    return _client


def sync_prompts() -> None:
    """
    Langfuse 프롬프트 레지스트리를 로컬 파일과 동기화한다 (프로세스당 1회).

    각 프롬프트 이름에 대해:
      1. Langfuse에서 get_prompt()로 조회
      2. 없으면(404 등) 로컬 prompts/*.txt 파일 내용으로 create_prompt() 호출
      3. 이미 존재하면 아무 것도 하지 않음 (기존 버전 보존)

    이것은 배포/부트스트랩 성격의 작업이므로 요청 처리 경로가 아니라 프로세스
    시작 시 1회만 실행해야 한다 (CLI: main(), 서비스: FastAPI lifespan startup).
    모듈 전역 _prompts_synced 플래그로 멱등성(idempotency)을 보장하므로, 어디서
    몇 번 호출하든 실제 동기화 작업은 프로세스당 한 번만 일어난다.
    """
    global _prompts_synced
    if _prompts_synced:
        return

    client = get_client()
    for name in PROMPT_NAMES:
        try:
            client.get_prompt(name)
            logger.info(f"[langfuse] 프롬프트 이미 존재: {name}")
        except Exception:
            # Langfuse에 없음 → 로컬 파일에서 생성
            local_path = PROMPT_DIR / f"{name}.txt"
            if not local_path.exists():
                logger.warning(f"[langfuse] 로컬 프롬프트 파일 없음, 스킵: {local_path}")
                continue
            text = local_path.read_text(encoding="utf-8")
            try:
                client.create_prompt(
                    name=name,
                    prompt=text,
                    labels=["production"],
                    config={},
                )
                logger.info(f"[langfuse] 프롬프트 등록 완료: {name}")
            except Exception as e:
                logger.warning(f"[langfuse] 프롬프트 등록 실패: {name} ({e})")

    # 성공/실패와 무관하게 1회 시도 후 플래그를 세운다. Langfuse 미연결 시에도
    # get_langfuse_prompt()가 로컬 파일로 폴백하므로 매 요청마다 재시도할 필요가 없다.
    _prompts_synced = True


def get_langfuse_prompt(name: str) -> str:
    """
    Langfuse에서 프롬프트를 가져와 컴파일된 문자열로 반환한다.

    프롬프트에 template 변수가 없으므로 compile()은 텍스트 그대로를 반환한다.

    Fallback 동작:
      Langfuse 미연결 / 프롬프트 미존재 시 로컬 prompts/*.txt 파일을 읽는다.
      → 로컬 파일도 없으면 FileNotFoundError 발생.

    Args:
        name: 프롬프트 이름 (예: "synthesize_agent", "classify_agent")

    Returns:
        프롬프트 텍스트 문자열
    """
    try:
        client = get_client()
        prompt = client.get_prompt(name)
        return prompt.compile()
    except Exception as e:
        logger.warning(f"[langfuse] 프롬프트 가져오기 실패, 로컬 파일 fallback: {name} ({e})")
        local_path = PROMPT_DIR / f"{name}.txt"
        if not local_path.exists():
            raise FileNotFoundError(f"프롬프트 파일 없음: {local_path}") from e
        return local_path.read_text(encoding="utf-8")
