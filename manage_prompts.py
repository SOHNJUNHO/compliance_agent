# =============================================================================
# manage_prompts.py
# -----------------------------------------------------------------------------
# 로컬 prompts/*.txt 파일을 Langfuse에 새 버전으로 강제 푸시하는 개발자 도구.
# sync_prompts()는 부팅 시 create-only이므로, 프롬프트 수정 후 이 명령으로 올린다.
#
# 사용법: python manage_prompts.py     # 모든 프롬프트를 새 버전으로 푸시
# =============================================================================

from dotenv import load_dotenv
load_dotenv()

from workflow.langfuse_setup import get_client, PROMPT_DIR, PROMPT_NAMES


def push() -> None:
    """로컬 prompts/*.txt 를 Langfuse에 새 버전으로 푸시한다 (label: production)."""
    client = get_client()
    for name in PROMPT_NAMES:
        local_path = PROMPT_DIR / f"{name}.txt"
        if not local_path.exists():
            print(f"[skip] 로컬 파일 없음: {local_path}")
            continue
        text = local_path.read_text(encoding="utf-8")
        result = client.create_prompt(
            name=name,
            prompt=text,
            labels=["production"],
            config={},
        )
        print(f"[push] {name} → version {result.version}")
    client.flush()


if __name__ == "__main__":
    push()
