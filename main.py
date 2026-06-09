# =============================================================================
# main.py
# -----------------------------------------------------------------------------
# 역할: 컴플라이언스 Q&A 워크플로우 실행 진입점 (query 전용).
#
# 실행 방법:
#   python main.py query "질문"
#   → ComplianceWorkflow.run() → FinalAnswer 출력
#
# 데이터 적재는 run_ingest.py 를 먼저 실행한다:
#   python run_ingest.py [pdf_path ...]
#
# LLM 설정:
#   Ollama를 로컬에서 실행한 후 사용한다.
#   실행 방법: ollama serve (별도 터미널에서)
#   모델 설치: ollama pull qwen3:8b-q4_K_M
# =============================================================================

import asyncio
import sys
import logging

from dotenv import load_dotenv
load_dotenv()  # .env 파일에서 환경변수 로드 (없으면 무시)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s"
)

from langfuse import observe, get_client, propagate_attributes
# LlamaIndex auto-instrumentation is active: raw LLM/embedding spans are recorded in
# Langfuse in addition to the 6 manually-decorated business spans (compliance_query
# root + classify + search×3 + synthesize). Remove the two lines below to disable.
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
LlamaIndexInstrumentor().instrument()

# =============================================================================
# LLM 설정 (교체 지점)
# =============================================================================
# 옵션 설명:
#   model:           Ollama에 설치된 모델명 (ollama pull qwen3:8b-q4_K_M 필요)
#   request_timeout: 단일 요청 최대 대기 시간 (초)
#   json_mode:       True → Ollama가 항상 유효한 JSON을 반환하도록 강제
from llama_index.llms.ollama import Ollama

LLM_INSTANCE = Ollama(
    model="qwen3:8b-q4_K_M",
    request_timeout=120.0,
    json_mode=True,    # 출력을 유효한 JSON으로 강제
    thinking=False,    # qwen3 추론(<think>) 출력 비활성화 → 응답이 순수 JSON이 됨
)


# =============================================================================
# query 모드 실행 함수
# =============================================================================

@observe(name="compliance_query", as_type="span", capture_input=False, capture_output=False)
async def run_query(query: str, user_id: str = "anonymous") -> None:
    """컴플라이언스 질문을 워크플로우에 전달하고 결과를 출력한다."""
    # sys.path.insert(0, ".")
    # sys.path.insert(0, "data")
    # sys.path.insert(0, "workflow")

    from data.ingest import load_index
    from workflow.tools import tool_registry
    from workflow.compliance_workflow import ComplianceWorkflow
    from workflow.reranker import build_reranker

    client = get_client()
    # Langfuse 루트 트레이스 입력 기록
    client.update_current_span(input={"query": query})

    # user_id를 이 trace의 모든 하위 @observe 스팬에 전파한다.
    # OTEL baggage 기반이라 async 워크플로우 경계를 넘어 유지된다.
    # → Langfuse UI의 Users 필터가 trace 전체에 일관되게 적용된다.
    with propagate_attributes(
        user_id=user_id,
        session_id=user_id,                      # 한 사용자의 질의들을 한 세션으로 묶음
        trace_name=f"compliance: {query[:60]}",  # trace 목록에 질문이 바로 보이도록
        tags=["compliance-query"],
    ):
        # 프롬프트 동기화는 main()에서 프로세스 시작 시 1회 수행한다 (요청 경로 아님).

        # load_index()가 Qdrant 컬렉션 부재/공백 시 명확한 오류를 발생시킨다.
        # article_lookup 테이블도 동일 컬렉션에서 구성되므로 별도 파일 검사는 불필요.
        index = load_index()

        # 재순위기 초기화 (USE_RERANKER=0 이면 None → 기존 동작 유지)
        reranker = build_reranker()
        tool_registry.build(index, reranker=reranker)

        wf = ComplianceWorkflow(
            llm=LLM_INSTANCE,
            registry=tool_registry,
            timeout=300,
            verbose=True,
        )

        print(f"\n{'='*50}")
        print(f"질문: {query}")
        print(f"{'='*50}")

        handler = wf.run(query=query)
        result = await handler

        if hasattr(result, "reasoning"):
            print(f"\n[답변]\n{result.reasoning}")
            print(f"[사용된 근거 ID] {result.cited_ids}")
            print(f"[실행 에이전트] {result.agents_used}")
            print(f"[인용 에이전트] {result.cited_agents}")
            print(f"[활성화 근거] {result.routing_reasoning}")

            if result.cited_passages:
                print("\n[인용 근거]")
                for p in result.cited_passages:
                    print(f"{p['evidence_id']}\n  \"{p['text'][:300]}\"")

            # Langfuse 루트 트레이스 출력 기록
            client.update_current_span(
                output={
                    "answer": result.reasoning,
                    "cited_ids": result.cited_ids,
                },
                metadata={
                    "agents_used": str(result.agents_used),
                },
            )
        else:
            print(f"결과: {result}")

    # 프로세스 종료 전 Langfuse 이벤트 큐를 강제 전송
    # 스크립트/CLI 환경에서는 백그라운드 스레드가 종료 전에 flush를 보장하지 않으므로 필수
    client.flush()


# =============================================================================
# CLI 진입점
# =============================================================================

def main():
    """
    사용법:
      python main.py query "65세 고객에게 레버리지 ETF 권유 가능한가요?" [user_id]

    user_id (선택, 기본 'anonymous'):
      Langfuse trace에 부여되어 Users 필터/그룹화에 사용된다.

    데이터 적재 (최초 1회):
      python run_ingest.py [pdf_path ...]
    """
    if len(sys.argv) < 2 or sys.argv[1] != "query":
        print("사용법: python main.py query '질문' [user_id]")
        print("데이터 적재: python run_ingest.py [pdf_path ...]")
        sys.exit(1)

    if len(sys.argv) < 3:
        print("오류: 질문을 입력하세요.")
        print("예: python main.py query '65세 고객에게 레버리지 ETF 권유 가능한가요?' cust-001")
        sys.exit(1)

    # 프롬프트 프로비저닝은 배포/개발 단계의 책임이다 (manage_prompts.py).
    # 서빙 경로는 기본적으로 동기화하지 않는다 — load_prompt()가 사용 시점에 Langfuse에서
    # lazy하게 가져오고, 실패 시 로컬 prompts/*.txt 로 fallback한다.
    # 최초 부트스트랩이 필요하면: LANGFUSE_SYNC_PROMPTS=1 python main.py query ...
    import os
    if os.getenv("LANGFUSE_SYNC_PROMPTS") == "1":
        from workflow.langfuse_setup import sync_prompts
        sync_prompts()

    query = sys.argv[2]
    user_id = sys.argv[3] if len(sys.argv) > 3 else "anonymous"
    asyncio.run(run_query(query, user_id))


if __name__ == "__main__":
    main()
