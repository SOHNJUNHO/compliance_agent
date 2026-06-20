# =============================================================================
# main.py  (LangGraph/LangChain 버전)
# -----------------------------------------------------------------------------
# 역할: LangGraph 기반 컴플라이언스 Q&A 워크플로우 실행 진입점.
#
# 실행 방법:
#   python main.py query "질문" [user_id]
#
# 데이터 적재는 먼저 실행한다:
#   python data/run_ingest_lc.py [pdf_path ...]
#
# LLM 설정:
#   Ollama를 로컬에서 실행한 후 사용한다.
#   실행 방법: ollama serve (별도 터미널에서)
#   모델 설치: ollama pull qwen3:8b-q4_K_M
#
# 관찰성 (Observability):
#   LlamaIndex 버전의 @observe + LlamaIndexInstrumentor를 대체한다.
#   langfuse.langchain.CallbackHandler를 graph.ainvoke의 config에 전달하면
#   모든 노드, LLM 호출, 리트리버 호출이 자동으로 Langfuse 트레이스에 기록된다.
# =============================================================================

import asyncio
import sys
import logging

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)

# =============================================================================
# LLM 설정 (교체 지점)
# =============================================================================

from langchain_ollama import ChatOllama

LLM_INSTANCE = ChatOllama(
    model="qwen3:8b-q4_K_M",
    temperature=0,
)


# =============================================================================
# query 모드 실행 함수
# =============================================================================

async def run_query(query: str, user_id: str = "anonymous") -> None:
    """LangGraph 그래프로 컴플라이언스 질문을 처리하고 결과를 출력한다."""
    from data.ingest_lc import load_store
    from workflow.tools_lc import tool_registry_lc
    from workflow.reranker_lc import build_reranker
    from workflow.graph import build_graph
    from workflow.langfuse_setup import get_client

    # 벡터 스토어 로드 + 레지스트리 초기화
    store = load_store()
    reranker = build_reranker()
    tool_registry_lc.build(store, reranker=reranker)

    # LangGraph 그래프 컴파일 (빌드 시 registry + llm 바인딩)
    graph = build_graph(LLM_INSTANCE, tool_registry_lc)

    # Langfuse CallbackHandler: graph.ainvoke config에 전달하면
    # 모든 노드·LLM 호출·리트리버 호출이 자동 트레이스된다.
    try:
        from langfuse.langchain import CallbackHandler
        lf_handler = CallbackHandler()
    except Exception:
        lf_handler = None  # Langfuse 미연결 시 무시

    config: dict = {
        "metadata": {
            "langfuse_user_id":    user_id,
            "langfuse_session_id": user_id,
            "langfuse_tags":       ["compliance-query"],
            "langfuse_trace_name": f"compliance: {query[:60]}",
        }
    }
    if lf_handler:
        config["callbacks"] = [lf_handler]

    initial_state = {
        "query":             query,
        "agent_list":        [],
        "routing_reasoning": "",
        "evidence":          [],
        "reasoning":         "",
        "cited_ids":         [],
        "cited_passages":    [],
        "agents_used":       [],
        "cited_agents":      [],
    }

    print(f"\n{'='*50}")
    print(f"질문: {query}")
    print(f"{'='*50}")

    final_state = await graph.ainvoke(initial_state, config=config)

    reasoning  = final_state.get("reasoning",         "")
    cited_ids  = final_state.get("cited_ids",         [])
    cited_pass = final_state.get("cited_passages",    [])
    agents     = final_state.get("agents_used",       [])
    c_agents   = final_state.get("cited_agents",      [])
    routing    = final_state.get("routing_reasoning", "")

    print(f"\n[답변]\n{reasoning}")
    print(f"[사용된 근거 ID] {cited_ids}")
    print(f"[실행 에이전트] {agents}")
    print(f"[인용 에이전트] {c_agents}")
    print(f"[활성화 근거] {routing}")

    if cited_pass:
        print("\n[인용 근거]")
        for p in cited_pass:
            print(f"{p['evidence_id']}\n  \"{p['text'][:300]}\"")

    # 프로세스 종료 전 Langfuse 이벤트 큐를 강제 전송
    if lf_handler:
        try:
            get_client().flush()
        except Exception:
            pass


# =============================================================================
# CLI 진입점
# =============================================================================

def main():
    """
    사용법:
      python main.py query "65세 고객에게 레버리지 ETF 권유 가능한가요?" [user_id]

    데이터 적재 (최초 1회):
      python data/run_ingest_lc.py [pdf_path ...]
    """
    if len(sys.argv) < 2 or sys.argv[1] != "query":
        print("사용법: python main.py query '질문' [user_id]")
        print("데이터 적재: python data/run_ingest_lc.py [pdf_path ...]")
        sys.exit(1)

    if len(sys.argv) < 3:
        print("오류: 질문을 입력하세요.")
        print("예: python main.py query '65세 고객에게 레버리지 ETF 권유 가능한가요?' cust-001")
        sys.exit(1)

    import os
    if os.getenv("LANGFUSE_SYNC_PROMPTS") == "1":
        from workflow.langfuse_setup import sync_prompts
        sync_prompts()

    query   = sys.argv[2]
    user_id = sys.argv[3] if len(sys.argv) > 3 else "anonymous"
    asyncio.run(run_query(query, user_id))


if __name__ == "__main__":
    main()
