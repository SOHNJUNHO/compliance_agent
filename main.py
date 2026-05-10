# =============================================================================
# main.py
# -----------------------------------------------------------------------------
# 역할: 프로젝트의 진입점. 두 가지 실행 모드를 제공한다.
#
# 모드 1: ingest — 데이터 수집/파싱/적재 파이프라인 실행
#   python main.py ingest [pdf_path ...]
#   → scraper → parser → ingest (VectorStoreIndex + article_lookup.json 생성)
#
# 모드 2: query — 컴플라이언스 질문에 대한 워크플로우 실행
#   python main.py query "질문"
#   → ComplianceWorkflow.run() → FinalAnswer 출력
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

from langfuse import observe, get_client
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
LlamaIndexInstrumentor().instrument()

# =============================================================================
# LLM 설정 (교체 지점)
# =============================================================================
# 옵션 설명:
#   model:           Ollama에 설치된 모델명 (ollama pull qwen2.5:7b 필요)
#   request_timeout: 단일 요청 최대 대기 시간 (초)
#   json_mode:       True → Ollama가 항상 유효한 JSON을 반환하도록 강제
from llama_index.llms.ollama import Ollama

LLM_INSTANCE = Ollama(
    model="qwen3:8b-q4_K_M",
    request_timeout=120.0,
    json_mode=True,
)


# =============================================================================
# query 모드 실행 함수
# =============================================================================

@observe(name="compliance_query", as_type="span")
async def run_query(query: str) -> None:
    """컴플라이언스 질문을 워크플로우에 전달하고 결과를 출력한다."""
    sys.path.insert(0, ".")
    sys.path.insert(0, "data")
    sys.path.insert(0, "workflow")

    from data.ingest import ingest
    from data.scraper import scrape_all
    from data.parser import parse_all
    from workflow.tools import tool_registry
    from workflow.compliance_workflow import ComplianceWorkflow
    from workflow.reranker import build_reranker
    from langfuse_setup import sync_prompts

    # Langfuse 루트 트레이스 입력 기록
    get_client().update_current_span(input={"query": query})

    # Langfuse 프롬프트 동기화 (없으면 로컬 파일에서 업로드)
    sync_prompts()

    print("데이터 준비 중... (처음 실행 시 시간이 걸립니다)")

    raw_docs = scrape_all()
    chunks = parse_all(raw_docs)
    index = ingest(chunks)

    # 재순위기 초기화 (USE_RERANKER=0 이면 None → 기존 동작 유지)
    reranker = build_reranker()
    tool_registry.build(index, reranker=reranker)

    wf = ComplianceWorkflow(
        llm=LLM_INSTANCE,
        registry=tool_registry,
        timeout=120,
        verbose=True,
    )

    print(f"\n{'='*50}")
    print(f"질문: {query}")
    print(f"{'='*50}")

    handler = wf.run(query=query)
    result = await handler

    if hasattr(result, "verdict"):
        print(f"\n[판정] {result.verdict}")
        print(f"[근거] {result.reasoning}")
        print(f"[인용 조항] {result.cited_articles}")
        print(f"[위험도] {result.risk_level}/3")
        print(f"[팩트체크] {'통과 ✓' if result.factcheck_passed else '실패 ✗'}")
        print(f"[실행 에이전트] {result.agents_used}")
        print(f"[총 토큰 사용량] {result.token_used}")

        # Langfuse 루트 트레이스 출력 기록
        get_client().update_current_span(
            output={
                "verdict": result.verdict,
                "factcheck_passed": result.factcheck_passed,
                "risk_level": result.risk_level,
            },
            metadata={
                "agents_used": result.agents_used,
                "token_used": result.token_used,
            },
        )
    else:
        print(f"결과: {result}")

    # 프로세스 종료 전 Langfuse 이벤트 큐를 강제 전송
    # 스크립트/CLI 환경에서는 백그라운드 스레드가 종료 전에 flush를 보장하지 않으므로 필수
    get_client().flush()


# =============================================================================
# CLI 진입점
# =============================================================================

def main():
    """
    사용법:
      python main.py ingest               # PDF 없이 웹 데이터만 적재
      python main.py ingest ./fss.pdf     # PDF 포함 적재
      python main.py query "65세 고객에게 레버리지 ETF 권유 가능한가요?"
    """
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python main.py ingest [pdf_path ...]")
        print("  python main.py query '질문'")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "ingest":
        sys.path.insert(0, ".")
        sys.path.insert(0, "data")
        sys.path.insert(0, "workflow")
        from data.scraper import scrape_all
        from data.parser import parse_all
        from data.ingest import ingest

        pdf_paths = sys.argv[2:]
        raw = scrape_all(pdf_paths=pdf_paths)
        chunks = parse_all(raw)
        ingest(chunks)
        print(f"적재 완료: {len(chunks)}개 청크")

    elif mode == "query":
        if len(sys.argv) < 3:
            print("오류: 질문을 입력하세요.")
            print("예: python main.py query '65세 고객에게 레버리지 ETF 권유 가능한가요?'")
            sys.exit(1)

        query = sys.argv[2]
        asyncio.run(run_query(query))

    else:
        print(f"오류: 알 수 없는 모드 '{mode}' (ingest 또는 query)")
        sys.exit(1)


if __name__ == "__main__":
    main()
