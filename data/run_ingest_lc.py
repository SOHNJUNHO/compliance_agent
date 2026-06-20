# =============================================================================
# run_ingest_lc.py
# -----------------------------------------------------------------------------
# 역할: LangChain/LangGraph 버전의 데이터 수집/파싱/적재 파이프라인 진입점.
#       run_ingest.py(LlamaIndex)의 LangChain 대응 구현.
#
# 실행 방법:
#   python data/run_ingest_lc.py               # 웹 데이터만 수집
#   python data/run_ingest_lc.py ./fss.pdf     # PDF 추가 적재
#
# 데이터 흐름:
#   scraper.py → parser.py → ingest_lc.py
#   → QdrantVectorStore (compliance_agent_lc 컬렉션)
#
# 주의:
#   기존 LlamaIndex 컬렉션(compliance_agent)과 별도 컬렉션을 사용한다.
#   scraper.py / parser.py 는 변경 없이 재사용된다.
#
# 사전 조건:
#   ollama serve  (임베딩 모델 서버: qwen3-embedding:0.6b)
#   Qdrant 실행 중 (기본 http://localhost:6333)
# =============================================================================

import sys
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.scraper import scrape_all
from data.parser import parse_all
from data.ingest_lc import ingest


def main():
    pdf_paths = sys.argv[1:]
    print("=== 1. 수집 ===")
    raw = scrape_all(pdf_paths=pdf_paths)
    print("=== 2. 파싱 ===")
    chunks = parse_all(raw)
    print(f"파싱 완료: {len(chunks)}개 청크")
    print("=== 3. 적재 (LangChain → compliance_agent_lc) ===")
    ingest(chunks)
    print(f"적재 완료: {len(chunks)}개 청크")


if __name__ == "__main__":
    main()
