# =============================================================================
# run_ingest.py
# -----------------------------------------------------------------------------
# 역할: 데이터 수집/파싱/적재 파이프라인의 진입점.
#
# 실행 방법:
#   python run_ingest.py               # 웹 데이터만 수집 (사규·법규·판례)
#   python run_ingest.py ./fss.pdf     # PDF 추가 적재
#
# 데이터 흐름:
#   scraper.py → parser.py → ingest.py
#   → VectorStoreIndex (Qdrant) + data/article_lookup.json
#
# 사전 조건:
#   ollama serve  (임베딩 모델 서버)
#   Qdrant 실행 중 (또는 USE_QDRANT=0 으로 인메모리 모드)
# =============================================================================

import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# sys.path: data/ 와 workflow/ 의 bare import(from parser import ..., 등)가
# 동작하려면 ROOT, ROOT/data, ROOT/workflow 가 모두 경로에 있어야 한다.
ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "data", ROOT / "workflow"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from data.scraper import scrape_all
from data.parser import parse_all
from data.ingest import ingest


def main():
    pdf_paths = sys.argv[1:]
    print("=== 1. 수집 ===")
    raw = scrape_all(pdf_paths=pdf_paths)
    print("=== 2. 파싱 ===")
    chunks = parse_all(raw)
    print("=== 3. 적재 ===")
    ingest(chunks)
    print(f"적재 완료: {len(chunks)}개 청크")


if __name__ == "__main__":
    main()
